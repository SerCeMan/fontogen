import os
from dataclasses import dataclass, asdict
from typing import List, Any, Dict, Union, Tuple

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, random_split
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from xformers.components.attention import BlockSparseAttention
from xformers.components.attention.sparsity_config import BigBirdSparsityConfig
from xformers.components.positional_embedding import SinePositionalEmbedding
from xformers.factory import xFormerEncoderConfig, xFormer

from config import FontogenConfig
from dataset.loader import FontDataset
from font_codec import FontCodec
from model.text_embedding import BERT_MODEL


class FontEmbeddings(nn.Module):
    def __init__(self, d_model: int, font_codec: FontCodec, glyph_res: int, max_seq_len: int, device: str):
        super(FontEmbeddings, self).__init__()
        self.d_model = d_model
        self.coord_size = glyph_res
        self.max_seq_len = max_seq_len
        vocab_size = font_codec.vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=font_codec.pad_token)
        # Two extra learnable matrices
        self.coord_pad = 0
        self.coord_embedding_x = nn.Embedding(glyph_res + 1, d_model, padding_idx=self.coord_pad)
        self.coord_embedding_y = nn.Embedding(glyph_res + 1, d_model, padding_idx=self.coord_pad)
        self.first_system_token = font_codec.first_system_token
        self.positional_encoding = SinePositionalEmbedding(d_model, seq_len=max_seq_len)
        self.device = device

    def font_to_vec(self, tokens: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
        token_inds = tokens < self.first_system_token

        points = torch.zeros_like(tokens, device=self.device)
        points[token_inds] = tokens[token_inds]

        x_coords = (points % self.coord_size) + 1
        y_coords = (points // self.coord_size) + 1

        return x_coords, y_coords

    def forward(self, font_tokens: torch.Tensor):
        # x: (batch_size, seq_len)
        x_coords, y_coords = self.font_to_vec(font_tokens)

        # Embedding for the token
        token_embedding = self.embedding(font_tokens)  # (batch_size, seq_len, d_model)

        # Embedding for the coordinates
        coord_embedding_x = self.coord_embedding_x(x_coords)  # (batch_size, seq_len, d_model)
        coord_embedding_y = self.coord_embedding_y(y_coords)  # (batch_size, seq_len, d_model)

        # Final embedding
        embedding = token_embedding + coord_embedding_x + coord_embedding_y

        embedding = self.positional_encoding(embedding)

        return embedding


class FontogenDataModule(pl.LightningDataModule):
    def __init__(self, config: FontogenConfig, dataset_path: str, val_perc=0.04) -> None:
        super().__init__()
        self.config = config
        self.dataset = FontDataset(dataset_path, config.device)
        dataset_size = len(self.dataset)

        val_size = int(dataset_size * val_perc)
        train_size = dataset_size - val_size

        # split the dataset, ensure consistency
        g = torch.Generator().manual_seed(42)
        self.train_dataset, self.val_dataset = random_split(self.dataset, [train_size, val_size], generator=g)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset,
                          pin_memory=True,
                          num_workers=int(os.cpu_count() / 3),
                          # num_workers=0,
                          batch_size=self.config.batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset,
                          num_workers=int(os.cpu_count() / 3),
                          # num_workers=0,
                          batch_size=self.config.batch_size)


@dataclass
class StepLoss:
    loss: torch.Tensor


class FontogenModule(pl.LightningModule):
    def __init__(self, config: FontogenConfig) -> None:
        super().__init__()
        self.save_hyperparameters(asdict(config))
        self.config = config
        self.font_codec = FontCodec(config.glyphs, config.glyph_res, config.max_font_tokens, config.max_glyph_tokens)
        self.model = Fontogen(config, self.font_codec)
        self.bert_tokenizer_pad_token_id = 0

    def forward(self,
                text_embeddings: torch.Tensor,
                font_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        text_out, font_out = self.model(text_embeddings, font_tokens)
        return text_out, font_out

    def make_a_step(self, batch) -> StepLoss:
        font_tokens = batch['font_tokens']
        text_embeddings = batch['text_embeddings']

        text_out, font_out = self.forward(text_embeddings, font_tokens)

        B, L, D = font_out.shape
        font_loss = F.cross_entropy(font_out.reshape(B * L, D), font_tokens.view(-1),
                                    ignore_index=self.font_codec.pad_token)

        return StepLoss(font_loss)

    def training_step(self, batch: Dict[str, List[str]], batch_idx: int) -> Dict[str, torch.Tensor]:
        step_loss: StepLoss = self.make_a_step(batch)
        self.log_dict({
            "loss": step_loss.loss,
        }, batch_size=self.config.batch_size, on_step=True, prog_bar=True, logger=True)
        return {"loss": step_loss.loss}

    def validation_step(self, batch: Dict[str, List[str]], batch_idx: int) -> Dict[str, torch.Tensor]:
        step_loss: StepLoss = self.make_a_step(batch)
        self.log_dict({
            "val_loss": step_loss.loss,
        }, batch_size=self.config.batch_size, on_step=True, prog_bar=True, logger=True)
        return {"val_loss": step_loss.loss}

    def configure_optimizers(self) -> Union[torch.optim.Optimizer, Tuple[List[torch.optim.Optimizer], List[Any]]]:
        print(f'creating an optimizer with LR: {self.config.learning_rate}')
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = {
            'scheduler': get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=1000,
                num_training_steps=total_steps,
            ),
            'interval': 'step',
        }
        return [optimizer], [scheduler]


class Fontogen(nn.Module):
    def __init__(self, config: FontogenConfig, font_codec: FontCodec):
        super(Fontogen, self).__init__()
        self.font_embedding = FontEmbeddings(
            config.d_model,
            font_codec,
            config.glyph_res,
            config.max_font_tokens,
            config.device
        )
        self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
        self.text_vocab_size = self.tokenizer.vocab_size
        self.transformer = FontogenTransformer(
            config.d_model,
            config.nhead,
            config.num_layers,
            self.text_vocab_size,
            font_codec.vocab_size,
            config.max_text_tokens,
            config.max_font_tokens,
            config.max_glyph_tokens,
            config.glyph_res,
            config.device,
        )
        self.text_padding_token = self.tokenizer.pad_token_id

    def font_embeddings(self, font_tokens: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
        return self.font_embedding(font_tokens)

    def forward(self, text_embeddings, font_tokens):
        font_embeddings = self.font_embeddings(font_tokens)
        return self.transformer(text_embeddings, font_embeddings)


class FontogenTransformer(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_layers: int, text_vocab_size: int, font_vocab_size: int,
                 max_text_tokens: int, max_font_tokens: int, max_glyph_tokens: int, glyph_res: int, device: str):
        super(FontogenTransformer, self).__init__()
        self.d_model = d_model
        self.seq_len = max_text_tokens + max_font_tokens
        self.max_text_len = max_text_tokens
        # noinspection PyUnresolvedReferences
        from model.attention import GlobalLocalAttention
        # let's try to match the text size
        block_size = 16
        assert block_size == max_text_tokens
        x_config = xFormerEncoderConfig(
            num_layers=num_layers,
            dim_model=d_model,
            feedforward_config={
                "name": "MLP",
                "dropout": 0.1,
                "activation": "gelu",
                "hidden_layer_multiplier": 4,
            },
            multi_head_config={
                "num_heads": nhead,
                "attention": BlockSparseAttention(
                    num_heads=nhead,
                    layout=BigBirdSparsityConfig(
                        num_heads=nhead,
                        block_size=block_size,
                        num_global_blocks=1,
                        num_random_blocks=2,
                        num_sliding_window_blocks=int(3 * max_glyph_tokens / block_size)
                    ).make_layout(seq_len=self.seq_len),
                    block_size=block_size,
                    dropout=0.1,
                    causal=True,
                )
            },
        )
        self.transformer = xFormer(x_config)
        self.text_vocab_size = text_vocab_size
        self.svg_vocab_size = font_vocab_size
        self.to_logits = nn.Sequential(
            nn.LayerNorm(d_model),
            # Note, we use nn.Linear here instead of FusedLinear as FusedLinear makes the model not converge.
            # bias is redundant after layer norm
            nn.Linear(d_model, font_vocab_size, bias=False),
        )
        self.device = device

    def forward(self, text_embeddings, font_embeddings):
        sos_value = 0.42  # replace with your special value
        batch_size, _, _ = text_embeddings.shape
        sos_token = torch.full((batch_size, 1, self.d_model), sos_value, device=self.device)

        tgt = torch.cat([
            sos_token,
            text_embeddings,
            font_embeddings[:, :-1, :]
        ], dim=1)

        output = self.transformer(
            src=tgt,
            # in practice, the mask doesn't work, see DALLE paper :(
            # > In preliminary experiments on Conceptual Captions (Sharma et al., 2018), we found that this
            # > resulted in higher validation loss, but better performance on out-of-distribution captions.
            # encoder_input_mask=tgt_key_padding_mask,
        )

        logits = self.to_logits(output)
        text_probabilities = logits[:, :self.max_text_len, :]
        svg_probabilities = logits[:, self.max_text_len:, :]
        return text_probabilities, svg_probabilities
