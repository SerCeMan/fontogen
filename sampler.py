import os
import random
import traceback

import lightning.pytorch as pl
import torch
import torch.nn.functional as F

from config import fontogen_config, FontogenConfig
from fonts import Fonts, Font
from model.model import FontogenModule
from model.text_embedding import TextEmbedder


class SamplingCallback(pl.Callback):
    def __init__(self, config: FontogenConfig, sample_every_epoch: int, out_folder: str):
        super().__init__()
        self.config = config
        self.device = config.device
        self.out_folder = out_folder
        self.sample_every_epoch = sample_every_epoch

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if (trainer.current_epoch + 1) % self.sample_every_epoch == 0:
            model: FontogenModule = pl_module
            model.eval()
            try:
                sampler = FontogenSampler(model, self.config, self.out_folder)
                sampler.sample('bold sans', trainer.current_epoch)
            except Exception as e:
                print(f'failed to sample', e)
            model.train()


# Source: https://github.com/samxuxiang/SkexGen/blob/c38f30e8ac40aabfa2a71d6842cc585faa9b9862/model/code.py#L12
def top_k_top_p_filtering(logits: torch.Tensor, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with the highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


class FontogenSampler:
    def __init__(self, model: FontogenModule, config: FontogenConfig, out_folder: str, glyphs: str = None):
        self.model = model
        if glyphs is None:
            glyphs = config.glyphs
        self.glyphs = glyphs
        self.glyph_res = config.glyph_res
        # import string
        # self.glyphs = 'ABCDEFGHIJK'
        self.font_codec = model.font_codec
        self.max_glyph_tokens = config.max_glyph_tokens
        self.max_font_tokens = config.max_font_tokens
        self.out_folder = out_folder
        self.device = config.device
        # do not consume GPU memory for the text module
        self.text_embedder = TextEmbedder(config.max_text_tokens, self.device)

    def sample_next_token(self, logit: torch.Tensor, strategy: str = 'greedy', temperature: float = 1.0):
        max_index: torch.Tensor
        logit /= temperature
        max_index: torch.Tensor
        if strategy == 'multinomial':
            probabilities = torch.softmax(logit, dim=0)
            max_index = torch.multinomial(probabilities, num_samples=1)
        elif strategy == 'greedy':
            probabilities = torch.softmax(logit, dim=0)
            max_index = torch.argmax(probabilities, dim=0)
        elif strategy == 'topknuc':
            # Top-K and/or Nucleus Filtering
            top_k = 10
            top_p = 0.9
            logit = top_k_top_p_filtering(logit, top_k=top_k, top_p=top_p)

            probabilities = torch.softmax(logit, dim=0)
            max_index = torch.multinomial(probabilities, num_samples=1)
        else:
            raise ValueError(f'Unknown type {strategy}')
        return max_index.item()

    def sample_font(self, text_embeddings: torch.Tensor, temperature: int = None, strategy: str = None) -> Font:
        if temperature is None:
            temperature = 0.6
        if strategy is None:
            strategy = 'multinomial'
        font_tokens, _ = self.font_codec.encode_font(Font({'A': []}))
        font_tokens = font_tokens.to(self.device)
        font_tokens = torch.stack([font_tokens])
        max_length = self.max_font_tokens
        curr_token = 'A'
        tokens_per_glyph = 0
        token_idx = 1
        token_attempt = 1
        i = 1
        while i < max_length:
            if tokens_per_glyph > self.max_glyph_tokens * 2:
                if token_attempt > 5:
                    raise Exception(f'too many tokens for glyph {curr_token}')
                else:
                    print(f'too many tokens for glyph {curr_token}, trying again')
                    token_attempt += 1
                    tokens_per_glyph = 0
                    i = token_idx
                    font_tokens[0][token_idx] = self.font_codec.eos_token

            _, font_out = self.model(text_embeddings, font_tokens)

            last_predicted = font_out[0][i]  # B, S, Prob

            next_token = self.sample_next_token(last_predicted, strategy=strategy, temperature=temperature)
            if next_token in self.font_codec.glyph_vocab or next_token == self.font_codec.eos_token:
                if self.glyphs.index(curr_token) + 1 == len(self.glyphs):
                    #
                    next_token = self.font_codec.eos_token
                else:
                    curr_token = self.glyphs[self.glyphs.index(curr_token) + 1]
                    next_token = self.font_codec.mapping[str(curr_token)]
                    print(f'Starting building {self.font_codec.reverse_mapping[next_token]}')
                    tokens_per_glyph = 0
                    token_idx = i + 1
                    token_attempt = 1

            if next_token == self.font_codec.eos_token:
                break

            # put EOS at the end
            eos = font_tokens[0][i].item()
            assert eos == self.font_codec.eos_token, f'unexpected eos at position {i}'
            font_tokens[0][i] = next_token
            if i + 1 < max_length:
                font_tokens[0][i + 1] = eos
            if i == max_length - 1:
                raise ValueError('Max length reached')
            tokens_per_glyph += 1
            i += 1

        return self.font_codec.decode_font(font_tokens[0])

    def sample(self, text: str, step: int = -1, temperature: int = None, strategy: str = None) -> str:
        print(f'sampling {text}')
        out_folder = self.out_folder
        with torch.no_grad():
            text_tokens, _ = self.text_embedder.tokenize_batch([text])

            text_tokens = text_tokens.to(self.device)

            text_embeddings = self.text_embedder.embed_tokens(text_tokens)
            text_embeddings = text_embeddings.to(self.device)

            out_font: Font = self.sample_font(text_embeddings, temperature, strategy)
            step_str = "" if step < 0 else f"{step}_"
            out_path = f'{out_folder}/{step_str}{text.replace(" ", "_")}_{random.randint(0, 100000)}.ttf'
            self.try_save_font_tokens(out_path, out_font)
            return out_path

    def try_save_font_tokens(self, font_path: str, font: Font):
        print(f'saving font as {font_path}')
        try:
            fonts = Fonts(self.glyphs, self.glyph_res)
            os.makedirs(os.path.dirname(font_path), exist_ok=True)
            fonts.save_as_ttf(font_path, font)
        except Exception as e:
            print(f'failed to save font', e)


def create_sampler(
        out_folder: str,
        glyphs: str = None,
        checkpoint_path: str = 'models/fontogen_78958.ckpt') -> FontogenSampler:
    config = fontogen_config()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps else "cpu"
    model = FontogenModule.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        config=config,
    )
    model.eval()
    return FontogenSampler(model, config, out_folder, glyphs)


if __name__ == '__main__':
    texts = [
        'bold sans',
        'canvalove sans',
        'basic, serif, new times',
        'Beauty, Script, Calligraphy',
        'techno, sci-fi, extrabold',
        'handwritten, beauty, script',
        'soviet, block, russian',
        'serif, newspaper, news',
        'tech blog, perfect',
        'fancy, horror, scary, halloween',
        'horror scary',
        'anime, Fancy, Horror',
        'bold, cyrrilic',
        'basic,serif,harry, potter',
        'taco, bell',
        'celtic, magic',
        'elvish, runes',
        'fancy, kids',
        'Script,Handwritten,Painter',
        'medieval, gothic',
        'newspaper, typewriter',
        'comic, superman',
        'sci-fi, futuristic',
        '_',
        '',
    ]

    sampler = create_sampler('training/samples')
    for (ind, text) in enumerate(texts):
        for i in range(2):
            try:
                sampler.sample(text, -1, strategy='multinomial')
            except Exception as e:
                traceback.print_exc()
                print(f'Failed to generate font for {text}', e)
