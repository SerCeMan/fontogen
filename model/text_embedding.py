from typing import List

import torch
from transformers import BertTokenizer, BertModel


class TextEmbedder:
    def __init__(self, max_text_token_len: int, device: str):
        self.max_token_len = max_text_token_len
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
        self.bert = BertModel.from_pretrained(BERT_MODEL)
        self.device = device

    def tokenize_batch(self, text_batch: List[str]) -> [torch.Tensor, torch.Tensor]:
        encoded_input = self.tokenizer.batch_encode_plus(
            text_batch,
            add_special_tokens=True,
            max_length=self.max_token_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        ).to(self.device)
        input_ids = encoded_input['input_ids']
        attn_mask_ids = encoded_input['attention_mask']
        return input_ids, attn_mask_ids

    def embed_tokens(self, text_tokens: torch.Tensor):
        # Predict hidden states features for each layer
        text_tokens = text_tokens.to(self.bert.device)
        with torch.no_grad():
            batch_embeddings = self.bert.embeddings(text_tokens)
        # prepare for transformers:
        # -> (sequence_length, batch_size, embedding_dimension)
        return batch_embeddings

    def embed_text(self, text_batch: List[str]):
        input_ids, attn_mask_ids = self.tokenize_batch(text_batch)
        # ignore attention mask as advised by DALLE paper
        return self.embed_tokens(input_ids), attn_mask_ids


BERT_MODEL = 'google/bert_uncased_L-12_H-512_A-8'
