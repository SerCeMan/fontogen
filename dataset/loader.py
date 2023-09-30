from dataclasses import dataclass
from typing import List

import torch
from torch.utils.data import Dataset


@dataclass
class FontDatasetEntry:
    font_tokens: torch.Tensor
    text_tokens: torch.Tensor
    text_embeddings: torch.Tensor


class FontDataset(Dataset):
    def __init__(self, ds_file, device):
        self.data: List[FontDatasetEntry] = torch.load(ds_file)
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        return {
            'font_tokens': row.font_tokens,
            'text_tokens': row.text_tokens,
            'text_embeddings': row.text_embeddings,
        }
