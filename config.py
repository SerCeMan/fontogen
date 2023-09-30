import string
from dataclasses import dataclass

import torch


@dataclass
class FontogenConfig:
    d_model: int
    nhead: int
    device: str
    num_layers: int
    max_text_tokens: int
    max_glyph_tokens: int
    max_font_tokens: int
    glyphs: str
    glyph_res: int
    learning_rate: float
    batch_size: int


def fontogen_config() -> FontogenConfig:
    d_model = 512
    nhead = 8
    num_layers = 16
    glyphs = string.ascii_uppercase + string.digits + '.,!?'
    max_text_tokens = 16
    max_glyph_tokens = 320
    # ensure a power of two to use block sparse attention from Triton.
    max_font_tokens = 8192 - max_text_tokens
    glyph_res = 150

    learning_rate = 0.0003
    batch_size = 2  # proper output

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps else "cpu"

    return FontogenConfig(
        d_model=d_model,
        nhead=nhead,
        device=device,
        num_layers=num_layers,
        max_text_tokens=max_text_tokens,
        max_glyph_tokens=max_glyph_tokens,
        max_font_tokens=max_font_tokens,
        glyphs=glyphs,
        glyph_res=glyph_res,
        learning_rate=learning_rate,
        batch_size=batch_size,
    )
