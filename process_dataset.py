import os

import torch

from config import fontogen_config
from dataset.loader import FontDatasetEntry
from font_codec import FontCodec
from fonts import FontScaler, Fonts
from model.text_embedding import TextEmbedder


def main():
    device = 'cpu'
    config = fontogen_config()
    glyphs = config.glyphs
    glyph_res = config.glyph_res
    max_text_length = config.max_text_tokens
    max_glyph_tokens = config.max_glyph_tokens
    max_font_tokens = config.max_font_tokens
    dataset_path = 'example/processed_dataset/fonts.ds'

    scaler = FontScaler(glyph_res)
    fonts = Fonts(glyphs, glyph_res)
    codec = FontCodec(glyphs, glyph_res, max_font_tokens, max_glyph_tokens)
    text_embedder = TextEmbedder(max_text_length, device)

    dataset = []
    for dirpath, dirnames, filenames in os.walk('example/dataset'):
        for filename in filenames:
            if filename.endswith('.ttf'):
                filepath = os.path.join(dirpath, filename)
                font = fonts.load_font(filepath)
                font = scaler.normalize_font(font)
                with open(filepath.replace('.ttf', '.txt'), 'r') as file:
                    text = file.read()
                text_tokens, _ = text_embedder.tokenize_batch([text])
                text_embeddings = text_embedder.embed_tokens(text_tokens)
                font_tokens, _ = codec.encode_font(font)
                dataset.append(FontDatasetEntry(
                    font_tokens,
                    text_tokens[0],
                    text_embeddings[0],
                ))
    torch.save(dataset, dataset_path)

if __name__ == '__main__':
    main()