from typing import Dict, Tuple

import torch

from fonts import Font, GlyphPaths


class FontCodec:
    def __init__(self, glyps: str, glyph_res: int, max_font_tokens: int, max_glyph_tokens: int) -> None:
        self.glyphs = glyps
        self.glyph_res = glyph_res
        self.max_glyph_tokens = max_glyph_tokens
        self.max_font_tokens = max_font_tokens
        self.first_system_token = glyph_res ** 2
        self.glyph_vocab = range(self.first_system_token, self.first_system_token + len(glyps))
        system_tokens = list(glyps) + [
            "moveTo",
            "lineTo",
            "qCurveTo",
            "curveTo",
            "closePath",
            "<EOS>",
            "<PAD>",
        ]
        self.mapping: Dict[str, int] = {
            token: self.first_system_token + i for i, token in enumerate(system_tokens)
        }
        self.vocab_size = glyph_res ** 2 + len(self.mapping)
        self.reverse_mapping = {value: key for key, value in self.mapping.items()}
        self.pad_token = self.mapping["<PAD>"]
        self.eos_token = self.mapping["<EOS>"]

    def encode_point(self, point: Tuple[int, int]) -> int:
        x, y = point
        assert 0 <= x < self.glyph_res, f"x={x} is out of bounds"
        assert 0 <= y < self.glyph_res, f"y={y} is out of bounds"
        return int(x + self.glyph_res * y)

    def encode_font(self, font: Font, limit: int = -1, pad_glyphs: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        # fill with padding tokens.
        tokens_size = self.max_font_tokens if limit == -1 else limit
        tokens = torch.fill(torch.zeros(tokens_size, dtype=torch.long), self.pad_token)
        idx = 0

        def next_idx():
            nonlocal idx
            idx += 1
            return idx - 1

        last_glyph_letter = list(font.glyph_paths.keys())[-1]

        for (glyph_letter, glyph_paths) in font.glyph_paths.items():
            glyph_paths: GlyphPaths
            glyph_start_idx = idx
            tokens[next_idx()] = self.mapping[glyph_letter]
            for command, data in glyph_paths:
                if command == "moveTo":
                    assert len(data) == 1, f"moveTo should have 1 point, got {len(data)}"
                    tokens[next_idx()] = self.mapping[command]
                    tokens[next_idx()] = self.encode_point(data[0])
                elif command == "lineTo":
                    assert len(data) == 1, f"lineTo should have 1 point, got {len(data)}"
                    tokens[next_idx()] = self.mapping[command]
                    tokens[next_idx()] = self.encode_point(data[0])
                elif command == "qCurveTo":
                    control_points = data
                    tokens[next_idx()] = self.mapping[command]
                    for point in control_points:
                        tokens[next_idx()] = self.encode_point(point)
                elif command == "curveTo":
                    assert len(data) == 3, f"curveTo should have 3 points, got {len(data)}"
                    control_points = data
                    tokens[next_idx()] = self.mapping[command]
                    tokens[next_idx()] = self.encode_point(control_points[0])
                    tokens[next_idx()] = self.encode_point(control_points[1])
                    tokens[next_idx()] = self.encode_point(control_points[2])
                elif command == "closePath":
                    assert len(data) == 0, f"closePath should have 0 points, got {len(data)}"
                    tokens[next_idx()] = self.mapping[command]
                else:
                    raise Exception(f'unknown command {command}')
                if idx - glyph_start_idx > self.max_glyph_tokens:
                    raise Exception(f'too many tokens for glyph {glyph_letter}')
            # Add padding for the current glyph, do not pad the last glyph so that we have enough space for EOS.
            while pad_glyphs and glyph_letter != last_glyph_letter and idx - glyph_start_idx < self.max_glyph_tokens:
                tokens[next_idx()] = self.pad_token
        tokens[next_idx()] = self.mapping["<EOS>"]
        mask = tokens.not_equal(self.pad_token)
        return tokens, mask

    def decode_point(self, token: int) -> Tuple[int, int]:
        x = int(token % self.glyph_res)
        y = int(token // self.glyph_res)
        assert 0 <= x < self.glyph_res, f"x={x} is out of bounds, token={token}"
        assert 0 <= y < self.glyph_res, f"y={y} is out of bounds, token={token}"
        return x, y

    def decode_font(self, tokens: torch.Tensor) -> Font:
        glyph_paths: Dict[str, GlyphPaths] = {}
        current_glyph_path: GlyphPaths = []

        idx = 0

        def next_idx():
            nonlocal idx
            idx += 1
            return idx - 1

        token = tokens[next_idx()].item()
        # the first token should always be a letter
        glyph_letter = self.reverse_mapping[token]
        assert glyph_letter in self.glyphs, f"first token should be a glyph, got {glyph_letter}"

        while idx < len(tokens):
            command = self.reverse_mapping[tokens[next_idx()].item()]
            if command == "<EOS>":
                glyph_paths[glyph_letter] = current_glyph_path
                return Font(glyph_paths)
            elif command == "<PAD>":
                # skip PAD tokens
                continue
            elif command in self.glyphs:
                print(f'decoding glyph {glyph_letter}')
                glyph_paths[glyph_letter] = current_glyph_path
                glyph_letter = command
                current_glyph_path = []
            elif command == "closePath":
                current_glyph_path.append((command, tuple()))
            elif command == "curveTo":
                data = tuple([self.decode_point(tokens[next_idx()].item()),
                              self.decode_point(tokens[next_idx()].item()),
                              self.decode_point(tokens[next_idx()].item())])
                current_glyph_path.append((command, data))
            elif command == "moveTo":
                data = tuple([self.decode_point(tokens[next_idx()].item())])
                current_glyph_path.append((command, data))
            elif command == "lineTo":
                data = tuple([self.decode_point(tokens[next_idx()].item())])
                current_glyph_path.append((command, data))
            elif command == "qCurveTo":
                control_points = []
                while True:
                    if tokens[idx].item() >= self.first_system_token:
                        break
                    control_point = self.decode_point(tokens[next_idx()].item())
                    control_points.append(control_point)
                current_glyph_path.append((command, tuple(control_points)))
            else:
                raise Exception(f'unknown command {command}')

        return Font(glyph_paths)
