import string
import unittest

from font_codec import FontCodec
from fonts import Fonts, FontScaler


class TestFontCodec(unittest.TestCase):
    def test_font_coding_decoding_ttf(self):
        glyph_res = 150
        glyphs = string.ascii_uppercase + string.digits + '.,;:!?\'"/&+@()-_'
        scaler = FontScaler(glyph_res)
        fonts = Fonts(glyphs, glyph_res)
        font = fonts.load_font('./OpenSans-Medium.ttf')
        font = scaler.normalize_font(font)

        codec = FontCodec(glyphs, glyph_res, 200 * len(glyphs), 200)
        tokens, mask = codec.encode_font(font)
        decoded_font = codec.decode_font(tokens)
        print(font)
        print(decoded_font)
        self.assertEqual(font, decoded_font)

    def test_font_coding_decoding_otf(self):
        glyph_res = 150
        glyphs = string.ascii_uppercase + string.digits + '.,;:!?\'"/&+@()-_'
        scaler = FontScaler(glyph_res)
        fonts = Fonts(glyphs, glyph_res)
        font = fonts.load_font('./Roboto-Medium.ttf')
        font = scaler.normalize_font(font)

        codec = FontCodec(glyphs, glyph_res, 200 * len(glyphs), 200)
        tokens, mask = codec.encode_font(font)
        decoded_font = codec.decode_font(tokens)
        print(font)
        print(decoded_font)
        self.assertEqual(font, decoded_font)


if __name__ == '__main__':
    unittest.main()
