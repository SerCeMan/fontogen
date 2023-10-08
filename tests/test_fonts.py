import string
import unittest

from fonts import Fonts, FontScaler, FontPlotter


class TestFontLoader(unittest.TestCase):
    def test_font_saving_ttf(self):
        fonts = [
            './Roboto-Medium.ttf',
            './OpenSans-Medium.ttf',
            # TODO: learn how to deal with Dune Rise and the Nones
            # './Dune_Rise.ttf',
        ]
        for f in fonts:
            glyph_res = 160
            glyphs = string.ascii_uppercase + string.digits + '.,;:!?\'"/&+@()-_'
            fonts = Fonts(glyphs, glyph_res)
            font = fonts.load_font(f)

            scaler = FontScaler(glyph_res)
            font = scaler.normalize_font(font)

            fonts.save_as_ttf('./converted.ttf', font)
            font2 = fonts.load_font('./converted.ttf')
            if f.endswith('.otf'):
                # With OTF fonts, cubic bézier curves have to be converted to quadratic bézier curves
                # so comparison isn't possible
                continue
            print(font.glyph_paths)
            print(font2.glyph_paths)
            self.assertEqual(font.glyph_paths, font2.glyph_paths)

    def test_font_plotting_ttf(self):
        glyph_res = 160
        fonts = Fonts('ABCD', glyph_res)
        font = fonts.load_font('./OpenSans-Medium.ttf')

        scaler = FontScaler(glyph_res)
        scaled_font = scaler.normalize_font(font)

        font_plotter = FontPlotter()
        font_plotter.plt_font(font)
        font_plotter.plt_font(scaled_font)

    def test_font_plotting_otf(self):
        glyph_res = 160
        fonts = Fonts('.,;:!?\'"/&+@()-_', glyph_res)
        font = fonts.load_font('./Roboto-Medium.ttf')

        scaler = FontScaler(glyph_res)
        scaled_font = scaler.normalize_font(font)

        font_plotter = FontPlotter()
        font_plotter.plt_font(font)
        font_plotter.plt_font(scaled_font)


if __name__ == '__main__':
    unittest.main()
