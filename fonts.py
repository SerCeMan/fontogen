import string
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
from fontTools.cu2qu import curve_to_quadratic
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.recordingPen import RecordingPen, DecomposingRecordingPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont
from fontTools.ufoLib.glifLib import Glyph

import uharfbuzz as hb

GlyphPaths = List[Tuple[str, Tuple[Tuple[int, int]]]]
NormalizedCoords = Dict[str,float]


@dataclass
class Font:
    glyph_paths: Dict[str, GlyphPaths]


class Fonts:
    def __init__(self, glyps: str, glyph_res: int) -> None:
        self.glyphs = glyps
        self.glyph_res = glyph_res
        self.scaler = FontScaler(glyph_res)
        self.draw_funcs = hb.DrawFuncs()
        def move_to(x,y,c):
            c.append(("moveTo", ((x,y),)))
        def line_to(x,y,c):
            c.append(("lineTo", ((x,y),)))
        def cubic_to(c1x,c1y,c2x,c2y,x,y,c):
            c.append(("curveTo", ((c1x,c1y),(c2x,c2y),(x,y))))
        def quadratic_to(c1x,c1y,x,y,c):
            c.append(("qCurveTo", ((c1x,c1y),(x,y))))
        def close_path(c):
            c.append(("closePath", ()))

        self.draw_funcs.set_move_to_func(move_to)
        self.draw_funcs.set_line_to_func(line_to)
        self.draw_funcs.set_cubic_to_func(cubic_to)
        self.draw_funcs.set_quadratic_to_func(quadratic_to)
        self.draw_funcs.set_close_path_func(close_path)

    def extract_path_data_using_pen(self, font: hb.Font, char: str) -> GlyphPaths | None:
        """Extract glyph path data using the pen API."""
        gid = font.get_nominal_glyph(ord(char))
        if gid is None:
            return None
        container = []
        font.draw_glyph(gid, self.draw_funcs, container)
        return container

    def load_font(self, path: str, allow_missing: bool = False, coordinates: Optional[NormalizedCoords]=None) -> Font:
        blob = hb.Blob.from_file_path(path)
        face = hb.Face(blob)
        font = hb.Font(face)
        if coordinates is not None:
            font.set_var_coords_normalized(coordinates)
        glyph_paths = {}
        for glyph in self.glyphs:
            font_data = self.extract_path_data_using_pen(font, glyph)
            if font_data is None:
                if glyph in string.ascii_uppercase or not allow_missing:
                    # always fail if the glyph is an letter to skip wild fonts
                    raise Exception(f'character is missing in the font: {glyph}')
                else:
                    continue
            assert font_data is not None, f'font_data must not be None for glyph {glyph}'
            glyph_paths[glyph] = font_data
        return Font(glyph_paths)

    def record_glyph_paths(self, glyph_paths: Dict[str, GlyphPaths]) -> Dict[str, Glyph]:
        glyphs = {".notdef": TTGlyphPen(None).glyph()}
        for glyph_name, path_data in glyph_paths.items():
            pen = TTGlyphPen(None)
            curr_point = None
            for command, data in path_data:
                if command == "moveTo":
                    pen.moveTo(*data)
                    curr_point = data[-1]
                elif command == "lineTo":
                    pen.lineTo(*data)
                    curr_point = data[-1]
                elif command == "qCurveTo":
                    pen.qCurveTo(*data)
                    curr_point = data[-1]
                elif command == "curveTo":
                    pen.qCurveTo(*curve_to_quadratic(tuple([curr_point] + list(data)), 0.5))
                    curr_point = data[-1]
                elif command == "closePath":
                    pen.closePath()
                else:
                    raise Exception(f'Unknown command: {command}')
            glyphs[glyph_name] = pen.glyph()
        return glyphs

    def save_as_ttf(self, path: str, font: Font):
        fb = FontBuilder(self.glyph_res, isTTF=True)
        glyphs = self.record_glyph_paths(font.glyph_paths)

        cmap = {}
        glyph_order = []
        for glyph in glyphs.keys():
            if glyph != ".notdef":
                cmap[ord(glyph)] = glyph
            glyph_order.append(glyph)

        fb.setupGlyphOrder(glyph_order)
        fb.setupCharacterMap(cmap)
        fb.setupGlyf(glyphs)

        metrics = {}
        glyph_table = fb.font["glyf"]
        for glyph in glyphs.keys():
            width_boundary = self.glyph_res / 10
            glyph_width = width_boundary + (glyph_table[glyph].xMax - glyph_table[glyph].xMin)
            metrics[glyph] = (glyph_width, glyph_table[glyph].xMin)
        fb.setupHorizontalMetrics(metrics)
        # name/horiz are required by Chrome
        fb.setupNameTable(dict(
            familyName=dict(en="Fontogen"),
            styleName=dict(en="Classy"),
        ))
        fb.setupHorizontalHeader(ascent=self.glyph_res, descent=0)
        fb.setupOS2()
        fb.setupPost()
        fb.setupDummyDSIG()

        fb.save(path)


class FontScaler:
    def __init__(self, glyph_res: int) -> None:
        self.glyph_res = glyph_res

    def find_boundaries(self, font) -> Tuple[int, int]:
        min_y_min = sys.maxsize
        max_dim = 0
        for (glyph, pen_data) in font.glyph_paths.items():
            all_coords = [coord for command, data in pen_data for coord in data]
            x_coords, y_coords = zip(*all_coords)
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            width = x_max - x_min
            height = y_max - y_min
            if min_y_min > y_min:
                min_y_min = y_min
            if max_dim < max(width, height):
                max_dim = max(width, height)
        assert max_dim > 0, "max_dim should be greater than 0"
        return max_dim, min_y_min

    def scale_and_translate_path_data(self, pen_data: GlyphPaths, max_dim: int, min_y_min: int) -> GlyphPaths:
        """
        Scale the path data to fit within the target range, round to integers,
        and then translate it to make all coordinates non-negative.
        """
        target_range = self.glyph_res
        all_coords = [coord for command, data in pen_data for coord in data]
        x_coords, y_coords = zip(*all_coords)
        # apply the vertical offset from the glyph
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        y_min = min(y_min, min_y_min)
        scale_factor = target_range / max_dim
        translated_and_scaled_pen_data = []
        for command, data in pen_data:
            scaled_data = tuple([(min(round((x - x_min) * scale_factor), target_range - 1),
                                  min(round((y - y_min) * scale_factor), target_range - 1)) for x, y in data])
            translated_and_scaled_pen_data.append((command, scaled_data))
        return translated_and_scaled_pen_data

    def normalize_font(self, font: Font) -> Font:
        scaled_glyphs = {}
        max_dim, min_y_min = self.find_boundaries(font)
        for (glyph, glyph_path) in font.glyph_paths.items():
            scaled_glyphs[glyph] = self.scale_and_translate_path_data(glyph_path, max_dim, min_y_min)
        return Font(scaled_glyphs)


class FontPlotter:
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def plot_quadratic_bezier_curve(ax, control_points):
        """Plot a quadratic Bézier curve given its control points."""
        p0, p1, p2 = control_points
        t = np.linspace(0, 1, 100)
        curve = np.outer((1 - t) ** 2, p0) + np.outer(2 * (1 - t) * t, p1) + np.outer(t ** 2, p2)
        ax.plot(curve[:, 0], curve[:, 1], 'r-')

    @staticmethod
    def plot_cubic_bezier_curve(ax, control_points):
        """Plot a cubic Bézier curve given its control points."""
        p0, p1, p2, p3 = control_points
        t = np.linspace(0, 1, 100)
        curve = (
                np.outer((1 - t) ** 3, p0) +
                np.outer(3 * (1 - t) ** 2 * t, p1) +
                np.outer(3 * (1 - t) * t ** 2, p2) +
                np.outer(t ** 3, p3)
        )
        ax.plot(curve[:, 0], curve[:, 1], 'g-')

    @staticmethod
    def plot_glyph_from_pen_data(pen_data: GlyphPaths, title: str) -> None:
        """Plot glyph from pen data."""
        ax = plt.gca()
        start_point: Tuple[int, int] = None
        current_point: Tuple[int, int] = None

        for command, data in pen_data:
            if command == "moveTo":
                start_point = data[0]
                current_point = start_point
            elif command == "lineTo":
                end_point = data[0]
                ax.plot([current_point[0], end_point[0]], [current_point[1], end_point[1]], 'b-')
                current_point = end_point
            elif command == "qCurveTo":
                control_points = data
                num_points = len(control_points)
                # Process quadratic Bézier curves with pairs of control points
                for i in range(0, num_points - 1, 2):
                    if i + 2 < num_points:
                        # If there are more control points ahead, compute the implied on-curve point
                        implied_point = (round((control_points[i][0] + control_points[i + 1][0]) / 2),
                                         round((control_points[i][1] + control_points[i + 1][1]) / 2))
                        FontPlotter.plot_quadratic_bezier_curve(ax, (current_point, control_points[i], implied_point))
                        current_point = implied_point
                    else:
                        # If these are the last two control points, use the last point as on-curve
                        FontPlotter.plot_quadratic_bezier_curve(ax, (
                            current_point, control_points[i], control_points[i + 1]))
                        current_point = control_points[i + 1]
            elif command == "curveTo":
                FontPlotter.plot_cubic_bezier_curve(ax, (current_point, data[0], data[1], data[2]))
                current_point = data[2]
            elif command == "closePath":
                ax.plot([current_point[0], start_point[0]], [current_point[1], start_point[1]], 'b-')
                current_point = start_point
            else:
                raise Exception(f'unknown command {command}')

        ax.set_title(title)
        ax.set_aspect('equal', 'box')
        ax.axis('off')  # Hide axis

    def plt_font(self, font: Font):
        # Plot original and scaled glyphs side by side for comparison
        for glyph in font.glyph_paths.keys():
            plt.figure(figsize=(6, 6))
            self.plot_glyph_from_pen_data(font.glyph_paths[glyph], f"glyph={glyph}")
            plt.show()
