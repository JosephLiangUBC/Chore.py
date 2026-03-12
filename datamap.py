"""
datamap.py
----------
Python port of DataMapper.java – the ``--map`` rendering pipeline.

Renders MWT worm tracks onto a 2-D spatial canvas, colouring each
position by a scalar quantity (speed, length, time, etc.).

Class hierarchy
---------------
ColorMapper          (base → grayscale)
  RainbowMapper
  SunsetMapper
  SpatterMapper

Backgrounder         (base → transparent/black)
  GreenGrounder
  WhiteGrounder
  ImageGrounder
  DimImageGrounder

DotPainter           (base → single pixel)
  CirclePainter
  SpotPainter
  IdentityPainter
  FramePainter
  ValuePainter
  LinePainter

ValueSource          (base → uniform 0.5)
  ValueIdentity      (no value, uses 0.5)
  ValueValue         (uses a Dance quantity array)

ViewRequest          – holds all render parameters for one frame

DataMapper           – main orchestrator: render(dances, …) → PIL.Image
"""

from __future__ import annotations
import math
import io
import sys
import warnings
import itertools
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import (Dict, List, Optional, Tuple, Union,
                    Callable, TYPE_CHECKING)

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False
    warnings.warn("Pillow not installed – DataMapper.render() will be unavailable. "
                  "Install with: pip install Pillow")

if TYPE_CHECKING:
    from .dance import Dance
    from .choreography import Choreography


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _d2i(v: float) -> int:
    """mirrors ColorMapper.d2i: floor(v * 255.99999999)"""
    return int(math.floor(v * 255.99999999))


def _argb_to_rgba(argb: int) -> Tuple[int, int, int, int]:
    """Convert Java ARGB int to (R, G, B, A) tuple."""
    a = (argb >> 24) & 0xFF
    r = (argb >> 16) & 0xFF
    g = (argb >>  8) & 0xFF
    b =  argb        & 0xFF
    return r, g, b, a


def _rgba(r: int, g: int, b: int, a: int = 255) -> Tuple[int, int, int, int]:
    return r, g, b, a


# Sentinel ARGB colors (decoded from bytecode)
_SENTINEL_NAN      = _argb_to_rgba(0xFF800080)   # (128, 0, 128, 255) – purple
_SENTINEL_UNDER    = _argb_to_rgba(0xFFB080FF)   # (176, 128, 255, 255) – lavender
_SENTINEL_OVER     = _argb_to_rgba(0xFFFF80B0)   # (255, 128, 176, 255) – pink
_SENTINEL_SKIP     = None                         # -1 from Rainbow = transparent/skip


# ---------------------------------------------------------------------------
# Color Mappers
# ---------------------------------------------------------------------------

class ColorMapper:
    """
    Base grayscale color mapper.  Maps a normalised value v ∈ [0, 1] to an
    RGBA tuple.  Mirrors DataMapper$ColorMapper.

    Values outside [0, 1] or NaN use fixed sentinel colors.
    """

    NAME = "Grayscale"

    # Sentinels for grayscale base
    _NAN_RGBA   = _argb_to_rgba(0xFFFF00FF)   # magenta
    _UNDER_RGBA = _argb_to_rgba(0xFF0000FF)   # blue
    _OVER_RGBA  = _argb_to_rgba(0xFFFF0000)   # red

    def map_color(self, v: float) -> Optional[Tuple[int, int, int, int]]:
        """
        Return (R, G, B, A) for normalised value v, or None to skip.
        """
        if math.isnan(v):
            return self._NAN_RGBA
        if v < 0.0:
            return self._UNDER_RGBA
        if v > 1.0:
            return self._OVER_RGBA
        i = _d2i(v)
        return i, i, i, 255

    def map_array(self, values: np.ndarray) -> np.ndarray:
        """
        Vectorised color mapping.  Returns (N, 4) uint8 RGBA array.
        NaN / out-of-range entries get their sentinel colors.
        """
        n = len(values)
        rgba = np.zeros((n, 4), dtype=np.uint8)
        for i, v in enumerate(values):
            c = self.map_color(float(v))
            if c is not None:
                rgba[i] = c
        return rgba

    def __str__(self) -> str:
        return self.NAME


class RainbowMapper(ColorMapper):
    """
    Rainbow color map: red → yellow → green → cyan → blue → magenta.
    Mirrors DataMapper$RainbowMapper.

    Exact formula decoded from bytecode:
        r = min(1, max(0, 2-6v) + max(0, 6v-4))
        g = min(1, max(0, 2-6|v-1/3|))
        b = min(1, max(0, 2-6|v-2/3|))
    """

    NAME = "Rainbow"
    _NAN_RGBA   = None   # skip (−1 in Java = transparent)
    _UNDER_RGBA = None
    _OVER_RGBA  = None

    def map_color(self, v: float) -> Optional[Tuple[int, int, int, int]]:
        if math.isnan(v) or v < 0.0 or v > 1.0:
            return None  # skip
        r = min(1.0, max(0.0, 2 - 6*v) + max(0.0, 6*v - 4))
        g = min(1.0, max(0.0, 2 - 6*abs(v - 1/3)))
        b = min(1.0, max(0.0, 2 - 6*abs(v - 2/3)))
        return _d2i(r), _d2i(g), _d2i(b), 255

    def map_array(self, values: np.ndarray) -> np.ndarray:
        v = np.asarray(values, dtype=float)
        r = np.clip(np.maximum(0, 2 - 6*v) + np.maximum(0, 6*v - 4), 0, 1)
        g = np.clip(2 - 6*np.abs(v - 1/3), 0, 1)
        b = np.clip(2 - 6*np.abs(v - 2/3), 0, 1)
        rgba = np.zeros((len(v), 4), dtype=np.uint8)
        valid = np.isfinite(v) & (v >= 0) & (v <= 1)
        rgba[valid, 0] = (r[valid] * 255.9999).astype(np.uint8)
        rgba[valid, 1] = (g[valid] * 255.9999).astype(np.uint8)
        rgba[valid, 2] = (b[valid] * 255.9999).astype(np.uint8)
        rgba[valid, 3] = 255
        return rgba


class SunsetMapper(ColorMapper):
    """
    Sunset color map: deep blue → pink/magenta → orange.
    Mirrors DataMapper$SunsetMapper.

    Formula (decoded from bytecode):
        R = min(v, 0.5) * 2 * 255
        G = 128
        B = min(1−v, 0.5) * 2 * 255
    Sentinels:
        NaN   → (128, 0, 128)   – purple
        v < 0 → (176, 128, 255) – lavender
        v > 1 → (255, 128, 176) – pink-red
    """

    NAME = "Sunset"
    _NAN_RGBA   = _SENTINEL_NAN
    _UNDER_RGBA = _SENTINEL_UNDER
    _OVER_RGBA  = _SENTINEL_OVER

    def map_color(self, v: float) -> Optional[Tuple[int, int, int, int]]:
        if math.isnan(v):
            return self._NAN_RGBA
        if v < 0.0:
            return self._UNDER_RGBA
        if v > 1.0:
            return self._OVER_RGBA
        r = int(min(v, 0.5) * 2 * 255)
        g = 128
        b = int(min(1 - v, 0.5) * 2 * 255)
        return r, g, b, 255

    def map_array(self, values: np.ndarray) -> np.ndarray:
        v = np.asarray(values, dtype=float)
        rgba = np.zeros((len(v), 4), dtype=np.uint8)
        valid = np.isfinite(v) & (v >= 0) & (v <= 1)
        rgba[valid, 0] = (np.minimum(v[valid], 0.5) * 510).astype(np.uint8)
        rgba[valid, 1] = 128
        rgba[valid, 2] = (np.minimum(1 - v[valid], 0.5) * 510).astype(np.uint8)
        rgba[valid, 3] = 255
        # Sentinels
        for i in np.where(np.isnan(v))[0]:
            rgba[i] = self._NAN_RGBA
        for i in np.where(np.isfinite(v) & (v < 0))[0]:
            rgba[i] = self._UNDER_RGBA
        for i in np.where(np.isfinite(v) & (v > 1))[0]:
            rgba[i] = self._OVER_RGBA
        return rgba


class SpatterMapper(ColorMapper):
    """
    Spatter / random-scatter color map: produces many distinct hues so
    individual worms/paths stand out.
    Mirrors DataMapper$SpatterMapper.

    Uses irrational divisors (π, e, √2) to produce pseudo-random but
    deterministic colors.
    """

    NAME = "Spatter"

    def __init__(self, entries: int = 256):
        self.entries = max(1, entries)

    def _frac(self, x: float) -> float:
        return x - math.floor(x)

    def map_color(self, v: float) -> Optional[Tuple[int, int, int, int]]:
        if math.isnan(v) or v < 0.0 or v > 1.0:
            return None
        n = float(self.entries)
        rf = self._frac(v * n / math.pi)
        bf = self._frac(v * n / math.e)
        gf = self._frac(v * n / math.sqrt(2))
        brightness = 1.0 / max(0.1, rf*rf + bf*bf + gf*gf)
        r = _d2i(rf * brightness)
        g = _d2i(gf * brightness)
        b = _d2i(bf * brightness)
        return min(255, r), min(255, g), min(255, b), 255


# ---------------------------------------------------------------------------
# Background renderers
# ---------------------------------------------------------------------------

class Backgrounder:
    """
    Base transparent background renderer.
    Mirrors DataMapper$Backgrounder.
    """

    NAME = "None"

    def highlight(self) -> Tuple[int, int, int, int]:
        return 255, 255, 255, 255   # white

    def midlight(self) -> Tuple[int, int, int, int]:
        return 128, 128, 128, 255   # grey

    def background_color(self) -> Tuple[int, int, int, int]:
        return 0, 0, 0, 0   # transparent

    def fill_image(self, img: "Image.Image") -> None:
        """Fill the PIL image with this background."""
        img.paste(_rgba_to_pil(self.background_color()), [0, 0, img.width, img.height])

    def __str__(self) -> str:
        return self.NAME


class BlackBackgrounder(Backgrounder):
    """Opaque black background."""
    NAME = "Black"

    def background_color(self) -> Tuple[int, int, int, int]:
        return 0, 0, 0, 255


class GreenGrounder(Backgrounder):
    """Dark-green background.  Mirrors DataMapper$Greengrounder."""
    NAME = "Green"

    def background_color(self) -> Tuple[int, int, int, int]:
        return 0, 64, 0, 255

    def highlight(self) -> Tuple[int, int, int, int]:
        return 255, 255, 255, 255

    def midlight(self) -> Tuple[int, int, int, int]:
        return 0, 128, 0, 255


class WhiteGrounder(Backgrounder):
    """White background.  Mirrors DataMapper$Whitegrounder."""
    NAME = "White"

    def background_color(self) -> Tuple[int, int, int, int]:
        return 255, 255, 255, 255

    def highlight(self) -> Tuple[int, int, int, int]:
        return 0, 0, 0, 255

    def midlight(self) -> Tuple[int, int, int, int]:
        return 160, 160, 160, 255

class ImageGrounder(Backgrounder):
    """
    Uses a PNG frame as the background image.
    Mirrors DataMapper$Imagegrounder.

    Parameters
    ----------
    image_path : path to the background PNG (or list of paths for time series)
    """
    NAME = "Image"

    def __init__(self, image_path: Union[str, Path,
                                         List[Union[str, Path]]] = None,
                 dim: bool = False):
        self._paths: List[Path] = []
        self._dim = dim
        if image_path is not None:
            if isinstance(image_path, (str, Path)):
                self._paths = [Path(image_path)]
            else:
                self._paths = [Path(p) for p in image_path]
        self._cache: Optional["Image.Image"] = None
        self._last_path_idx: int = -1

    def get_image_at(self, t: float) -> Optional["Image.Image"]:
        """Return the background image at time *t* (seconds)."""
        if not self._paths:
            return None
        if not _PIL_AVAILABLE:
            return None
        idx = min(int(t), len(self._paths) - 1)
        idx = max(0, idx)
        if idx != self._last_path_idx:
            try:
                self._cache = Image.open(self._paths[idx]).convert("RGBA")
                if self._dim:
                    arr = np.array(self._cache, dtype=float)
                    arr[:, :, :3] *= 0.4
                    self._cache = Image.fromarray(arr.astype(np.uint8))
                self._last_path_idx = idx
            except Exception:
                self._cache = None
        return self._cache

    def fill_image(self, img: "Image.Image",
                   t: float = 0.0) -> None:
        bg = self.get_image_at(t)
        if bg is None:
            img.paste((0, 0, 0, 255), [0, 0, img.width, img.height])
        else:
            bg_resized = bg.resize(img.size, Image.BILINEAR)
            img.paste(bg_resized)

    def background_color(self) -> Tuple[int, int, int, int]:
        return 0, 0, 0, 255


class DimImageGrounder(ImageGrounder):
    """Dimmed image background.  Mirrors DataMapper$Dimimagegrounder."""
    NAME = "Dim Image"

    def __init__(self, image_path=None):
        super().__init__(image_path, dim=True)

    def highlight(self) -> Tuple[int, int, int, int]:
        return 220, 220, 220, 255

    def midlight(self) -> Tuple[int, int, int, int]:
        return 80, 80, 80, 255


def _rgba_to_pil(c: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """PIL expects (R,G,B,A)."""
    return c


# ---------------------------------------------------------------------------
# Value sources
# ---------------------------------------------------------------------------

class ValueSource:
    """
    Maps a (Dance, frame_index) pair to a scalar value in [0, 1].
    Mirrors DataMapper$ValueSource.
    """

    def value_at(self, dance: "Dance", frame_idx: int) -> float:
        return 0.5   # base: constant midpoint

    def __str__(self) -> str:
        return "Constant"


class ValueIdentity(ValueSource):
    """No data – all points get value 0.5 (mid-range color)."""
    def value_at(self, dance: "Dance", frame_idx: int) -> float:
        return 0.5


class ValueValue(ValueSource):
    """
    Returns per-frame values from a precomputed quantity array.
    Mirrors DataMapper$ValueValue.

    Parameters
    ----------
    quantity_fn : callable (Dance) → np.ndarray, evaluated lazily
    v_min, v_max : normalisation range; if None, use per-worm min/max
    """

    def __init__(self,
                 quantity_fn: Callable[["Dance"], np.ndarray],
                 v_min: Optional[float] = None,
                 v_max: Optional[float] = None):
        self._fn     = quantity_fn
        self._v_min  = v_min
        self._v_max  = v_max
        self._cache: Dict[int, np.ndarray] = {}
        self._norm:  Dict[int, Tuple[float, float]] = {}

    def _get_array(self, dance: "Dance") -> np.ndarray:
        if dance.ID not in self._cache:
            self._cache[dance.ID] = self._fn(dance)
        return self._cache[dance.ID]

    def _get_norm(self, dance: "Dance") -> Tuple[float, float]:
        if dance.ID not in self._norm:
            arr = self._get_array(dance)
            finite = arr[np.isfinite(arr)]
            if len(finite) == 0:
                self._norm[dance.ID] = (0.0, 1.0)
            else:
                lo = self._v_min if self._v_min is not None else float(finite.min())
                hi = self._v_max if self._v_max is not None else float(finite.max())
                self._norm[dance.ID] = (lo, hi if hi != lo else lo + 1.0)
        return self._norm[dance.ID]

    def value_at(self, dance: "Dance", frame_idx: int) -> float:
        arr = self._get_array(dance)
        if frame_idx >= len(arr):
            return float('nan')
        raw = float(arr[frame_idx])
        lo, hi = self._get_norm(dance)
        return (raw - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Dot / path painters
# ---------------------------------------------------------------------------

class DotPainter:
    """
    Base single-pixel dot painter.
    By default it is invisible (`alpha=0`), which makes it usable as a
    deliberate no-op painter.
    Mirrors DataMapper$DotPainter.
    """
    NAME = "Dot"

    def __init__(self, alpha: int = 0):
        self.alpha = max(0, min(255, int(alpha)))

    def paint(self, draw: "ImageDraw.Draw",
              x: float, y: float,
              rgba: Tuple[int, int, int, int],
              frame_idx: int,
              value: float,
              dance: "Dance") -> None:
        """Paint one position onto the PIL drawing context."""
        if self.alpha <= 0:
            return
        px, py = int(round(x)), int(round(y))
        draw.point((px, py), fill=(rgba[0], rgba[1], rgba[2], self.alpha))


class CirclePainter(DotPainter):
    """
    Filled circle of fixed diameter.
    Mirrors DataMapper$CirclePainter.
    """
    NAME = "Circle"

    def __init__(self, diameter: float = 4.0):
        self.diameter = diameter

    def paint(self, draw, x, y, rgba, frame_idx, value, dance):
        r = self.diameter / 2
        draw.ellipse([x - r, y - r, x + r, y + r], fill=rgba)


class SpotPainter(CirclePainter):
    """
    Ellipse whose size is proportional to the scalar value.
    Mirrors DataMapper$SpotPainter.
    """
    NAME = "Spot"

    def __init__(self, max_diameter: float = 8.0):
        super().__init__(max_diameter)

    def paint(self, draw, x, y, rgba, frame_idx, value, dance):
        if math.isnan(value):
            return
        r = (self.diameter / 2) * max(0.0, min(1.0, value))
        if r < 0.5:
            r = 0.5
        draw.ellipse([x - r, y - r, x + r, y + r], fill=rgba)


class IdentityPainter(CirclePainter):
    """
    Circle + worm ID text.
    Mirrors DataMapper$IdentityPainter.
    """
    NAME = "ID"

    def __init__(self, diameter: float = 6.0):
        super().__init__(diameter)

    def label_text(self, dance: "Dance", frame_idx: int) -> str:
        return str(dance.ID)

    def paint(self, draw, x, y, rgba, frame_idx, value, dance):
        super().paint(draw, x, y, rgba, frame_idx, value, dance)
        txt = self.label_text(dance, frame_idx)
        # Outline text in background color for legibility
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            draw.text((x + dx, y + dy), txt, fill=(0, 0, 0, 255))
        draw.text((x, y), txt, fill=rgba)


class FramePainter(IdentityPainter):
    """
    Circle + frame number.
    Mirrors DataMapper$FramePainter.
    """
    NAME = "Frame"

    def label_text(self, dance: "Dance", frame_idx: int) -> str:
        if frame_idx < len(dance.frames):
            return str(int(dance.frames[frame_idx]))
        return str(frame_idx)


class ValuePainter(IdentityPainter):
    """
    Circle + numeric value text.
    Mirrors DataMapper$ValuePainter.
    """
    NAME = "Value"

    def label_text(self, dance: "Dance", frame_idx: int) -> str:
        return ""  # set externally via value

    def paint(self, draw, x, y, rgba, frame_idx, value, dance):
        CirclePainter.paint(self, draw, x, y, rgba, frame_idx, value, dance)
        if not math.isnan(value):
            txt = f"{value:.2f}"
            draw.text((x, y), txt, fill=rgba)


class LinePainter(DotPainter):
    """
    Connected path painter: draws line segments between consecutive positions,
    arc segments where the path curves, and a dwell circle when the worm
    is nearly stationary. Direction arrowheads are added at intervals.

    Mirrors DataMapper$LinePainter (the most complex painter).

    Parameters
    ----------
    backup       : fallback DotPainter used for isolated points
    line_width   : stroke width (pixels)
    show_arrows  : draw arrowhead at every *arrow_interval* frames
    """

    NAME = "Line"

    DWELL_RADIUS_FACTOR = 0.35    # mirrors DataMapper.DWELL_RADIUS_FACTOR
    BEARING_SIZE        = 5.0     # mirrors DataMapper.BEARING_SIZE

    def __init__(self,
                 backup: Optional[DotPainter] = None,
                 line_width: float = 1.5,
                 show_arrows: bool = True,
                 arrow_interval: int = 10,
                 dwell_speed_threshold: float = 0.01):
        self._backup   = backup or CirclePainter(3.0)
        self.line_width = line_width
        self.show_arrows = show_arrows
        self.arrow_interval = arrow_interval
        self.dwell_speed_threshold = dwell_speed_threshold

        # State carried between consecutive paint() calls for one dance
        self._prev_x:   Optional[float] = None
        self._prev_y:   Optional[float] = None
        self._prev_rgba: Optional[Tuple] = None
        self._dance_id: int = -1

    def reset(self) -> None:
        self._prev_x = self._prev_y = self._prev_rgba = None
        self._dance_id = -1

    def paint(self, draw, x, y, rgba, frame_idx, value, dance):
        # Reset state when switching to a different worm
        if dance.ID != self._dance_id:
            self.reset()
            self._dance_id = dance.ID

        if self._prev_x is None:
            self._backup.paint(draw, x, y, rgba, frame_idx, value, dance)
        else:
            # Draw line from previous to current position
            lw = max(1, int(round(self.line_width)))
            draw.line(
                [(self._prev_x, self._prev_y), (x, y)],
                fill=rgba,
                width=lw
            )

            # Arrowhead every arrow_interval frames
            if self.show_arrows and (frame_idx % self.arrow_interval == 0):
                self._draw_arrowhead(draw, x, y, rgba, dance, frame_idx)

        self._prev_x = x
        self._prev_y = y
        self._prev_rgba = rgba

    def _draw_arrowhead(self, draw, x, y, rgba, dance, frame_idx):
        """Draw a small triangle arrowhead aligned to the bearing."""
        if frame_idx >= len(dance.bearing):
            return
        bx = float(dance.bearing[frame_idx, 0])
        by = float(dance.bearing[frame_idx, 1])
        sz = self.BEARING_SIZE
        # Arrowhead tip at (x, y), wings perpendicular
        ax =  by * sz * 0.5
        ay = -bx * sz * 0.5
        tip_x = x + bx * sz
        tip_y = y + by * sz
        draw.polygon(
            [(int(tip_x), int(tip_y)),
             (int(x + ax), int(y + ay)),
             (int(x - ax), int(y - ay))],
            fill=rgba
        )


# ---------------------------------------------------------------------------
# Viewport / view request
# ---------------------------------------------------------------------------

@dataclass
class ViewRequest:
    """
    Describes what to render: spatial viewport, time slice, render options.
    Mirrors DataMapper$ViewRequest.

    Parameters
    ----------
    center_x, center_y : viewport centre in data (pixel) coordinates
    width_px, height_px: output image dimensions
    pixel_size         : data pixels per image pixel (zoom factor)
    t_at               : reference time (seconds)
    t_range            : (t_start, t_end) time window to render
    color_mapper       : ColorMapper instance
    backgrounder       : Backgrounder instance
    dot_painter        : DotPainter instance
    value_source       : ValueSource instance
    """
    center_x:    float = 0.0
    center_y:    float = 0.0
    width_px:    int   = 800
    height_px:   int   = 600
    pixel_size:  float = 1.0   # data px per image px

    t_at:        float = 0.0
    t_range:     Tuple[float, float] = (0.0, float('inf'))

    color_mapper:  ColorMapper  = field(default_factory=ColorMapper)
    backgrounder:  Backgrounder = field(default_factory=BlackBackgrounder)
    dot_painter:   DotPainter   = field(default_factory=DotPainter)
    value_source:  ValueSource  = field(default_factory=ValueIdentity)
    show_paths:    bool = False
    show_history_morphology: bool = False
    show_crosshairs: bool = False
    show_fallback_glyphs: bool = False
    show_body_details: bool = False

    def data_to_image(self, dx: float, dy: float) -> Tuple[float, float]:
        """Convert data coordinates to image pixel coordinates."""
        ix = (dx - self.center_x) / self.pixel_size + self.width_px  / 2.0
        iy = (dy - self.center_y) / self.pixel_size + self.height_px / 2.0
        return ix, iy

    def image_to_data(self, ix: float, iy: float) -> Tuple[float, float]:
        """Convert image pixel coordinates to data coordinates."""
        dx = (ix - self.width_px  / 2.0) * self.pixel_size + self.center_x
        dy = (iy - self.height_px / 2.0) * self.pixel_size + self.center_y
        return dx, dy

    @staticmethod
    def auto_bounds(dances: Dict[int, "Dance"],
                    width_px: int = 800, height_px: int = 600,
                    border: float = 0.05) -> "ViewRequest":
        """
        Create a ViewRequest that fits all worm tracks.
        *border* is the fractional padding around the data bounds.
        """
        all_x = np.concatenate([d.centroid[:, 0] for d in dances.values()])
        all_y = np.concatenate([d.centroid[:, 1] for d in dances.values()])
        all_x = all_x[np.isfinite(all_x)]
        all_y = all_y[np.isfinite(all_y)]
        if len(all_x) == 0:
            return ViewRequest(width_px=width_px, height_px=height_px)
        cx = (all_x.max() + all_x.min()) / 2
        cy = (all_y.max() + all_y.min()) / 2
        data_w = (all_x.max() - all_x.min()) * (1 + 2*border) or 100
        data_h = (all_y.max() - all_y.min()) * (1 + 2*border) or 100
        # Choose pixel_size so data fits in the image
        ps = max(data_w / width_px, data_h / height_px)
        t_min = min(float(d.times[0])  for d in dances.values() if len(d.times) > 0)
        t_max = max(float(d.times[-1]) for d in dances.values() if len(d.times) > 0)
        return ViewRequest(
            center_x=cx, center_y=cy,
            width_px=width_px, height_px=height_px,
            pixel_size=max(ps, 0.1),
            t_at=t_min,
            t_range=(t_min, t_max),
        )


# ---------------------------------------------------------------------------
# DataMapper — main renderer
# ---------------------------------------------------------------------------

class DataMapper:
    """
    Full Python port of DataMapper.java – the ``--map`` rendering pipeline.

    Renders the spatial trajectory of all worms onto a 2-D canvas,
    colouring each position by an arbitrary scalar quantity.

    Parameters
    ----------
    mm_per_pixel : spatial calibration used when converting positions

    Usage
    -----
    >>> dm = DataMapper()
    >>> # Configure with built-in helpers:
    >>> img = dm.render(
    ...     dances,
    ...     quantity=lambda d: d.quantityIsSpeed(0.025, 0.5),
    ...     color_mapper=ColorMapper(),
    ...     backgrounder=BlackBackgrounder(),
    ...     dot_painter=DotPainter(),
    ...     width_px=1024, height_px=768,
    ... )
    >>> img.save("map.png")

    Or for a single time-point snapshot:
    >>> img = dm.render_at_time(dances, t=30.0, ...)
    """

    # Constants (mirrors DataMapper.java static fields)
    ANTIALIASING_FACTOR  = 2.0
    AREA_FRACTION_OPT    = 0.25
    BORDER_FRACTION      = 0.05
    TIME_TOLERANCE       = 0.1
    BEARING_SIZE         = 5.0
    OUTLINE_SIZE         = 0.5

    def __init__(self, mm_per_pixel: float = 0.025):
        if not _PIL_AVAILABLE:
            raise RuntimeError(
                "Pillow is required for DataMapper. "
                "Install with: pip install Pillow"
            )
        self.mm_per_pixel = mm_per_pixel

    # ------------------------------------------------------------------
    # Main render entry points
    # ------------------------------------------------------------------

    def render(self,
               dances: Dict[int, "Dance"],
               quantity: Optional[Union[str, Callable]] = None,
               v_min: Optional[float] = None,
               v_max: Optional[float] = None,
               color_mapper: Optional[ColorMapper] = None,
               backgrounder: Optional[Backgrounder] = None,
               dot_painter:  Optional[DotPainter]   = None,
               width_px: int = 800,
               height_px: int = 600,
               t_start: Optional[float] = None,
               t_end:   Optional[float] = None,
               t_at: Optional[float] = None,
               show_paths: bool = False,
               show_history_morphology: bool = False,
               show_crosshairs: bool = False,
               show_fallback_glyphs: bool = False,
               show_body_details: bool = False,
               view: Optional[ViewRequest] = None,
               ) -> "Image.Image":
        """
        Render all worm tracks onto a single composite image.

        Parameters
        ----------
        dances        : mapping of worm_id → Dance
        quantity      : what to colour by:
                        - None or 'time' → time (default)
                        - str code matching chore.OUTPUT_CODES
                        - callable(Dance) → np.ndarray
        v_min, v_max  : normalisation range (None = per-worm auto)
        color_mapper  : ColorMapper (default: grayscale)
        backgrounder  : Backgrounder (default: black)
        dot_painter   : DotPainter (default: DotPainter)
        width_px      : output image width
        height_px     : output image height
        t_start, t_end: time window (None = whole track)
        t_at          : current time used for morphology emphasis
        show_paths    : whether to render centroid/path history
        show_history_morphology : whether to render spine/outline history
        show_crosshairs : whether to render centroid crosshairs
        show_fallback_glyphs : whether to render bearing/crosshair fallback glyphs
        show_body_details : whether to render spine/midline overlays on silhouettes
        view          : ViewRequest (overrides all spatial parameters)

        Returns
        -------
        PIL.Image.Image (RGBA)
        """
        if not dances:
            return Image.new("RGBA", (width_px, height_px))

        # Build value source
        value_source = self._make_value_source(quantity, v_min, v_max, dances)

        # Defaults
        if color_mapper is None:
            color_mapper = ColorMapper()
        if backgrounder is None:
            backgrounder = BlackBackgrounder()
        if dot_painter is None:
            dot_painter = DotPainter()

        # Build or use ViewRequest
        if view is None:
            t0 = t_start if t_start is not None else min(
                float(d.times[0]) for d in dances.values() if len(d.times) > 0
            )
            t1 = t_end if t_end is not None else max(
                float(d.times[-1]) for d in dances.values() if len(d.times) > 0
            )
            view = ViewRequest.auto_bounds(dances, width_px, height_px,
                                           self.BORDER_FRACTION)
            view.t_range = (t0, t1)
            view.t_at = t1 if t_at is None else float(t_at)
            view.color_mapper = color_mapper
            view.backgrounder = backgrounder
            view.dot_painter  = dot_painter
            view.value_source = value_source
            view.show_paths = show_paths
            view.show_history_morphology = show_history_morphology
            view.show_crosshairs = show_crosshairs
            view.show_fallback_glyphs = show_fallback_glyphs
            view.show_body_details = show_body_details
        else:
            view.color_mapper  = color_mapper
            view.backgrounder  = backgrounder
            view.dot_painter   = dot_painter
            view.value_source  = value_source
            view.show_paths = show_paths
            view.show_history_morphology = show_history_morphology
            view.show_crosshairs = show_crosshairs
            view.show_fallback_glyphs = show_fallback_glyphs
            view.show_body_details = show_body_details

        return self._render_view(dances, view)

    def render_at_time(self,
                       dances: Dict[int, "Dance"],
                       t: float,
                       trail_s: float = 5.0,
                       **render_kwargs) -> "Image.Image":
        """
        Render a snapshot showing only the portion of each track within
        *trail_s* seconds before time *t*.

        Parameters
        ----------
        t       : snapshot time (seconds)
        trail_s : how many seconds of history to show
        """
        return self.render(
            dances,
            t_start=max(0.0, t - trail_s),
            t_end=t,
            t_at=t,
            **render_kwargs
        )

    def render_timeseries(self,
                          dances: Dict[int, "Dance"],
                          times: Optional[np.ndarray] = None,
                          fps: float = 10.0,
                          trail_s: float = 2.0,
                          **render_kwargs) -> List["Image.Image"]:
        """
        Render a sequence of frames suitable for animation.

        Parameters
        ----------
        times  : array of snapshot times; if None, sample at *fps* from
                 the track time range
        fps    : frames per second (used only when times=None)
        trail_s: history window in seconds

        Returns
        -------
        List of PIL images (one per frame)
        """
        if times is None:
            t0 = min(float(d.times[0]) for d in dances.values() if len(d.times))
            t1 = max(float(d.times[-1]) for d in dances.values() if len(d.times))
            times = np.arange(t0, t1, 1.0 / fps)

        # Pre-compute the view bounds once
        width_px  = render_kwargs.pop('width_px',  800)
        height_px = render_kwargs.pop('height_px', 600)
        view_base = ViewRequest.auto_bounds(dances, width_px, height_px)

        frames = []
        for t in times:
            img = self.render_at_time(
                dances, float(t), trail_s=trail_s,
                width_px=width_px, height_px=height_px,
                **render_kwargs
            )
            frames.append(img)
        return frames

    def save_gif(self,
                 dances: Dict[int, "Dance"],
                 output_path: Union[str, Path],
                 fps: float = 10.0,
                 trail_s: float = 2.0,
                 **render_kwargs) -> None:
        """
        Save an animated GIF of worm tracks.

        Parameters
        ----------
        output_path : output file path (.gif)
        fps         : animation frame rate
        trail_s     : seconds of trail to show
        """
        frames = self.render_timeseries(dances, fps=fps, trail_s=trail_s,
                                        **render_kwargs)
        if not frames:
            return
        delay_ms = int(1000 / fps)
        frames[0].save(
            str(output_path),
            save_all=True,
            append_images=frames[1:],
            duration=delay_ms,
            loop=0,
        )

    # ------------------------------------------------------------------
    # Core rendering
    # ------------------------------------------------------------------

    def _render_view(self,
                     dances: Dict[int, "Dance"],
                     view: ViewRequest) -> "Image.Image":
        """Internal: execute one full render pass."""
        img  = Image.new("RGBA", (view.width_px, view.height_px))
        draw = ImageDraw.Draw(img)

        # Fill background
        if isinstance(view.backgrounder, ImageGrounder):
            view.backgrounder.fill_image(img, view.t_at)
        else:
            img.paste(view.backgrounder.background_color(),
                      [0, 0, view.width_px, view.height_px])

        t0, t1 = view.t_range

        # Reset line painter state between worms
        if isinstance(view.dot_painter, LinePainter):
            view.dot_painter.reset()

        for wid in sorted(dances.keys()):
            dance = dances[wid]
            if len(dance.times) == 0:
                continue
            self._paint_dance(draw, dance, view, t0, t1)

        del draw
        return img

    # Rendering-tier thresholds (in image pixels), matching Java constants:
    #   if r_screen < _TIER_DOT   → crosshair
    #   if r_screen < _TIER_LINE  → bearing line (major-axis segment)
    #   else                      → full body (outline polygon or spine+widths)
    _TIER_DOT  = 2.0
    _TIER_LINE = 7.0

    def _paint_dance(self, draw, dance: "Dance",
                     view: ViewRequest,
                     t0: float, t1: float) -> None:
        """
        Paint one worm's track segment onto the drawing context.

        Rendering tier (mirrors DataMapper$ViewRequest.setView bytecode):

        r_screen = extent_major_px / pixel_size           (screen radius in px)

        r_screen <  2  →  crosshair (two 3-px lines)
        r_screen <  7  →  bearing-axis segment (head/tail line; no body)
        r_screen >= 7  →  full body:
                            • outline polygon if available
                            • else spine polyline + perpendicular width bars
                          plus a cross-hair at the centroid in highlight colour
        """
        if isinstance(view.dot_painter, LinePainter):
            view.dot_painter.reset()

        hl = view.backgrounder.highlight()   # highlight colour
        ml = view.backgrounder.midlight()    # midlight colour
        time_tol = max(self.TIME_TOLERANCE, 1e-6)

        visible_indices = []
        for i in range(dance.n_frames):
            t = float(dance.times[i])
            if t0 <= t <= t1:
                visible_indices.append(i)
        if not visible_indices:
            return

        morphology_indices = visible_indices
        if not view.show_history_morphology:
            morphology_indices = [
                idx for idx in visible_indices
                if abs(float(dance.times[idx]) - float(view.t_at)) <= time_tol
            ]

        for i in visible_indices:
            t = float(dance.times[i])
            cx = float(dance.centroid[i, 0])
            cy = float(dance.centroid[i, 1])
            if not (math.isfinite(cx) and math.isfinite(cy)):
                if isinstance(view.dot_painter, LinePainter):
                    view.dot_painter.reset()
                continue

            # Screen coordinates of centroid
            ix, iy = view.data_to_image(cx, cy)
            if ix < -20 or ix > view.width_px + 20:
                continue
            if iy < -20 or iy > view.height_px + 20:
                continue

            # Screen radius (major axis half-length in image pixels)
            r_screen = 0.0
            if i < len(dance.extent) and dance.extent[i] is not None:
                major = float(dance.extent[i][0])   # pixels
                r_screen = major / view.pixel_size
            elif i < len(dance.area) and dance.area[i] > 0:
                # Fallback: estimate from area
                r_screen = math.sqrt(float(dance.area[i]) / math.pi) / view.pixel_size

            # Normalised value and colour
            raw_val = view.value_source.value_at(dance, i)
            rgba    = view.color_mapper.map_color(raw_val)
            if rgba is None:
                continue

            if view.show_paths:
                view.dot_painter.paint(draw, ix, iy, rgba, i, raw_val, dance)

            if i not in morphology_indices:
                continue

            is_current = abs(t - float(view.t_at)) <= time_tol
            morph_rgba = hl if is_current else rgba

            # Tier 1: crosshair dot
            if r_screen < self._TIER_DOT:
                if view.show_crosshairs:
                    draw.line([(ix - 1, iy), (ix + 1, iy)], fill=hl)
                    draw.line([(ix, iy - 1), (ix, iy + 1)], fill=hl)
                continue

            # Tier 2: bearing-axis line
            if r_screen < self._TIER_LINE:
                if view.show_fallback_glyphs:
                    self._draw_bearing_axis(draw, dance, i, ix, iy, r_screen,
                                            morph_rgba, hl, view)
                continue

            # Tier 3: full body
            if view.show_crosshairs:
                draw.line([(ix - 1, iy), (ix + 1, iy)], fill=hl)
                draw.line([(ix, iy - 1), (ix, iy + 1)], fill=hl)

            has_outline = (i < len(dance.outline) and dance.outline[i] is not None)
            has_spine   = (i < len(dance.spine)   and dance.spine[i]   is not None)

            if has_outline:
                self._draw_outline(draw, dance, i, ix, iy, morph_rgba, hl, ml, view)
            elif has_spine:
                self._draw_spine_with_widths(draw, dance, i, ix, iy,
                                             r_screen, morph_rgba, hl, ml, view)
            elif view.show_fallback_glyphs:
                self._draw_bearing_axis(draw, dance, i, ix, iy, r_screen,
                                        morph_rgba, hl, view)

    # ------------------------------------------------------------------
    # Body-rendering helpers
    # ------------------------------------------------------------------

    def _img_pt(self, px: float, py: float,
                view: ViewRequest) -> Tuple[float, float]:
        """Convert data pixel (px, py) → image pixel."""
        return view.data_to_image(px, py)

    def _draw_bearing_axis(self, draw, dance, i: int,
                           ix: float, iy: float, r_screen: float,
                           rgba, hl, view: ViewRequest) -> None:
        """
        Draw major/minor axis cross.
        Mirrors the bearing-line path in setView for 2 ≤ r < 7 px.
        """
        if i >= len(dance.bearing) or dance.bearing[i] is None:
            if view.show_crosshairs:
                draw.line([(ix - 1, iy), (ix + 1, iy)], fill=rgba)
                draw.line([(ix, iy - 1), (ix, iy + 1)], fill=rgba)
            return

        bx = float(dance.bearing[i][0])
        by = float(dance.bearing[i][1])
        blen = math.hypot(bx, by)
        if blen > 0:
            bx /= blen; by /= blen

        major = r_screen   # half-length in image pixels
        minor = major * 0.5 if (i < len(dance.extent) and dance.extent[i] is not None
                                and float(dance.extent[i][0]) > 0) else major * 0.3
        if i < len(dance.extent) and dance.extent[i] is not None:
            ex, ey = dance.extent[i][0], dance.extent[i][1]
            if ex > 0:
                minor = major * (float(ey) / float(ex))

        # Major axis (along bearing)
        ax0, ay0 = ix - bx * major, iy - by * major
        ax1, ay1 = ix + bx * major, iy + by * major
        draw.line([(ax0, ay0), (ax1, ay1)], fill=rgba, width=1)

        # Minor axis (perpendicular)
        mx0, my0 = ix - by * minor, iy + bx * minor
        mx1, my1 = ix + by * minor, iy - bx * minor
        draw.line([(mx0, my0), (mx1, my1)], fill=rgba, width=1)

    def _draw_outline(self, draw, dance, i: int,
                      ix: float, iy: float,
                      rgba, hl, ml,
                      view: ViewRequest) -> None:
        """
        Draw the worm body outline as a closed polygon.
        Mirrors the outline rendering path in setView for r ≥ 7 px.

        First pass: draw in *hl* (highlight).
        Second pass (if spine also available): redraw in *ml* (midlight).
        """
        out = dance.outline[i]
        cx  = float(dance.centroid[i][0])
        cy  = float(dance.centroid[i][1])

        # Unpack outline points
        try:
            pts = out.points   # (N, 2) array or list of (x, y)
        except AttributeError:
            return
        if len(pts) < 2:
            return

        # Outline coords are stored relative to centroid in some formats
        try:
            absolute = out.absolute   # True if already in global px coords
        except AttributeError:
            absolute = False

        screen_pts = []
        for p in pts:
            px = float(p[0]) + (0.0 if absolute else cx)
            py = float(p[1]) + (0.0 if absolute else cy)
            screen_pts.append(view.data_to_image(px, py))

        has_spine = (i < len(dance.spine) and dance.spine[i] is not None)
        draw.polygon(screen_pts, fill=rgba)
        if view.show_body_details:
            draw.polygon(screen_pts, outline=hl)
            if has_spine:
                draw.polygon(screen_pts, outline=ml)
        if has_spine and view.show_body_details:
            self._draw_spine_line(draw, dance, i, cx, cy, rgba, view)

    def _draw_spine_line(self, draw, dance, i: int,
                         cx: float, cy: float,
                         rgba, view: ViewRequest) -> None:
        """Draw the spine as a simple polyline in *rgba*."""
        sp = dance.spine[i]
        try:
            absolute = bool(getattr(sp, "absolute", False))
            pts = [
                (
                    float(p[0]) + (0.0 if absolute else cx),
                    float(p[1]) + (0.0 if absolute else cy),
                )
                for p in sp.points
            ]
        except AttributeError:
            return
        if len(pts) < 2:
            return
        screen = [view.data_to_image(x, y) for x, y in pts]
        draw.line(screen, fill=rgba, width=1)

    def _infer_spine_half_widths(self, dance, i: int, n_pts: int) -> List[float]:
        """
        Infer per-knot half-widths in data-pixel units when explicit spine
        widths are unavailable. Falls back to the frame's minor extent with a
        tapered body profile.
        """
        base_half_width = 0.0
        if i < len(dance.extent) and dance.extent[i] is not None:
            try:
                base_half_width = max(0.0, float(dance.extent[i][1]))
            except (TypeError, ValueError, IndexError):
                base_half_width = 0.0
        if base_half_width <= 0.0 and i < len(dance.area) and float(dance.area[i]) > 0.0:
            # Circle-equivalent fallback.
            base_half_width = math.sqrt(float(dance.area[i]) / math.pi) * 0.35

        if base_half_width <= 0.0:
            return [0.0] * n_pts

        widths = []
        for k in range(n_pts):
            if n_pts == 1:
                frac = 0.5
            else:
                frac = k / (n_pts - 1)
            # Smooth tapered profile: narrow head/tail, fullest mid-body.
            taper = math.sin(math.pi * frac) ** 0.8
            widths.append(base_half_width * taper)
        return widths

    def _spine_body_polygon(self, sp, cx: float, cy: float,
                            view: ViewRequest,
                            widths: Optional[List[float]] = None) -> Optional[List[Tuple[float, float]]]:
        try:
            n_pts = len(sp.points)
        except AttributeError:
            return None
        if n_pts < 2:
            return None

        absolute = bool(getattr(sp, "absolute", False))
        pts = []
        half_widths = []
        for k in range(n_pts):
            try:
                px = float(sp.points[k][0]) + (0.0 if absolute else cx)
                py = float(sp.points[k][1]) + (0.0 if absolute else cy)
            except (IndexError, TypeError, ValueError):
                return None
            pts.append((px, py))
            if widths is None:
                try:
                    w = float(sp.width(k))
                except (AttributeError, IndexError, TypeError, ValueError):
                    w = 0.0
                if not math.isfinite(w):
                    w = 0.0
                half_widths.append(max(0.0, w) * 0.5)
            else:
                half_widths.append(max(0.0, float(widths[k])))

        if not any(w > 0 for w in half_widths):
            return None

        left = []
        right = []
        for k in range(n_pts):
            xk, yk = pts[k]
            if k == 0:
                x0, y0 = pts[k]
                x1, y1 = pts[k + 1]
                tx, ty = x1 - x0, y1 - y0
            elif k == n_pts - 1:
                x0, y0 = pts[k - 1]
                x1, y1 = pts[k]
                tx, ty = x1 - x0, y1 - y0
            else:
                x0, y0 = pts[k - 1]
                x1, y1 = pts[k + 1]
                tx, ty = x1 - x0, y1 - y0
            tlen = math.hypot(tx, ty)
            if tlen < 1e-9:
                if k > 0:
                    x0, y0 = pts[k - 1]
                    tx, ty = xk - x0, yk - y0
                    tlen = math.hypot(tx, ty)
                if tlen < 1e-9 and k + 1 < n_pts:
                    x1, y1 = pts[k + 1]
                    tx, ty = x1 - xk, y1 - yk
                    tlen = math.hypot(tx, ty)
            if tlen < 1e-9:
                nx, ny = 0.0, 0.0
            else:
                tx /= tlen
                ty /= tlen
                nx, ny = -ty, tx
            hw = half_widths[k]
            left.append(view.data_to_image(xk + nx * hw, yk + ny * hw))
            right.append(view.data_to_image(xk - nx * hw, yk - ny * hw))

        poly = left + right[::-1]
        return poly if len(poly) >= 3 else None

    def _draw_spine_with_widths(self, draw, dance, i: int,
                                ix: float, iy: float,
                                r_screen: float,
                                rgba, hl, ml,
                                view: ViewRequest) -> None:
        """
        Draw the spine midline plus perpendicular width bars at each knot.
        Mirrors the spine+width rendering path in setView for r ≥ 7 px.

        For each interior knot k (1 … N-2):
          - Compute the tangent direction from knots k-1 → k+1
          - Perpendicular direction = (-dy, dx)
          - Half-width = spine.width(k) * 1000 * mm_per_pixel / pixel_size / 2
          - Draw a line segment of length half_width on each side
        """
        sp = dance.spine[i]
        cx = float(dance.centroid[i][0])
        cy = float(dance.centroid[i][1])

        try:
            n_pts = len(sp.points)
        except AttributeError:
            return
        if n_pts < 2:
            return

        widths = []
        try:
            for k in range(n_pts):
                w = sp.width(k)
                widths.append(None if (w != w) else float(w))
        except (AttributeError, IndexError):
            widths = [None] * n_pts

        has_widths = any(w is not None and w > 0 for w in widths)
        if not has_widths:
            inferred = self._infer_spine_half_widths(dance, i, n_pts)
            widths = inferred
        else:
            widths = [0.0 if w is None else max(0.0, w * 0.5) for w in widths]

        body_poly = self._spine_body_polygon(sp, cx, cy, view, widths=widths)
        if body_poly is not None:
            draw.polygon(body_poly, fill=rgba)
            if not view.show_body_details:
                return
            draw.polygon(body_poly, outline=hl)
            draw.polygon(body_poly, outline=ml)
        elif not view.show_body_details:
            return

        # Collect knot positions in image space
        knots_img = []
        absolute = bool(getattr(sp, "absolute", False))
        for k in range(n_pts):
            try:
                kx = float(sp.points[k][0]) + (0.0 if absolute else cx)
                ky = float(sp.points[k][1]) + (0.0 if absolute else cy)
            except (IndexError, TypeError):
                return
            knots_img.append(view.data_to_image(kx, ky))

        # Draw midline (highlight first, then midlight → visible layering)
        draw.line(knots_img, fill=hl, width=1)

        has_widths = any(w > 0 for w in widths)

        # Width bars at interior knots
        if has_widths:
            for k in range(1, n_pts - 1):
                w = widths[k]
                if w is None or w <= 0:
                    continue
                # Half-width in image pixels
                hw = float(w) / view.pixel_size
                if hw * 2.0 < 4.0:
                    continue   # too small to bother (mirrors Java threshold)

                # Tangent from previous to next knot
                x0, y0 = knots_img[k - 1]
                x2, y2 = knots_img[k + 1]
                tx, ty = x2 - x0, y2 - y0
                tlen = math.hypot(tx, ty)
                if tlen < 1e-9:
                    continue
                tx /= tlen; ty /= tlen

                # Perpendicular direction
                nx, ny = -ty, tx

                kx, ky = knots_img[k]
                draw.line([(kx - nx * hw, ky - ny * hw),
                           (kx + nx * hw, ky + ny * hw)],
                          fill=ml, width=1)

        # Re-draw midline in primary colour on top
        draw.line(knots_img, fill=rgba, width=1)

    # ------------------------------------------------------------------
    # Value source factory
    # ------------------------------------------------------------------

    def _make_value_source(self,
                           quantity,
                           v_min, v_max,
                           dances) -> ValueSource:
        """Convert quantity spec to a ValueSource."""
        if quantity is None or quantity == 'time':
            # Default: normalise by global time range
            t_min = min(float(d.times[0]) for d in dances.values() if len(d.times))
            t_max = max(float(d.times[-1]) for d in dances.values() if len(d.times))

            def time_fn(d):
                return d.times.astype(float)

            return ValueValue(time_fn, t_min, t_max)

        if callable(quantity):
            return ValueValue(quantity, v_min, v_max)

        # String code → dispatch
        from .choreography import OUTPUT_CODES
        name = OUTPUT_CODES.get(quantity, quantity)
        mpp  = self.mm_per_pixel

        dispatch = {
            'speed':       lambda d: d.quantityIsSpeed(mpp, 0.5),
            'angular_speed': lambda d: d.quantityIsAngularSpeed(mpp, 0.5),
            'length':      lambda d: d.quantityIsLength(),
            'rel_length':  lambda d: d.quantityIsLength(relative=True),
            'width':       lambda d: d.quantityIsWidth(),
            'aspect':      lambda d: d.quantityIsAspect(),
            'midline':     lambda d: d.quantityIsMidline(),
            'kink':        lambda d: d.quantityIsKink(),
            'bias':        lambda d: d.quantityIsBias(mpp, 0.5),
            'path':        lambda d: d.quantityIsPath(mpp),
            'curve':       lambda d: d.quantityIsCurve(),
            'dir_change':  lambda d: d.quantityIsDirectionChange(mpp, 0.5),
            'loc_x':       lambda d: d.quantityIsX(),
            'loc_y':       lambda d: d.quantityIsY(),
            'theta':       lambda d: d.quantityIsTheta(),
            'crab':        lambda d: d.quantityIsCrab(mpp, 0.5),
            'qxfw':        lambda d: d.quantityIsQxfw(),
            'area':        lambda d: d.quantityIsArea(),
            'time':        lambda d: d.quantityIsTime(),
            'frame':       lambda d: d.quantityIsFrame().astype(float),
        }
        fn = dispatch.get(name)
        if fn is None:
            warnings.warn(f"Unknown quantity '{quantity}', using time.")
            fn = lambda d: d.quantityIsTime()

        return ValueValue(fn, v_min, v_max)

    # ------------------------------------------------------------------
    # Overlay helpers (outline, spine, bearings)
    # ------------------------------------------------------------------

    def overlay_outlines(self,
                         img: "Image.Image",
                         dances: Dict[int, "Dance"],
                         view: ViewRequest,
                         t: float,
                         color: Tuple[int, int, int, int] = (255, 255, 0, 200),
                         line_width: int = 1) -> "Image.Image":
        """Draw body outlines on an existing image at time *t*."""
        draw = ImageDraw.Draw(img)
        tol  = self.TIME_TOLERANCE
        for dance in dances.values():
            idx = dance.seek_time_index(t)
            if abs(dance.times[idx] - t) > tol:
                continue
            out = dance.outline[idx] if idx < len(dance.outline) else None
            if out is None:
                continue
            cx = float(dance.centroid[idx, 0])
            cy = float(dance.centroid[idx, 1])
            absolute = bool(getattr(out, "absolute", False))
            pts_img = [
                view.data_to_image(
                    float(p[0]) + (0.0 if absolute else cx),
                    float(p[1]) + (0.0 if absolute else cy),
                )
                for p in out.points
            ]
            if len(pts_img) >= 2:
                draw.polygon(pts_img, outline=color)
        del draw
        return img

    def overlay_spines(self,
                       img: "Image.Image",
                       dances: Dict[int, "Dance"],
                       view: ViewRequest,
                       t: float,
                       color: Tuple[int, int, int, int] = (255, 128, 0, 200),
                       line_width: int = 2) -> "Image.Image":
        """Draw spine midlines on an existing image at time *t*."""
        draw = ImageDraw.Draw(img)
        tol  = self.TIME_TOLERANCE
        for dance in dances.values():
            idx = dance.seek_time_index(t)
            if abs(dance.times[idx] - t) > tol:
                continue
            sp = dance.spine[idx] if idx < len(dance.spine) else None
            if sp is None:
                continue
            cx = float(dance.centroid[idx, 0])
            cy = float(dance.centroid[idx, 1])
            absolute = bool(getattr(sp, "absolute", False))
            pts_img = [
                view.data_to_image(
                    float(p[0]) + (0.0 if absolute else cx),
                    float(p[1]) + (0.0 if absolute else cy),
                )
                for p in sp.points
            ]
            if len(pts_img) >= 2:
                draw.line(pts_img, fill=color, width=line_width)
        del draw
        return img

    def overlay_bearings(self,
                         img: "Image.Image",
                         dances: Dict[int, "Dance"],
                         view: ViewRequest,
                         t: float,
                         length: float = 20.0,
                         color: Tuple[int, int, int, int] = (255, 255, 255, 220)
                         ) -> "Image.Image":
        """Draw bearing (heading) arrows on an existing image at time *t*."""
        draw = ImageDraw.Draw(img)
        tol  = self.TIME_TOLERANCE
        for dance in dances.values():
            idx = dance.seek_time_index(t)
            if abs(dance.times[idx] - t) > tol:
                continue
            if idx >= len(dance.bearing):
                continue
            cx = float(dance.centroid[idx, 0])
            cy = float(dance.centroid[idx, 1])
            bx = float(dance.bearing[idx, 0])
            by = float(dance.bearing[idx, 1])
            x0, y0 = view.data_to_image(cx, cy)
            x1 = x0 + bx * length / view.pixel_size
            y1 = y0 + by * length / view.pixel_size
            draw.line([(x0, y0), (x1, y1)], fill=color, width=2)
            # Arrowhead
            ax, ay = -by * 3, bx * 3
            draw.polygon(
                [(int(x1), int(y1)),
                 (int(x1 - bx*5 + ax), int(y1 - by*5 + ay)),
                 (int(x1 - bx*5 - ax), int(y1 - by*5 - ay))],
                fill=color
            )
        del draw
        return img

    def save_frames(self,
                    dances: Dict[int, "Dance"],
                    output_dir: Union[str, Path],
                    fps: float = 10.0,
                    trail_s: float = 2.0,
                    prefix: str = "frame",
                    fmt: str = "png",
                    **render_kwargs) -> List[Path]:
        """
        Save individual PNG (or JPEG) frames to *output_dir*.

        Parameters
        ----------
        output_dir : directory to write frames into (created if absent)
        fps        : how many frames per second of experiment time to render
        trail_s    : history window (seconds) per frame
        prefix     : filename prefix
        fmt        : image format ('png' or 'jpeg')

        Returns
        -------
        List of Path objects for the written files.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        frames = self.render_timeseries(dances, fps=fps, trail_s=trail_s, **render_kwargs)
        paths = []
        for i, frame in enumerate(frames):
            p = output_dir / f"{prefix}_{i:05d}.{fmt}"
            frame.save(str(p))
            paths.append(p)
        return paths

    def save_video(self,
                   dances: Dict[int, "Dance"],
                   output_path: Union[str, Path],
                   fps: float = 10.0,
                   trail_s: float = 2.0,
                   **render_kwargs) -> None:
        """
        Save an MP4 video of worm tracks using ffmpeg.

        Requires ``ffmpeg`` on the system PATH.

        Parameters
        ----------
        output_path : output .mp4 file path
        fps         : frame rate
        trail_s     : history window per frame
        """
        import subprocess, tempfile, shutil
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found. Install ffmpeg or use save_gif().")
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self.save_frames(dances, tmpdir, fps=fps, trail_s=trail_s,
                                     prefix="frame", fmt="png", **render_kwargs)
            if not paths:
                return
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", str(Path(tmpdir) / "frame_%05d.png"),
                "-vcodec", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "18",
                str(output_path),
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)

    def add_colorbar_to_image(self,
                               img: "Image.Image",
                               color_mapper: ColorMapper,
                               v_min: float = 0.0,
                               v_max: float = 1.0,
                               label: str = "",
                               width: int = 24,
                               margin: int = 6) -> "Image.Image":
        """
        Append a vertical colorbar to a rendered PIL image.

        Parameters
        ----------
        img          : source RGBA image
        color_mapper : mapper used to produce the main render
        v_min, v_max : data range for axis labels
        label        : quantity name shown at top of bar
        width        : colorbar width in pixels
        margin       : gap between image and bar

        Returns
        -------
        New PIL.Image with colorbar appended on the right.
        """
        bar_w = width + margin * 2
        out = Image.new("RGBA", (img.width + bar_w, img.height), (30, 30, 30, 255))
        out.paste(img, (0, 0))

        draw = ImageDraw.Draw(out)
        n = img.height
        for py in range(n):
            # v goes from 1 (top) to 0 (bottom)
            v = 1.0 - py / max(n - 1, 1)
            rgba = color_mapper.map_color(v)
            if rgba is None:
                rgba = (0, 0, 0, 0)
            x0 = img.width + margin
            draw.rectangle([x0, py, x0 + width - 1, py], fill=rgba)

        # Labels
        try:
            from PIL import ImageFont
            font = ImageFont.load_default()
        except Exception:
            font = None

        lx = img.width + margin
        # Top label (v_max), bottom label (v_min)
        draw.text((lx, 2), f"{v_max:.3g}", fill=(220, 220, 220, 255), font=font)
        draw.text((lx, img.height - 12), f"{v_min:.3g}", fill=(220, 220, 220, 255), font=font)
        if label:
            draw.text((lx, img.height // 2 - 6), label[:10],
                      fill=(200, 200, 200, 255), font=font)
        del draw
        return out

    # ------------------------------------------------------------------
    # Matplotlib interactive viewer
    # ------------------------------------------------------------------

    def show(self,
             dances: Dict[int, "Dance"],
             quantity=None,
             color_mapper: Optional[ColorMapper] = None,
             backgrounder: Optional[Backgrounder] = None,
             dot_painter:  Optional[DotPainter]   = None,
             width_px: int = 900,
             height_px: int = 700,
             title: str = "Choreography Data Map",
             colorbar: bool = True,
             terminate_on_close: bool = True,
             **render_kwargs) -> None:
        """
        Display the data map in a matplotlib window.

        This is the Python equivalent of the interactive Java
        DataMapVisualizer GUI launched by ``--map``.

        Supports:
          - Zoom (scroll wheel)
          - Pan (click-drag)
          - Time scrubbing (slider widget)
          - Play/pause playback button
          - Playback FPS slider
          - Path visibility toggle
          - Morphology-history toggle
          - Colourbar with quantity label
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.widgets as mwidgets
        except ImportError:
            raise ImportError("matplotlib is required for DataMapper.show()")

        # Initial full render
        show_paths = bool(render_kwargs.pop('show_paths', False))
        show_history_morphology = bool(
            render_kwargs.pop('show_history_morphology', False)
        )
        show_crosshairs = bool(render_kwargs.pop('show_crosshairs', False))
        show_fallback_glyphs = bool(render_kwargs.pop('show_fallback_glyphs', False))
        show_body_details = bool(render_kwargs.pop('show_body_details', False))
        playback_fps_init = float(render_kwargs.pop('playback_fps', 10.0))
        trail_s = float(render_kwargs.get('trail_s', 0.0))

        img = self.render(
            dances, quantity=quantity,
            color_mapper=color_mapper,
            backgrounder=backgrounder,
            dot_painter=dot_painter,
            width_px=width_px, height_px=height_px,
            show_paths=show_paths,
            show_history_morphology=show_history_morphology,
            show_crosshairs=show_crosshairs,
            show_fallback_glyphs=show_fallback_glyphs,
            show_body_details=show_body_details,
            **render_kwargs
        )

        fig, ax = plt.subplots(figsize=(width_px / 100, height_px / 100),
                               facecolor='#1a1a1a')
        fig.suptitle(title, color='white', fontsize=12)

        im_obj = ax.imshow(np.array(img), origin='upper')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor('#1a1a1a')
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Colour scale legend
        if colorbar and quantity is not None:
            cmap_name = type(color_mapper).__name__ if color_mapper else 'Rainbow'
            self._add_colorbar(fig, ax, cmap_name,
                               str(quantity) if isinstance(quantity, str)
                               else "value")

        # Time slider
        t_all  = np.concatenate([d.times for d in dances.values() if len(d.times)])
        t_min, t_max = float(t_all.min()), float(t_all.max())

        ax_slider = plt.axes([0.18, 0.065, 0.67, 0.025],
                             facecolor='#333333')
        slider = mwidgets.Slider(ax_slider, 'Time (s)', t_min, t_max,
                                 valinit=t_min, color='#66aaff')
        slider.label.set_color('white')
        slider.valtext.set_color('white')
        playback_state = {
            'playing': False,
            'fps': max(playback_fps_init, 0.25),
            'show_paths': show_paths,
            'show_history_morphology': show_history_morphology,
            'show_crosshairs': show_crosshairs,
            'show_fallback_glyphs': show_fallback_glyphs,
            'show_body_details': show_body_details,
        }

        ax_button = plt.axes([0.07, 0.025, 0.08, 0.04], facecolor='#333333')
        play_button = mwidgets.Button(ax_button, 'Play', color='#444444', hovercolor='#666666')
        play_button.label.set_color('white')

        ax_fps = plt.axes([0.18, 0.02, 0.45, 0.025], facecolor='#333333')
        fps_slider = mwidgets.Slider(
            ax_fps,
            'FPS',
            0.5,
            30.0,
            valinit=playback_state['fps'],
            color='#ffaa66',
        )
        fps_slider.label.set_color('white')
        fps_slider.valtext.set_color('white')

        ax_checks = plt.axes([0.67, 0.008, 0.18, 0.07], facecolor='#222222')
        checks = mwidgets.CheckButtons(
            ax_checks,
            ['Paths', 'Morph Hist'],
            [playback_state['show_paths'], playback_state['show_history_morphology']],
        )
        for txt in checks.labels:
            txt.set_color('white')
        checkbox_rects = getattr(checks, 'rectangles', None)
        if checkbox_rects is None:
            checkbox_rects = getattr(ax_checks, 'patches', [])
        for rect in checkbox_rects:
            rect.set_facecolor('#333333')
            rect.set_edgecolor('white')
        checkbox_lines = getattr(checks, 'lines', None)
        if checkbox_lines is None:
            checkbox_lines = []
        for lines in checkbox_lines:
            for line in lines:
                line.set_color('#66ddff')

        if trail_s <= 0.0:
            trail_s = (t_max - t_min) * 0.1

        def update(val):
            t = slider.val
            new_img = self.render_at_time(
                dances, t=t, trail_s=trail_s,
                quantity=quantity,
                color_mapper=color_mapper,
                backgrounder=backgrounder,
                dot_painter=dot_painter,
                width_px=width_px, height_px=height_px,
                show_paths=playback_state['show_paths'],
                show_history_morphology=playback_state['show_history_morphology'],
                show_crosshairs=playback_state['show_crosshairs'],
                show_fallback_glyphs=playback_state['show_fallback_glyphs'],
                show_body_details=playback_state['show_body_details'],
            )
            im_obj.set_data(np.array(new_img))
            fig.canvas.draw_idle()

        def timer_tick():
            if not playback_state['playing']:
                return
            playback_step = 1.0 / max(playback_state['fps'], 1e-6)
            next_t = slider.val + playback_step
            if next_t >= t_max:
                next_t = t_max
                playback_state['playing'] = False
                play_button.label.set_text('Play')
                timer.stop()
            slider.set_val(next_t)

        timer = fig.canvas.new_timer(interval=max(1, int(1000 / max(playback_state['fps'], 1e-6))))
        timer.add_callback(timer_tick)

        def toggle_play(_event):
            playback_state['playing'] = not playback_state['playing']
            play_button.label.set_text('Pause' if playback_state['playing'] else 'Play')
            if playback_state['playing']:
                if slider.val >= t_max:
                    slider.set_val(t_min)
                timer.start()
            else:
                timer.stop()

        def update_fps(_val):
            playback_state['fps'] = max(float(fps_slider.val), 0.25)
            timer.stop()
            timer.interval = max(1, int(1000 / playback_state['fps']))
            if playback_state['playing']:
                timer.start()

        def toggle_checks(_label):
            states = checks.get_status()
            playback_state['show_paths'] = bool(states[0])
            playback_state['show_history_morphology'] = bool(states[1])
            update(slider.val)

        slider.on_changed(update)
        play_button.on_clicked(toggle_play)
        fps_slider.on_changed(update_fps)
        checks.on_clicked(toggle_checks)

        plt.tight_layout(rect=[0, 0.06, 1, 1])
        plt.show()
        if terminate_on_close and sys.stdin.isatty():
            try:
                from IPython import get_ipython
                if get_ipython() is None:
                    raise SystemExit(0)
            except ImportError:
                raise SystemExit(0)

    def _add_colorbar(self, fig, ax, cmap_name, label):
        """Add a pseudo-colourbar using a gradient image."""
        import matplotlib.pyplot as plt
        from matplotlib.image import AxesImage
        try:
            # Build a 1×256 gradient image using the colour mapper
            mapper_map = {
                'RainbowMapper': RainbowMapper(),
                'SunsetMapper':  SunsetMapper(),
                'SpatterMapper': SpatterMapper(),
            }
            cm = mapper_map.get(cmap_name, RainbowMapper())
            vals = np.linspace(0, 1, 256)
            grad = np.array([cm.map_color(v) or (0,0,0,0) for v in vals],
                            dtype=np.uint8).reshape(1, 256, 4)
            ax_cb = fig.add_axes([0.92, 0.1, 0.015, 0.8])
            ax_cb.imshow(grad[:, ::-1, :].transpose(1, 0, 2),
                         aspect='auto', origin='lower')
            ax_cb.set_xticks([])
            ax_cb.set_yticks([0, 255])
            ax_cb.set_yticklabels(['min', 'max'], color='white', fontsize=8)
            ax_cb.set_xlabel(label, color='white', fontsize=8)
        except Exception:
            pass
