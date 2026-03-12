"""
spine_outline.py
----------------
Spine and Outline data structures plus the Fractionator and QuadRanger
helpers.

Mirrors:
  Spine (interface)         → SpineData
  Outline (interface)       → OutlineData
  Dance$RawSpine            → RawSpine
  Dance$RawOutline          → RawOutline
  Fractionator              → Fractionator
  QuadRanger                → QuadRanger
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

from .utils import Vec2F, Vec2S, Vec2I, resample_polyline, cumulative_arc_length


# ---------------------------------------------------------------------------
# Spine
# ---------------------------------------------------------------------------

class SpineData:
    """
    Stores spine key-points for one frame.
    Mirrors the Spine interface and Dance$RawSpine.

    *points* is an (N, 2) float32 array of (x, y) key-points.
    *widths* is an (N,) float32 array of cross-sectional widths at each point.
    """

    def __init__(self, points: np.ndarray,
                 widths: Optional[np.ndarray] = None,
                 absolute: bool = False):
        self.points: np.ndarray = np.asarray(points, dtype=np.float32)
        n = len(self.points)
        if widths is not None:
            self.widths: np.ndarray = np.asarray(widths, dtype=np.float32)
        else:
            self.widths = np.zeros(n, dtype=np.float32)
        self.absolute: bool = absolute
        self._flipped: bool = False

    def size(self) -> int:
        return len(self.points)

    def quantized(self) -> bool:
        return False

    def flip(self) -> None:
        """Reverse head-tail orientation (mirrors Spine.flip)."""
        self.points = self.points[::-1].copy()
        self.widths = self.widths[::-1].copy()
        self._flipped = not self._flipped

    def get(self, i: int) -> Vec2S:
        p = self.points[i]
        return Vec2S(int(round(p[0])), int(round(p[1])))

    def get_f(self, i: int) -> Vec2F:
        p = self.points[i]
        return Vec2F(float(p[0]), float(p[1]))

    def width(self, i: int) -> float:
        return float(self.widths[i])

    def length(self) -> float:
        """Arc length of the spine."""
        if len(self.points) < 2:
            return 0.0
        return float(cumulative_arc_length(self.points)[-1])

    def as_array(self) -> np.ndarray:
        return self.points.copy()

    def resampled(self, n: int) -> "SpineData":
        """Return a new SpineData resampled to exactly *n* points."""
        pts = resample_polyline(self.points, n)
        # Interpolate widths onto new sample points
        arc_old = cumulative_arc_length(self.points)
        arc_new = np.linspace(0, arc_old[-1], n)
        w_new = np.interp(arc_new, arc_old, self.widths)
        return SpineData(pts, w_new, absolute=self.absolute)

    def compact(self) -> None:
        """No-op: data is already compacted."""
        pass

    def __repr__(self) -> str:
        return f"SpineData(n={self.size()}, length={self.length():.2f})"


# ---------------------------------------------------------------------------
# Outline
# ---------------------------------------------------------------------------

class OutlineData:
    """
    Stores the body outline (contour) for one frame.
    Mirrors the Outline interface and Dance$RawOutline.

    *points* is an (M, 2) float32 array forming a closed polygon.
    """

    def __init__(self, points: np.ndarray, absolute: bool = False):
        self.points: np.ndarray = np.asarray(points, dtype=np.float32)
        self.absolute: bool = absolute

    def size(self) -> int:
        return len(self.points)

    def quantized(self) -> bool:
        return False

    def compact(self) -> None:
        pass

    def get_s(self, i: int) -> Vec2S:
        p = self.points[i % len(self.points)]
        return Vec2S(int(round(p[0])), int(round(p[1])))

    def get_f(self, i: int) -> Vec2F:
        p = self.points[i % len(self.points)]
        return Vec2F(float(p[0]), float(p[1]))

    def unpack_s(self, close: bool = False) -> np.ndarray:
        """Return integer array of outline points."""
        pts = self.points.astype(np.int16)
        if close:
            pts = np.vstack([pts, pts[:1]])
        return pts

    def unpack_f(self) -> np.ndarray:
        return self.points.copy()

    def area(self) -> float:
        """Shoelace area of the closed polygon."""
        pts = self.points
        n = len(pts)
        if n < 3:
            return 0.0
        x, y = pts[:, 0], pts[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))

    def perimeter(self) -> float:
        diffs = np.diff(np.vstack([self.points, self.points[:1]]), axis=0)
        return float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))

    def centroid(self) -> Vec2F:
        c = np.mean(self.points, axis=0)
        return Vec2F(float(c[0]), float(c[1]))

    def __repr__(self) -> str:
        return f"OutlineData(n={self.size()}, area={self.area():.1f})"


# ---------------------------------------------------------------------------
# Fractionator
# ---------------------------------------------------------------------------

class Fractionator:
    """
    Multi-scale (pyramid) representation of a 1-D float signal.
    Mirrors the Java Fractionator class.

    The signal is successively binned by factor *binsize*:
      level 0 → original
      level 1 → binned by *binsize*
      level 2 → binned by *binsize*²
      …
    """

    def __init__(self, data: np.ndarray, n: int, binsize: int = 2):
        """
        Parameters
        ----------
        data    : 1-D float array of length n (or more)
        n       : number of elements to use from data
        binsize : averaging factor per level (default 2)
        """
        self._binsize = binsize
        arr = np.asarray(data[:n], dtype=np.float32)
        levels: List[np.ndarray] = [arr.copy()]
        current = arr
        while len(current) >= binsize:
            trim = len(current) - len(current) % binsize
            if trim == 0:
                break
            reshaped = current[:trim].reshape(-1, binsize)
            current = reshaped.mean(axis=1)
            levels.append(current)
        self.data: List[np.ndarray] = levels
        self.diffsize: np.ndarray = np.array(
            [len(levels[i]) - len(levels[i + 1]) if i + 1 < len(levels)
             else len(levels[i])
             for i in range(len(levels))], dtype=np.float32
        )

    def depth(self) -> int:
        """Number of levels (including the original)."""
        return len(self.data)

    def get_level(self, level: int) -> np.ndarray:
        """Return the data array at a given pyramid level."""
        return self.data[level]

    def bin_factor(self, level: int) -> int:
        """Number of original samples that map to one sample at *level*."""
        return self._binsize ** level


# ---------------------------------------------------------------------------
# QuadRanger
# ---------------------------------------------------------------------------

@dataclass
class _Interval:
    start: int
    end: int  # exclusive


class QuadRanger:
    """
    Quad-tree–based spatial index for fast range queries on a 2-D path.

    Mirrors the Java QuadRanger class.  Given arrays of x and y
    coordinates (with the same time axis), it builds a recursive tree
    that allows efficient retrieval of all trajectory segments within an
    axis-aligned bounding box.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray,
                 start: int = 0, end: Optional[int] = None):
        """
        Parameters
        ----------
        x, y  : 1-D float arrays (same length)
        start : first index to include
        end   : one past the last index (default = len(x))
        """
        if end is None:
            end = len(x)
        self._x = np.asarray(x, dtype=np.float32)
        self._y = np.asarray(y, dtype=np.float32)
        self._root = self._build(start, end)

    # ------------------------------------------------------------------
    # Internal tree node
    # ------------------------------------------------------------------
    class _Node:
        __slots__ = ("start", "end", "x0", "y0", "x1", "y1",
                     "left", "right")

        def __init__(self, start: int, end: int,
                     x0: float, y0: float, x1: float, y1: float):
            self.start = start
            self.end = end
            self.x0 = x0; self.y0 = y0
            self.x1 = x1; self.y1 = y1
            self.left: Optional["QuadRanger._Node"] = None
            self.right: Optional["QuadRanger._Node"] = None

    def _build(self, start: int, end: int) -> Optional["QuadRanger._Node"]:
        if start >= end:
            return None
        xs = self._x[start:end]
        ys = self._y[start:end]
        node = QuadRanger._Node(start, end,
                                float(np.nanmin(xs)), float(np.nanmin(ys)),
                                float(np.nanmax(xs)), float(np.nanmax(ys)))
        if end - start <= 4:
            return node
        mid = (start + end) // 2
        node.left = self._build(start, mid)
        node.right = self._build(mid, end)
        return node

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def accumulate_valid(self,
                         lo: Vec2F, hi: Vec2F,
                         result: Optional[List[_Interval]] = None
                         ) -> List[_Interval]:
        """
        Collect all index intervals whose bounding box intersects [lo, hi].
        Mirrors QuadRanger.accumulateValid.
        """
        if result is None:
            result = []
        self._query(self._root, lo.x, lo.y, hi.x, hi.y, result)
        return result

    def _query(self, node: Optional["QuadRanger._Node"],
               x0: float, y0: float, x1: float, y1: float,
               result: List[_Interval]) -> None:
        if node is None:
            return
        # Prune if bounding boxes don't overlap
        if node.x1 < x0 or node.x0 > x1:
            return
        if node.y1 < y0 or node.y0 > y1:
            return
        if node.left is None and node.right is None:
            result.append(_Interval(node.start, node.end))
            return
        self._query(node.left, x0, y0, x1, y1, result)
        self._query(node.right, x0, y0, x1, y1, result)

    def bounded_area(self) -> float:
        """Approximate convex-hull area spanned by all stored points."""
        if self._root is None:
            return 0.0
        n = self._root
        return (n.x1 - n.x0) * (n.y1 - n.y0)
