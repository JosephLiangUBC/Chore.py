"""
utils.py
--------
Lightweight vector types and general helper functions mirroring kerr.Vec.*
(Vec2F, Vec2I, Vec2S, Vec2D) and other utility primitives used throughout
the Choreography library.
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# 2-D vector types
# ---------------------------------------------------------------------------

@dataclass
class Vec2F:
    """2-D float vector (mirrors kerr.Vec.Vec2F)."""
    x: float = 0.0
    y: float = 0.0

    def dist(self, other: "Vec2F") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def length(self) -> float:
        return math.hypot(self.x, self.y)

    def dot(self, other: "Vec2F") -> float:
        return self.x * other.x + self.y * other.y

    def cross(self, other: "Vec2F") -> float:
        return self.x * other.y - self.y * other.x

    def normalized(self) -> "Vec2F":
        mag = self.length()
        if mag == 0:
            return Vec2F(0.0, 0.0)
        return Vec2F(self.x / mag, self.y / mag)

    def __add__(self, other: "Vec2F") -> "Vec2F":
        return Vec2F(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vec2F") -> "Vec2F":
        return Vec2F(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> "Vec2F":
        return Vec2F(self.x * scalar, self.y * scalar)

    def __repr__(self) -> str:
        return f"Vec2F({self.x:.4f}, {self.y:.4f})"

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float32)

    @staticmethod
    def from_array(arr: np.ndarray) -> "Vec2F":
        return Vec2F(float(arr[0]), float(arr[1]))


@dataclass
class Vec2D:
    """2-D double (float64) vector (mirrors kerr.Vec.Vec2D)."""
    x: float = 0.0
    y: float = 0.0

    def dist(self, other: "Vec2D") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def length(self) -> float:
        return math.hypot(self.x, self.y)

    def dot(self, other: "Vec2D") -> float:
        return self.x * other.x + self.y * other.y

    def __add__(self, other: "Vec2D") -> "Vec2D":
        return Vec2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vec2D") -> "Vec2D":
        return Vec2D(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> "Vec2D":
        return Vec2D(self.x * scalar, self.y * scalar)


@dataclass
class Vec2I:
    """2-D integer vector (mirrors kerr.Vec.Vec2I)."""
    x: int = 0
    y: int = 0

    def dist(self, other: "Vec2I") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def __add__(self, other: "Vec2I") -> "Vec2I":
        return Vec2I(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vec2I") -> "Vec2I":
        return Vec2I(self.x - other.x, self.y - other.y)


@dataclass
class Vec2S:
    """2-D short integer vector (mirrors kerr.Vec.Vec2S)."""
    x: int = 0
    y: int = 0

    def to_vec2f(self) -> Vec2F:
        return Vec2F(float(self.x), float(self.y))

    def dist(self, other: "Vec2S") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def __add__(self, other: "Vec2S") -> "Vec2S":
        return Vec2S(self.x + other.x, self.y + other.y)


# ---------------------------------------------------------------------------
# Signed angle helpers (mirrors Dance.signedAngle)
# ---------------------------------------------------------------------------

def signed_angle_v2s(a: Vec2S, b: Vec2S) -> float:
    """Signed angle from vector *a* to vector *b* in radians (range −π..π)."""
    return math.atan2(a.x * b.y - a.y * b.x, a.x * b.x + a.y * b.y)


def signed_angle_v2f(a: Vec2F, b: Vec2F) -> float:
    """Signed angle from vector *a* to vector *b* in radians (range −π..π)."""
    return math.atan2(a.x * b.y - a.y * b.x, a.x * b.x + a.y * b.y)


def signed_angle_arrays(ax: float, ay: float, bx: float, by: float) -> float:
    """Scalar version of signed angle."""
    return math.atan2(ax * by - ay * bx, ax * bx + ay * by)


# ---------------------------------------------------------------------------
# Interpolation helpers
# ---------------------------------------------------------------------------

def linear_interp(t: float, t0: float, t1: float, v0: float, v1: float) -> float:
    """Linear interpolation of value between (t0,v0) and (t1,v1) at time t."""
    if t1 == t0:
        return v0
    frac = (t - t0) / (t1 - t0)
    return v0 + frac * (v1 - v0)


def interp_vec2f(t: float, t0: float, t1: float,
                 v0: Vec2F, v1: Vec2F) -> Vec2F:
    """Linearly interpolate between two Vec2F values."""
    if t1 == t0:
        return v0
    frac = (t - t0) / (t1 - t0)
    return Vec2F(v0.x + frac * (v1.x - v0.x),
                 v0.y + frac * (v1.y - v0.y))


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------

def clip(value: int, lo: int, hi: int) -> int:
    """Integer clip (mirrors Respine.clip)."""
    return max(lo, min(hi, value))


def ring_right(i: int, n: int) -> int:
    """Next index on a ring of length n."""
    return (i + 1) % n


def ring_left(i: int, n: int) -> int:
    """Previous index on a ring of length n."""
    return (i - 1) % n


def smooth_array(data: np.ndarray, window: int) -> np.ndarray:
    """Simple box-car (uniform) smoothing with edge reflection."""
    if window <= 1:
        return data.copy()
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='same')


def gaussian_smooth(data: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing (1-D)."""
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(data.astype(float), sigma)


# ---------------------------------------------------------------------------
# Path utilities
# ---------------------------------------------------------------------------

def cumulative_arc_length(points: np.ndarray) -> np.ndarray:
    """
    Cumulative arc length along an (N, 2) array of 2-D points.
    Returns array of length N (starts at 0).
    """
    diffs = np.diff(points, axis=0)
    step_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
    return np.concatenate([[0.0], np.cumsum(step_lengths)])


def resample_polyline(points: np.ndarray, n: int) -> np.ndarray:
    """
    Resample a polyline to exactly *n* equidistant points.

    Parameters
    ----------
    points : (M, 2) array
    n      : target number of points

    Returns
    -------
    (n, 2) array
    """
    arc = cumulative_arc_length(points)
    total = arc[-1]
    if total == 0:
        return np.tile(points[0], (n, 1))
    t_new = np.linspace(0, total, n)
    xs = np.interp(t_new, arc, points[:, 0])
    ys = np.interp(t_new, arc, points[:, 1])
    return np.column_stack([xs, ys])


# ---------------------------------------------------------------------------
# Angle / bearing
# ---------------------------------------------------------------------------

def wrap_angle(a: float) -> float:
    """Wrap angle to (−π, π]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def angle_diff(a: float, b: float) -> float:
    """Signed angular difference a − b, wrapped to (−π, π]."""
    return wrap_angle(a - b)
