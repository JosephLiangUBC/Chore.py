"""
dance.py
--------
The Dance class represents a single worm's (or animal's) track as recorded by
the Multi-Worm Tracker and analysed by Choreography.

This is the central data structure of the library.  All per-frame raw data
are stored as NumPy arrays; derived metrics are computed lazily on demand.

Mirrors the Java Dance class (Dance.java, 58 KB).
"""

from __future__ import annotations
import math
import warnings
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, TYPE_CHECKING

from .utils import (Vec2F, Vec2I, Vec2S, signed_angle_arrays,
                    wrap_angle, smooth_array, gaussian_smooth,
                    cumulative_arc_length, resample_polyline)
from .statistic import Statistic
from .fitter import Fitter
from .spine_outline import SpineData, OutlineData, Fractionator, QuadRanger

if TYPE_CHECKING:
    from .choreography import Choreography


# ---------------------------------------------------------------------------
# Small helper containers
# ---------------------------------------------------------------------------

@dataclass
class ReversalEvent:
    """One reversal in a worm's trajectory."""
    start_index: int
    end_index: int
    start_time: float
    end_time: float
    duration: float
    distance: float
    speed: float


@dataclass
class StyleSegment:
    """A contiguous segment of a trajectory with consistent behaviour."""
    start: int
    end: int
    label: str = ""


# ---------------------------------------------------------------------------
# Derivative noise / quality constants (mirrors Dance.java)
# ---------------------------------------------------------------------------
LN_FOUR: float = math.log(4)
DERIVATIVE_NOISE_FACTOR: float = math.sqrt(2)
CURVATURE_NOISE_FACTOR: float = math.sqrt(6)
ERROR_CONVERGE_ITERATIONS: int = 50
ERROR_CONVERGE_FRACTION: float = 0.01


# ---------------------------------------------------------------------------
# Dance class
# ---------------------------------------------------------------------------

class Dance:
    """
    One worm's track.

    Raw per-frame arrays
    --------------------
    times    : (N,) float32   – frame timestamps (seconds)
    frames   : (N,) int32     – frame indices
    area     : (N,) float32   – body area in pixels²
    centroid : (N, 2) float32 – centroid position in pixels (x, y)
    bearing  : (N, 2) float32 – instantaneous heading unit vector
    extent   : (N, 2) float32 – bounding-box half-widths (rx, ry) in pixels
    circles  : (N, 3) float32 – best-fit circle (cx, cy, r) in pixels
    spine    : list of SpineData | None per frame
    outline  : list of OutlineData | None per frame

    All derived quantities (speed, length, …) are computed lazily
    and cached in ``_cache``.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, worm_id: int = -1):
        self.ID: int = worm_id
        self.has_holes: bool = False

        # Time / frame index
        self.times: np.ndarray = np.empty(0, dtype=np.float32)
        self.frames: np.ndarray = np.empty(0, dtype=np.int32)

        # Raw morphology
        self.area: np.ndarray = np.empty(0, dtype=np.float32)
        self.centroid: np.ndarray = np.empty((0, 2), dtype=np.float32)
        self.bearing: np.ndarray = np.empty((0, 2), dtype=np.float32)
        self.extent: np.ndarray = np.empty((0, 2), dtype=np.float32)
        self.circles: np.ndarray = np.empty((0, 3), dtype=np.float32)

        # Per-frame spine / outline (sparse – may be None)
        self.spine: List[Optional[SpineData]] = []
        self.outline: List[Optional[OutlineData]] = []

        # Reference statistics (computed from whole track)
        self.body_area:       Statistic = Statistic()
        self.body_length:     Statistic = Statistic()
        self.body_width:      Statistic = Statistic()
        self.body_aspect:     Statistic = Statistic()
        self.body_spine:      Statistic = Statistic()
        self.noise_estimate:  Statistic = Statistic()
        self.directional_bias: Statistic = Statistic()

        # Position noise (from Fitter)
        self.global_position_noise: Fitter = Fitter()

        # Style / segmentation
        self.segmentation: List[StyleSegment] = []

        # Multi-scale fractionators (for readyMultiscale)
        self.multiscale_x: Optional[Fractionator] = None
        self.multiscale_y: Optional[Fractionator] = None
        self.multiscale_q: Optional[Fractionator] = None

        # Spatial range index
        self.ranges_xy: Optional[QuadRanger] = None

        # Ancestry tracking
        self.origins: List[int] = []
        self.fates:   List[int] = []

        # Shadow avoidance metadata
        self.shadow_avoided: bool = False
        self.ignored_dt: float = 0.0
        self.ignored_travel: float = 0.0

        # Receptive fields
        self.attend: List = []
        self.shun:   List = []

        # Cached computed quantities (lazy)
        self._cache: Dict[str, np.ndarray] = {}

        # Custom plugin quantities
        self.custom_quantities: List[Optional[np.ndarray]] = []

        # Endpoint angle fraction (for qxfw)
        self.endpoint_angle_fraction: float = 0.5

    # ------------------------------------------------------------------
    # Basic accessors / queries
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.times)

    @property
    def n_frames(self) -> int:
        return len(self.times)

    @property
    def first_frame(self) -> int:
        return int(self.frames[0]) if len(self.frames) else 0

    @property
    def last_frame(self) -> int:
        return int(self.frames[-1]) if len(self.frames) else 0

    def has_data(self) -> bool:
        return len(self.times) > 0 and np.any(np.isfinite(self.centroid))

    def duration(self) -> float:
        """Total track duration in seconds."""
        if len(self.times) < 2:
            return 0.0
        return float(self.times[-1] - self.times[0])

    def mean_body_length_estimate(self) -> float:
        """Return cached or computed mean body length in pixels."""
        if self.body_length.n > 0:
            return self.body_length.average
        lengths = self._compute_length_px()
        if len(lengths):
            return float(np.nanmean(lengths))
        return 1.0

    def position_noise_estimate(self) -> float:
        return self.global_position_noise.position_noise()

    # ------------------------------------------------------------------
    # Frame-time interpolation (mirrors Dance.seek / seekNearT)
    # ------------------------------------------------------------------

    def seek(self, t: float) -> Tuple[int, float]:
        """
        Find the fractional index in *times* closest to *t*.
        Returns (index, fraction) where fraction ∈ [0, 1).
        Mirrors Dance.seek(float, float[]).
        """
        idx = np.searchsorted(self.times, t)
        if idx == 0:
            return 0, 0.0
        if idx >= len(self.times):
            return len(self.times) - 1, 0.0
        t0, t1 = float(self.times[idx - 1]), float(self.times[idx])
        frac = (t - t0) / (t1 - t0) if t1 != t0 else 0.0
        return idx - 1, frac

    def seek_time_index(self, t: float, tol: float = 0.0) -> int:
        """Return the integer index nearest to time *t*."""
        if len(self.times) == 0:
            return 0
        idx = int(np.argmin(np.abs(self.times - t)))
        return idx

    # ------------------------------------------------------------------
    # Core statistics (mirrors Dance.calcBasicStatistics)
    # ------------------------------------------------------------------

    def calc_basic_statistics(self, relative: bool = False) -> bool:
        """
        Compute body_area, body_length, body_width, body_aspect statistics.
        If *relative* is True also normalise length/width by their means.
        Returns True if the worm appears to be moving (not a sitter).
        Mirrors Dance.calcBasicStatistics(boolean).
        """
        if len(self.area) == 0:
            return False

        self.body_area.compute(self.area)

        lengths = self._compute_length_px()
        widths  = self._compute_width_px()

        if len(lengths):
            self.body_length.compute(lengths)
        if len(widths):
            self.body_width.compute(widths)

        if (self.body_length.average > 0 and self.body_width.average > 0):
            aspect = lengths / np.maximum(widths, 1e-6)
            self.body_aspect.compute(aspect)

        # Spine length
        sp_lengths = np.array(
            [s.length() for s in self.spine if s is not None],
            dtype=np.float32
        )
        if len(sp_lengths):
            self.body_spine.compute(sp_lengths)

        return True

    def calc_position_noise(self) -> None:
        """Estimate positional noise from track residuals."""
        if len(self.centroid) < 3:
            return
        f = Fitter()
        for i in range(len(self.centroid)):
            if np.all(np.isfinite(self.centroid[i])):
                f.addC(float(self.centroid[i, 0]),
                       float(self.centroid[i, 1]))
        self.global_position_noise = f

    # ------------------------------------------------------------------
    # Computed quantity methods (mirrors Dance.quantityIs* methods)
    # ------------------------------------------------------------------

    def quantityIsTime(self, normalise: bool = False) -> np.ndarray:
        """Return frame timestamps."""
        q = self.times.copy()
        if normalise and len(q) and q[0] != 0:
            q -= q[0]
        return q

    def quantityIsFrame(self) -> np.ndarray:
        """Return frame indices."""
        return self.frames.astype(np.float32)

    def quantityIsArea(self, relative: bool = False) -> np.ndarray:
        """Body area in pixels² (or relative to mean)."""
        q = self.area.copy().astype(np.float32)
        if relative and self.body_area.average > 0:
            q /= self.body_area.average
        return q

    def quantityIsSpeed(self, mm_per_pixel: float = 1.0,
                        window: float = 0.5,
                        use_body_length: bool = False,
                        signed: bool = False) -> np.ndarray:
        """
        Centroid speed (mm/s or body-lengths/s).
        window : time window (seconds) over which speed is averaged.
        Mirrors Dance.quantityIsSpeed.
        """
        cache_key = f"speed_{mm_per_pixel:.6f}_{window:.3f}_{use_body_length}_{signed}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        q = self._compute_speed(mm_per_pixel, window, signed)
        if use_body_length:
            bl = self.mean_body_length_estimate() * mm_per_pixel
            if bl > 0:
                q /= bl
        self._cache[cache_key] = q
        return q

    def quantityIsAngularSpeed(self, mm_per_pixel: float = 1.0,
                               window: float = 0.5,
                               signed: bool = False) -> np.ndarray:
        """Angular speed (radians/s)."""
        cache_key = f"angular_{window:.3f}_{signed}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        q = self._compute_angular_speed(window, signed)
        self._cache[cache_key] = q
        return q

    def quantityIsLength(self, relative: bool = False) -> np.ndarray:
        """Body length in pixels (or relative to mean)."""
        q = self._compute_length_px()
        if relative and self.body_length.average > 0:
            q = q / self.body_length.average
        return q

    def quantityIsWidth(self, relative: bool = False) -> np.ndarray:
        """Body width in pixels (or relative to mean)."""
        q = self._compute_width_px()
        if relative and self.body_width.average > 0:
            q = q / self.body_width.average
        return q

    def quantityIsAspect(self, relative: bool = False) -> np.ndarray:
        """Aspect ratio (length/width)."""
        lengths = self._compute_length_px()
        widths  = self._compute_width_px()
        q = lengths / np.maximum(widths, 1e-6)
        if relative and self.body_aspect.average > 0:
            q /= self.body_aspect.average
        return q

    def quantityIsMidline(self, relative: bool = False) -> np.ndarray:
        """Spine (midline) length."""
        return self._compute_spine_length(relative)

    def quantityIsOutlineWidth(self, relative: bool = False) -> np.ndarray:
        """Outline-derived width estimate."""
        return self._compute_outline_width(relative)

    def quantityIsKink(self, relative: bool = False) -> np.ndarray:
        """
        End-wiggle (head/tail kink angle).
        Mirrors Dance.quantityIsKink / findEndWiggle.
        """
        return self._compute_end_wiggle()

    def quantityIsBias(self, mm_per_pixel: float = 1.0,
                       window: float = 0.5,
                       min_speed: float = 0.0) -> np.ndarray:
        """
        Directional bias: fraction of time spent moving forward vs.
        backward (range −1 to +1).
        Mirrors Dance.quantityIsBias.
        """
        return self._compute_direction_bias(mm_per_pixel, window, min_speed)

    def quantityIsPath(self, mm_per_pixel: float = 1.0,
                       window: float = 0.0,
                       min_displacement: float = 0.0) -> np.ndarray:
        """Cumulative path length (mm)."""
        return self._compute_path_length(mm_per_pixel, window, min_displacement)

    def quantityIsCurve(self, relative: bool = False) -> np.ndarray:
        """
        Body curvature (angle subtended by the spine, radians).
        Mirrors Dance.quantityIsCurve / findBodyWiggle.
        """
        return self._compute_body_wiggle()

    def quantityIsDirectionChange(self, mm_per_pixel: float = 1.0,
                                  window: float = 0.5) -> np.ndarray:
        """Rate of direction change (radians/s)."""
        return self._compute_direction_change(mm_per_pixel, window)

    def quantityIsX(self, mm: bool = False) -> np.ndarray:
        """X position (pixels or mm)."""
        q = self.centroid[:, 0].copy().astype(np.float32)
        if mm:
            q *= self._mm_per_pixel if hasattr(self, "_mm_per_pixel") else 1.0
        return q

    def quantityIsY(self, mm: bool = False) -> np.ndarray:
        """Y position (pixels or mm)."""
        q = self.centroid[:, 1].copy().astype(np.float32)
        if mm:
            q *= self._mm_per_pixel if hasattr(self, "_mm_per_pixel") else 1.0
        return q

    def quantityIsVx(self, mm_per_pixel: float = 1.0,
                     window: float = 0.5,
                     signed: bool = True,
                     smooth: bool = False) -> np.ndarray:
        """X component of velocity (mm/s)."""
        return self._compute_velocity(mm_per_pixel, window, component=0)

    def quantityIsVy(self, mm_per_pixel: float = 1.0,
                     window: float = 0.5,
                     signed: bool = True,
                     smooth: bool = False) -> np.ndarray:
        """Y component of velocity (mm/s)."""
        return self._compute_velocity(mm_per_pixel, window, component=1)

    def quantityIsTheta(self, radians: bool = True) -> np.ndarray:
        """Body orientation angle (radians or degrees)."""
        q = self._compute_orientation()
        if not radians:
            q = np.degrees(q)
        return q

    def quantityIsCrab(self, mm_per_pixel: float = 1.0,
                       window: float = 0.5,
                       signed: bool = False,
                       smooth: bool = False) -> np.ndarray:
        """
        Lateral (crab) speed component perpendicular to body axis (mm/s).
        Mirrors Dance.quantityIsCrab.
        """
        return self._compute_crab_speed(mm_per_pixel, window)

    def quantityIsQxfw(self, forward_fraction: float = 0.5) -> np.ndarray:
        """
        Qxfw: a posture-based forward/backward indicator.
        Uses spine endpoint positions relative to centroid.
        Mirrors Dance.quantityIsQxfw / findQxfw.
        """
        return self._compute_qxfw(forward_fraction)

    def quantityIsStim(self, events: np.ndarray, stim_index: int) -> np.ndarray:
        """Return a 0/1 array indicating when stimulus *stim_index* is on."""
        q = np.zeros(len(self.times), dtype=np.float32)
        if events is None or len(events) == 0:
            return q
        for on, off in events:
            mask = (self.times >= on) & (self.times < off)
            q[mask] = 1.0
        return q

    # ------------------------------------------------------------------
    # Reversal extraction (mirrors Dance.extractReversals)
    # ------------------------------------------------------------------

    def extract_reversals(self, mm_per_pixel: float = 1.0,
                          window: float = 0.5,
                          require_forward_after: bool = False
                          ) -> List[ReversalEvent]:
        """
        Detect backward locomotion events (reversals).

        A reversal is a continuous period during which the signed
        speed is negative (animal moving backward).

        Parameters
        ----------
        mm_per_pixel         : pixel→mm conversion
        window               : speed smoothing window (s)
        require_forward_after: only keep reversals followed by forward motion

        Returns
        -------
        List of ReversalEvent objects.
        """
        speed = self.quantityIsSpeed(mm_per_pixel, window, signed=True)
        times = self.times
        reversals = []
        in_rev = False
        start_i = 0
        for i, s in enumerate(speed):
            if not in_rev and s < 0:
                in_rev = True
                start_i = i
            elif in_rev and s >= 0:
                in_rev = False
                end_i = i
                t0 = float(times[start_i])
                t1 = float(times[end_i - 1])
                dur = t1 - t0
                # displacement during reversal
                seg = self.centroid[start_i:end_i]
                if len(seg) > 1:
                    disp = float(np.sum(np.hypot(
                        np.diff(seg[:, 0]), np.diff(seg[:, 1])
                    ))) * mm_per_pixel
                else:
                    disp = 0.0
                avg_speed = disp / dur if dur > 0 else 0.0
                reversals.append(ReversalEvent(
                    start_i, end_i, t0, t1, dur, disp, avg_speed
                ))
        return reversals

    # ------------------------------------------------------------------
    # Segmentation (mirrors Dance.findSegmentation)
    # ------------------------------------------------------------------

    def find_segmentation(self, min_frames: int = 5) -> None:
        """
        Partition the track into contiguous valid segments.
        Gaps (NaN centroids) separate segments.
        Mirrors Dance.findSegmentation.
        """
        self.segmentation = []
        if len(self.centroid) == 0:
            return
        valid = np.all(np.isfinite(self.centroid), axis=1)
        in_seg = False
        start = 0
        for i, v in enumerate(valid):
            if v and not in_seg:
                in_seg = True
                start = i
            elif not v and in_seg:
                in_seg = False
                if i - start >= min_frames:
                    self.segmentation.append(StyleSegment(start, i))
        if in_seg and len(self.centroid) - start >= min_frames:
            self.segmentation.append(StyleSegment(start, len(self.centroid)))

    def path_length(self, start: int, end: int) -> float:
        """Arc length of the centroid path from index *start* to *end* (pixels)."""
        if end <= start + 1:
            return 0.0
        seg = self.centroid[start:end]
        diffs = np.diff(seg, axis=0)
        return float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))

    # ------------------------------------------------------------------
    # Multiscale / readyMultiscale
    # ------------------------------------------------------------------

    def ready_multiscale(self, binsize: float = 2.0) -> None:
        """
        Build multi-resolution fractionators for x, y, and the currently
        set quantity array.  Mirrors Dance.readyMultiscale.
        """
        n = len(self.times)
        if n == 0:
            return
        bs = max(2, int(round(binsize)))
        x = self.centroid[:, 0]
        y = self.centroid[:, 1]
        self.multiscale_x = Fractionator(x, n, bs)
        self.multiscale_y = Fractionator(y, n, bs)
        if hasattr(self, '_quantity') and self._quantity is not None:
            self.multiscale_q = Fractionator(self._quantity, n, bs)

    # ------------------------------------------------------------------
    # Private computation helpers
    # ------------------------------------------------------------------

    def _dt(self) -> np.ndarray:
        """Frame-to-frame time differences (length n-1)."""
        return np.diff(self.times.astype(float))

    def _compute_speed(self, mm_per_pixel: float, window: float,
                       signed: bool = False) -> np.ndarray:
        """
        Instantaneous speed estimate using a centred finite difference
        over a time window.  Returns (N,) array in mm/s.
        """
        n = len(self.times)
        if n < 2:
            return np.zeros(n, dtype=np.float32)

        x = self.centroid[:, 0].astype(float)
        y = self.centroid[:, 1].astype(float)
        t = self.times.astype(float)

        speeds = np.zeros(n, dtype=np.float64)

        for i in range(n):
            # Find window boundaries
            t_lo = t[i] - window / 2
            t_hi = t[i] + window / 2
            lo = max(0, np.searchsorted(t, t_lo))
            hi = min(n - 1, np.searchsorted(t, t_hi, side='right') - 1)
            dt = t[hi] - t[lo]
            if dt <= 0 or lo == hi:
                continue
            dx = x[hi] - x[lo]
            dy = y[hi] - y[lo]
            if signed:
                # Project displacement onto bearing
                bx = float(self.bearing[i, 0]) if len(self.bearing) else 1.0
                by = float(self.bearing[i, 1]) if len(self.bearing) else 0.0
                sp = (dx * bx + dy * by) / dt
            else:
                sp = math.hypot(dx, dy) / dt
            speeds[i] = sp * mm_per_pixel

        return speeds.astype(np.float32)

    def _compute_angular_speed(self, window: float,
                               signed: bool = False) -> np.ndarray:
        """Angular speed in radians/s."""
        n = len(self.times)
        if n < 2 or len(self.bearing) < n:
            return np.zeros(n, dtype=np.float32)
        theta = self._compute_orientation()
        angular = np.zeros(n, dtype=np.float64)
        t = self.times.astype(float)
        for i in range(1, n - 1):
            t_lo = t[i] - window / 2
            t_hi = t[i] + window / 2
            lo = max(0, np.searchsorted(t, t_lo))
            hi = min(n - 1, np.searchsorted(t, t_hi, side='right') - 1)
            dt = t[hi] - t[lo]
            if dt <= 0:
                continue
            da = wrap_angle(theta[hi] - theta[lo])
            angular[i] = da / dt
        angular[0] = angular[1]
        angular[-1] = angular[-2]
        if not signed:
            angular = np.abs(angular)
        return angular.astype(np.float32)

    def _compute_length_px(self) -> np.ndarray:
        """Body length in pixels, estimated from major axis of bounding ellipse."""
        if len(self.extent) == 0:
            return np.zeros(len(self.times), dtype=np.float32)
        # extent stores (major_half, minor_half); length ≈ 2 * major_half
        return (2.0 * self.extent[:, 0]).astype(np.float32)

    def _compute_width_px(self) -> np.ndarray:
        """Body width in pixels, estimated from minor axis."""
        if len(self.extent) == 0:
            return np.zeros(len(self.times), dtype=np.float32)
        return (2.0 * self.extent[:, 1]).astype(np.float32)

    def _compute_spine_length(self, relative: bool = False) -> np.ndarray:
        n = len(self.times)
        q = np.full(n, np.nan, dtype=np.float32)
        for i, s in enumerate(self.spine):
            if s is not None:
                q[i] = s.length()
        if relative:
            mean_l = np.nanmean(q)
            if mean_l > 0:
                q /= mean_l
        return q

    def _compute_outline_width(self, relative: bool = False) -> np.ndarray:
        """Estimate width from outline (mean minor-axis of bounding ellipse)."""
        return self._compute_width_px()  # falls back to extent-based

    def _compute_end_wiggle(self) -> np.ndarray:
        """
        Kink angle at head/tail endpoints.
        Uses the angle between the outermost spine segment and the second one.
        Mirrors Dance.findEndWiggle.
        """
        n = len(self.times)
        q = np.zeros(n, dtype=np.float32)
        for i, s in enumerate(self.spine):
            if s is None or s.size() < 3:
                continue
            pts = s.points
            # Head kink: angle at second point
            a = pts[0] - pts[1]
            b = pts[2] - pts[1]
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a > 0 and norm_b > 0:
                cos_a = np.clip(np.dot(a, b) / (norm_a * norm_b), -1.0, 1.0)
                q[i] = float(np.arccos(cos_a))
        return q

    def _compute_body_wiggle(self) -> np.ndarray:
        """
        Total body curvature (sum of bend angles along spine).
        Mirrors Dance.findBodyWiggle.
        """
        n = len(self.times)
        q = np.zeros(n, dtype=np.float32)
        for i, s in enumerate(self.spine):
            if s is None or s.size() < 3:
                continue
            pts = s.points
            total_angle = 0.0
            for j in range(1, len(pts) - 1):
                a = pts[j - 1] - pts[j]
                b = pts[j + 1] - pts[j]
                na = np.linalg.norm(a)
                nb = np.linalg.norm(b)
                if na > 0 and nb > 0:
                    cos_v = np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)
                    total_angle += float(np.arccos(cos_v))
            q[i] = total_angle
        return q

    def _compute_direction_bias(self, mm_per_pixel: float, window: float,
                                min_speed: float) -> np.ndarray:
        """
        Directional bias: fraction of motion that is forward (range −1 to +1).
        Uses signed speed averaged over a window.
        Mirrors Dance.findDirectionBias.
        """
        sp = self.quantityIsSpeed(mm_per_pixel, window, signed=True)
        # Normalise: +1 = always forward, −1 = always backward
        total_sp = np.abs(sp)
        q = np.where(total_sp > min_speed, sp / np.maximum(total_sp, 1e-12),
                     np.float32(0.0))
        return q.astype(np.float32)

    def _compute_path_length(self, mm_per_pixel: float, window: float,
                             min_displacement: float) -> np.ndarray:
        """Cumulative path length up to each frame (mm)."""
        n = len(self.times)
        q = np.zeros(n, dtype=np.float32)
        if n < 2:
            return q
        x = self.centroid[:, 0].astype(float)
        y = self.centroid[:, 1].astype(float)
        dists = np.hypot(np.diff(x), np.diff(y)) * mm_per_pixel
        cumpath = np.concatenate([[0.0], np.cumsum(dists)])
        return cumpath.astype(np.float32)

    def _compute_direction_change(self, mm_per_pixel: float,
                                  window: float) -> np.ndarray:
        """Rate of change of direction (radians/s)."""
        return self._compute_angular_speed(window, signed=False)

    def _compute_orientation(self) -> np.ndarray:
        """Orientation angle (radians) from bearing vector."""
        if len(self.bearing) == 0:
            return np.zeros(len(self.times), dtype=np.float32)
        bx = self.bearing[:, 0].astype(float)
        by = self.bearing[:, 1].astype(float)
        return np.arctan2(by, bx).astype(np.float32)

    def _compute_velocity(self, mm_per_pixel: float, window: float,
                          component: int = 0) -> np.ndarray:
        """Velocity component (x=0 or y=1) in mm/s."""
        n = len(self.times)
        q = np.zeros(n, dtype=np.float32)
        if n < 2:
            return q
        c = self.centroid[:, component].astype(float)
        t = self.times.astype(float)
        for i in range(n):
            t_lo = t[i] - window / 2
            t_hi = t[i] + window / 2
            lo = max(0, np.searchsorted(t, t_lo))
            hi = min(n - 1, np.searchsorted(t, t_hi, side='right') - 1)
            dt = t[hi] - t[lo]
            if dt <= 0:
                continue
            q[i] = float((c[hi] - c[lo]) / dt * mm_per_pixel)
        return q

    def _compute_crab_speed(self, mm_per_pixel: float,
                            window: float) -> np.ndarray:
        """
        Lateral (crab) speed: velocity component perpendicular to body axis.
        Mirrors Dance.quantityIsCrab / findAbstractSpeed.
        """
        vx = self._compute_velocity(mm_per_pixel, window, 0)
        vy = self._compute_velocity(mm_per_pixel, window, 1)
        if len(self.bearing) < len(self.times):
            return np.zeros(len(self.times), dtype=np.float32)
        # Perpendicular to bearing = (−by, bx)
        bx = self.bearing[:len(vx), 0].astype(float)
        by = self.bearing[:len(vx), 1].astype(float)
        # Lateral component = vx*(−by) + vy*(bx)
        crab = (-vx * by + vy * bx)
        return crab.astype(np.float32)

    def _compute_qxfw(self, forward_fraction: float = 0.5) -> np.ndarray:
        """
        Qxfw metric: uses spine endpoint vectors to determine head orientation
        and thus forward/backward direction for each frame.
        Returns +1 (forward) or −1 (backward) or 0 (unknown).
        Mirrors Dance.findQxfw.
        """
        n = len(self.times)
        q = np.zeros(n, dtype=np.float32)
        if len(self.bearing) < n:
            return q
        for i in range(n):
            sp = self.spine[i] if i < len(self.spine) else None
            if sp is None or sp.size() < 2:
                continue
            pts = sp.points
            # Head vector: from centroid toward first spine point
            cx, cy = float(self.centroid[i, 0]), float(self.centroid[i, 1])
            hx = float(pts[0, 0]) - cx
            hy = float(pts[0, 1]) - cy
            # Bearing
            bx = float(self.bearing[i, 0])
            by = float(self.bearing[i, 1])
            dot = hx * bx + hy * by
            q[i] = 1.0 if dot >= 0 else -1.0
        return q

    # ------------------------------------------------------------------
    # Statistical summaries of computed quantities
    # ------------------------------------------------------------------

    def summarise_quantity(self, q: np.ndarray) -> Statistic:
        """Return a Statistic summarising array *q* (ignoring NaN)."""
        s = Statistic()
        finite = q[np.isfinite(q)]
        if len(finite):
            s.compute(finite)
        return s

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (f"Dance(id={self.ID}, frames={self.n_frames}, "
                f"duration={self.duration():.2f}s)")
