"""
measures/spatial.py
-------------------
Spatial / geometric analysis plugins:

  Flux          – region-crossing / gating analysis (mirrors Flux.java)
  MeasureRadii  – body radius profile (mirrors MeasureRadii.java)

measures/morphology.py  (also in this file)
  Respine       – spine resampling (mirrors Respine.java)
  Reoutline     – outline smoothing / refinement (mirrors Reoutline.java)
  Extract       – extract spine, outline, path data (mirrors Extract.java)
  SpinesForward – ensure consistent head-to-tail orientation (mirrors SpinesForward.java)
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ..dance import Dance

from ..spine_outline import SpineData, OutlineData
from ..utils import Vec2F, resample_polyline, cumulative_arc_length


# ===========================================================================
# Flux  (mirrors Flux.java)
# ===========================================================================

@dataclass
class FluxShape:
    """An axis-aligned region for flux/gating analysis."""
    x0: float; y0: float; x1: float; y1: float

    def contains(self, x: float, y: float) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    @staticmethod
    def from_circle(cx: float, cy: float, r: float) -> "FluxShape":
        return FluxShape(cx - r, cy - r, cx + r, cy + r)

    @staticmethod
    def from_rect(x0: float, y0: float, x1: float, y1: float) -> "FluxShape":
        return FluxShape(min(x0, x1), min(y0, y1),
                         max(x0, x1), max(y0, y1))


@dataclass
class FluxEvent:
    """A single region-crossing event."""
    worm_id: int
    time_enter: float
    time_exit:  float
    direction: int    # +1 = enter, -1 = exit

    @property
    def duration(self) -> float:
        return self.time_exit - self.time_enter


class Flux:
    """
    Detect when worms enter / exit defined spatial regions.
    Mirrors Flux.java (CustomOutputModification plugin).

    Parameters
    ----------
    shapes : list of FluxShape defining the region(s) of interest
    mode   : 'flux' = count crossings, 'gate' = filter by inside/outside,
             'report' = detailed events
    """

    EXTENSION = ".flux"

    def __init__(self,
                 shapes: Optional[List[FluxShape]] = None,
                 mode: str = 'report'):
        self.shapes = shapes or []
        self.mode   = mode
        self._events: Dict[int, List[FluxEvent]] = {}

    def add_rect(self, x0: float, y0: float,
                 x1: float, y1: float) -> None:
        self.shapes.append(FluxShape.from_rect(x0, y0, x1, y1))

    def add_circle(self, cx: float, cy: float, r: float) -> None:
        self.shapes.append(FluxShape.from_circle(cx, cy, r))

    def _inside(self, x: float, y: float) -> bool:
        return any(s.contains(x, y) for s in self.shapes)

    def compute_dancer(self, dance: "Dance") -> List[FluxEvent]:
        events = []
        cx = dance.centroid[:, 0]
        cy = dance.centroid[:, 1]
        times = dance.times.astype(float)

        was_inside = self._inside(float(cx[0]), float(cy[0]))
        enter_t = float(times[0]) if was_inside else None

        for i in range(1, dance.n_frames):
            now_inside = self._inside(float(cx[i]), float(cy[i]))
            if now_inside and not was_inside:
                enter_t = float(times[i])
            elif not now_inside and was_inside and enter_t is not None:
                events.append(FluxEvent(dance.ID, enter_t,
                                        float(times[i - 1]), +1))
                enter_t = None
            was_inside = now_inside

        self._events[dance.ID] = events
        return events

    def compute_all(self, dances: Dict[int, "Dance"]
                    ) -> Dict[int, List[FluxEvent]]:
        for d in dances.values():
            self.compute_dancer(d)
        return self._events

    def get_inside_timecourse(self, dance: "Dance") -> np.ndarray:
        """Return (N,) array: 1 if inside region, 0 otherwise."""
        q = np.zeros(dance.n_frames, dtype=np.float32)
        cx = dance.centroid[:, 0]
        cy = dance.centroid[:, 1]
        for i in range(dance.n_frames):
            if self._inside(float(cx[i]), float(cy[i])):
                q[i] = 1.0
        return q

    def modify_quantity(self, dance: "Dance",
                        quantity: np.ndarray) -> np.ndarray:
        """Zero out values outside the region (gate mode)."""
        inside = self.get_inside_timecourse(dance)
        return quantity * inside

    def to_dataframe(self):
        import pandas as pd
        rows = []
        for wid, evts in self._events.items():
            for e in evts:
                rows.append({'worm_id': wid,
                             'enter': e.time_enter,
                             'exit':  e.time_exit,
                             'duration': e.duration})
        return pd.DataFrame(rows)

    def quantifier_count(self) -> int:
        return 2

    def quantifier_title(self, i: int) -> str:
        return ['time_inside_region', 'crossing_count'][i]

    def compute_dancer_quantity(self, dance: "Dance", i: int) -> float:
        if dance.ID not in self._events:
            self.compute_dancer(dance)
        if i == 0:
            return float(np.sum(self.get_inside_timecourse(dance))
                         * (dance.duration() / max(dance.n_frames, 1)))
        return float(len(self._events[dance.ID]))


# ===========================================================================
# MeasureRadii  (mirrors MeasureRadii.java)
# ===========================================================================

@dataclass
class RadiiProfile:
    """Body radius profile for one frame."""
    radii: np.ndarray    # (N,) array of half-widths along spine


class MeasureRadii:
    """
    Compute the body radius (half-width) at each spine point.

    Mirrors MeasureRadii.java.  Uses the outline polygon to compute
    perpendicular distances from the spine.

    Parameters
    ----------
    n_points : number of spine sample points
    internal : use internal (spine-based) width estimate if no outline
    """

    EXTENSION = ".radii"

    def __init__(self, n_points: int = 11, internal: bool = True):
        self.n_points = n_points
        self.internal = internal
        self._library: Dict[int, List[Optional[RadiiProfile]]] = {}

    def validate_dancer(self, dance: "Dance") -> bool:
        has_spine = any(s is not None for s in dance.spine)
        has_outline = any(o is not None for o in dance.outline)
        return has_spine and (has_outline or self.internal)

    def compute_dancer(self, dance: "Dance"
                       ) -> List[Optional[RadiiProfile]]:
        profiles = []
        for i in range(dance.n_frames):
            sp  = dance.spine[i]   if i < len(dance.spine)   else None
            out = dance.outline[i] if i < len(dance.outline) else None
            if sp is None:
                profiles.append(None)
                continue
            radii = self._compute_radii(sp, out)
            profiles.append(RadiiProfile(radii))
        self._library[dance.ID] = profiles
        return profiles

    def compute_all(self, dances: Dict[int, "Dance"]) -> None:
        for d in dances.values():
            if self.validate_dancer(d):
                self.compute_dancer(d)

    def get_width_profile(self, dance: "Dance") -> np.ndarray:
        """
        Return (n_frames, n_points) array of body widths (2*radius)
        along the spine.
        """
        if dance.ID not in self._library:
            self.compute_dancer(dance)
        n = dance.n_frames
        result = np.full((n, self.n_points), np.nan, dtype=np.float32)
        for i, prof in enumerate(self._library[dance.ID]):
            if prof is not None:
                result[i] = np.interp(
                    np.linspace(0, 1, self.n_points),
                    np.linspace(0, 1, len(prof.radii)),
                    prof.radii
                ) * 2.0
        return result

    def quantifier_count(self) -> int:
        return self.n_points

    def quantifier_title(self, i: int) -> str:
        return f"width_at_{i / max(self.n_points - 1, 1):.2f}"

    def compute_dancer_quantity(self, dance: "Dance",
                                i: int) -> np.ndarray:
        profile = self.get_width_profile(dance)
        return profile[:, i]

    # ------------------------------------------------------------------
    def _compute_radii(self, sp: SpineData,
                       out: Optional[OutlineData]) -> np.ndarray:
        """Compute half-widths at each spine point."""
        if sp is None:
            return np.zeros(self.n_points)
        # Resample spine
        pts = resample_polyline(sp.points, self.n_points)

        if out is not None and out.size() >= 3:
            # Use outline: for each spine point, find nearest outline points
            # on each side and take their distances
            out_pts = out.points
            radii = np.zeros(self.n_points, dtype=np.float32)
            for j in range(self.n_points):
                p = pts[j]
                dists = np.hypot(out_pts[:, 0] - p[0],
                                 out_pts[:, 1] - p[1])
                radii[j] = float(np.min(dists))
            return radii
        else:
            # Fall back to extent-based (constant width)
            if hasattr(sp, 'widths') and sp.widths is not None:
                return np.interp(
                    np.linspace(0, 1, self.n_points),
                    np.linspace(0, 1, len(sp.widths)),
                    sp.widths / 2.0
                ).astype(np.float32)
            return np.zeros(self.n_points, dtype=np.float32)


# ===========================================================================
# Respine  (mirrors Respine.java)
# ===========================================================================

class Respine:
    """
    Re-sample all spine data to a fixed number of points.
    Mirrors Respine.java (CustomComputation plugin).

    Parameters
    ----------
    n_points  : target number of spine points after resampling
    fraction  : if given (0..1), resample to this fraction of original points
    subpixel  : if True, use sub-pixel interpolation
    """

    EXTENSION = ".respine"

    def __init__(self, n_points: int = 11,
                 fraction: float = 0.0,
                 subpixel: bool = True):
        self.n_points  = n_points
        self.fraction  = fraction
        self.subpixel  = subpixel

    def validate_dancer(self, dance: "Dance") -> bool:
        return any(s is not None for s in dance.spine)

    def compute_dancer(self, dance: "Dance") -> None:
        """In-place resampling of dance.spine."""
        for i, sp in enumerate(dance.spine):
            if sp is None:
                continue
            n_target = self.n_points
            if self.fraction > 0:
                n_target = max(2, int(sp.size() * self.fraction))
            dance.spine[i] = sp.resampled(n_target)

    def compute_all(self, dances: Dict[int, "Dance"]) -> None:
        for d in dances.values():
            if self.validate_dancer(d):
                self.compute_dancer(d)

    def quantifier_count(self) -> int:
        return 0

    def quantifier_title(self, i: int) -> str:
        return ""

    def compute_dancer_quantity(self, dance: "Dance", i: int) -> np.ndarray:
        return np.array([])


# ===========================================================================
# Reoutline  (mirrors Reoutline.java)
# ===========================================================================

class Reoutline:
    """
    Smooth and refine body outlines.
    Mirrors Reoutline.java (CustomComputation plugin).

    Parameters
    ----------
    despike : number of spike-removal passes
    convex  : convexity weight (higher = smoother convex hulls)
    concave : concavity weight (higher = tighter to body)
    blur    : smoothing kernel widths
    """

    EXTENSION = ".reoutline"

    def __init__(self,
                 despike: int   = 2,
                 convex:  float = 0.3,
                 concave: float = 0.1,
                 blur:    Optional[List[float]] = None):
        self.despike = despike
        self.convex  = convex
        self.concave = concave
        self.blur    = blur or [1.0, 2.0]

    def validate_dancer(self, dance: "Dance") -> bool:
        return any(o is not None for o in dance.outline)

    def compute_dancer(self, dance: "Dance") -> None:
        """In-place outline refinement."""
        for i, out in enumerate(dance.outline):
            if out is None or out.size() < 4:
                continue
            dance.outline[i] = self._smooth_outline(out)

    def compute_all(self, dances: Dict[int, "Dance"]) -> None:
        for d in dances.values():
            if self.validate_dancer(d):
                self.compute_dancer(d)

    def quantifier_count(self) -> int:
        return 0

    def quantifier_title(self, i: int) -> str:
        return ""

    def compute_dancer_quantity(self, dance: "Dance", i: int) -> np.ndarray:
        return np.array([])

    def _smooth_outline(self, out: OutlineData) -> OutlineData:
        """Gaussian smoothing of outline points."""
        from scipy.ndimage import gaussian_filter1d
        pts = out.points.copy()
        for sigma in self.blur:
            pts[:, 0] = gaussian_filter1d(pts[:, 0], sigma, mode='wrap')
            pts[:, 1] = gaussian_filter1d(pts[:, 1], sigma, mode='wrap')
        return OutlineData(pts)


# ===========================================================================
# Extract  (mirrors Extract.java)
# ===========================================================================

class Extract:
    """
    Extract raw spine, outline, and/or path data to text files.
    Mirrors Extract.java (CustomComputation plugin).
    """

    EXTENSION = ".extract"

    def __init__(self,
                 extract_spine:   bool = True,
                 extract_outline: bool = False,
                 extract_path:    bool = False):
        self.extract_spine   = extract_spine
        self.extract_outline = extract_outline
        self.extract_path    = extract_path

    def validate_dancer(self, dance: "Dance") -> bool:
        return dance.n_frames > 0

    def extract_spine_data(self, dance: "Dance") -> Dict[int, np.ndarray]:
        """
        Return dict mapping frame_index → (N, 2+1) spine array
        (columns: x, y, width).
        """
        result = {}
        for i, sp in enumerate(dance.spine):
            if sp is not None:
                pts = sp.points
                ws  = sp.widths.reshape(-1, 1)
                result[i] = np.hstack([pts, ws])
        return result

    def extract_outline_data(self, dance: "Dance"
                             ) -> Dict[int, np.ndarray]:
        """Return dict mapping frame_index → (M, 2) outline array."""
        result = {}
        for i, out in enumerate(dance.outline):
            if out is not None:
                result[i] = out.points.copy()
        return result

    def extract_path_data(self, dance: "Dance",
                          mm_per_pixel: float = 1.0) -> np.ndarray:
        """
        Return (N, 3) array: [time, x_mm, y_mm] for all frames.
        """
        cx = dance.centroid[:, 0] * mm_per_pixel
        cy = dance.centroid[:, 1] * mm_per_pixel
        return np.column_stack([dance.times, cx, cy])

    def save_spine(self, dance: "Dance",
                   output_path: Union[str, Path]) -> None:
        """Write spine data to a text file."""
        data = self.extract_spine_data(dance)
        with open(output_path, 'w') as fh:
            for frame_i, pts in data.items():
                fh.write(f"% {dance.frames[frame_i]} "
                         f"{dance.times[frame_i]:.4f}\n")
                for row in pts:
                    fh.write('\t'.join(f"{v:.4f}" for v in row) + '\n')

    def save_outline(self, dance: "Dance",
                     output_path: Union[str, Path]) -> None:
        """Write outline data to a text file."""
        data = self.extract_outline_data(dance)
        with open(output_path, 'w') as fh:
            for frame_i, pts in data.items():
                fh.write(f"% {dance.frames[frame_i]} "
                         f"{dance.times[frame_i]:.4f}\n")
                for row in pts:
                    fh.write(f"{row[0]:.4f}\t{row[1]:.4f}\n")

    def save_path(self, dance: "Dance",
                  output_path: Union[str, Path],
                  mm_per_pixel: float = 1.0) -> None:
        """Write centroid path to a text file."""
        arr = self.extract_path_data(dance, mm_per_pixel)
        np.savetxt(output_path, arr, delimiter='\t',
                   header='time\tx_mm\ty_mm', comments='')

    def quantifier_count(self) -> int:
        return 0

    def quantifier_title(self, i: int) -> str:
        return ""

    def compute_dancer_quantity(self, dance: "Dance", i: int) -> np.ndarray:
        return np.array([])


# ===========================================================================
# SpinesForward  (mirrors SpinesForward.java)
# ===========================================================================

class SpinesForward:
    """
    Ensure all spines are oriented head-first (consistent direction).

    Mirrors SpinesForward.java – uses the bearing (direction of motion)
    to determine which end of the spine is the head, then flips spines
    that are tail-first.
    """

    EXTENSION = ".spf"

    def __init__(self, flip_threshold: float = 0.0):
        """
        flip_threshold : minimum dot product of spine-direction and bearing
                         required to consider the spine correctly oriented.
                         Spines with dot product < flip_threshold are flipped.
        """
        self.flip_threshold = flip_threshold

    def validate_dancer(self, dance: "Dance") -> bool:
        return (any(s is not None for s in dance.spine)
                and len(dance.bearing) == dance.n_frames)

    def compute_dancer(self, dance: "Dance") -> None:
        """In-place orientation correction of dance.spine."""
        for i, sp in enumerate(dance.spine):
            if sp is None or sp.size() < 2:
                continue
            pts = sp.points
            # Spine direction: from first to last point
            spine_dir = pts[-1] - pts[0]
            norm = np.linalg.norm(spine_dir)
            if norm == 0:
                continue
            spine_dir /= norm

            # Bearing
            bx = float(dance.bearing[i, 0])
            by = float(dance.bearing[i, 1])

            dot = float(spine_dir[0] * bx + spine_dir[1] * by)
            if dot < self.flip_threshold:
                sp.flip()

    def compute_all(self, dances: Dict[int, "Dance"]) -> None:
        for d in dances.values():
            if self.validate_dancer(d):
                self.compute_dancer(d)

    def quantifier_count(self) -> int:
        return 0

    def quantifier_title(self, i: int) -> str:
        return ""

    def compute_dancer_quantity(self, dance: "Dance", i: int) -> np.ndarray:
        return np.array([])
