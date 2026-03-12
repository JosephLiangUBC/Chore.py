"""
measures/omega.py
-----------------
Mirrors MeasureOmega.java – detection and characterisation of omega
(Ω) turns in C. elegans locomotion.

An omega turn is a deep body bend in which the worm forms a Ω shape.
Detection relies on:
  1. High body curvature (total bend angle)
  2. Low "straightness" of the spine (first PC explains little variance)
  3. Occurs during / near a reversal
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .dance import Dance
    from .reversal import Reversal


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class OmegaBend:
    """One omega-turn episode."""
    start_index: int
    end_index:   int
    start_time:  float
    end_time:    float
    peak_curvature: float = 0.0
    is_reversal_linked: bool = False

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


# ---------------------------------------------------------------------------
# Helper: spine straightness (mirrors MeasureOmega.getStraightness)
# ---------------------------------------------------------------------------

def _spine_straightness(spine_pts: np.ndarray) -> float:
    """
    Ratio of end-to-end distance to arc length.
    1 = perfectly straight, 0 = tightly coiled.
    """
    if spine_pts is None or len(spine_pts) < 2:
        return 1.0
    arc = float(np.sum(
        np.hypot(np.diff(spine_pts[:, 0]), np.diff(spine_pts[:, 1]))
    ))
    end_to_end = float(np.hypot(
        spine_pts[-1, 0] - spine_pts[0, 0],
        spine_pts[-1, 1] - spine_pts[0, 1]
    ))
    return end_to_end / arc if arc > 0 else 1.0


def _frame_curvature(spine_pts: np.ndarray) -> float:
    """Total absolute bend angle of the spine (sum of inter-segment angles)."""
    if spine_pts is None or len(spine_pts) < 3:
        return 0.0
    total = 0.0
    for i in range(1, len(spine_pts) - 1):
        a = spine_pts[i - 1] - spine_pts[i]
        b = spine_pts[i + 1] - spine_pts[i]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na > 0 and nb > 0:
            cos_v = np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)
            total += float(np.arccos(cos_v))
    return total


# ---------------------------------------------------------------------------
# MeasureOmega class
# ---------------------------------------------------------------------------

class MeasureOmega:
    """
    Detects omega turns from C. elegans locomotion data.

    Mirrors MeasureOmega.java.

    Parameters
    ----------
    curvature_threshold : minimum total body curvature (radians) to be
                         considered an omega turn (default ≈ π)
    straightness_threshold : maximum spine straightness during omega (default 0.5)
    min_duration        : minimum duration (seconds)
    """

    EXTENSION = ".omega"

    def __init__(self,
                 curvature_threshold:   float = math.pi * 0.8,
                 straightness_threshold: float = 0.5,
                 min_duration:          float = 0.1,
                 reversal_link:         bool  = True):
        self.curvature_threshold    = curvature_threshold
        self.straightness_threshold = straightness_threshold
        self.min_duration           = min_duration
        self.reversal_link          = reversal_link
        self._omega_map: Dict[int, List[OmegaBend]] = {}

    def validate_dancer(self, dance: "Dance") -> bool:
        has_spines = any(s is not None for s in dance.spine)
        return dance.n_frames >= 5 and has_spines

    def compute_dancer(self, dance: "Dance",
                       reversals: Optional[List["Reversal"]] = None
                       ) -> List[OmegaBend]:
        """Detect omega turns for one Dance."""
        if not self.validate_dancer(dance):
            return []

        n = dance.n_frames
        curv  = np.zeros(n, dtype=np.float32)
        strt  = np.ones(n,  dtype=np.float32)

        for i, sp in enumerate(dance.spine):
            if sp is not None:
                pts = sp.points
                curv[i]  = _frame_curvature(pts)
                strt[i]  = _spine_straightness(pts)

        # Omega frames: high curvature AND low straightness
        is_omega = (curv >= self.curvature_threshold) & \
                   (strt <= self.straightness_threshold)

        bends = self._find_episodes(dance, is_omega, curv)

        # Tag omega bends that overlap reversals
        if reversals is not None:
            rev_mask = np.zeros(n, dtype=bool)
            for rev in reversals:
                rev_mask[rev.start_index:rev.end_index] = True
            for bend in bends:
                if np.any(rev_mask[bend.start_index:bend.end_index]):
                    bend.is_reversal_linked = True

        self._omega_map[dance.ID] = bends
        return bends

    def compute_all(self, dances: Dict[int, "Dance"],
                    reversal_map: Optional[Dict[int, List]] = None
                    ) -> Dict[int, List[OmegaBend]]:
        for d in dances.values():
            revs = reversal_map.get(d.ID) if reversal_map else None
            self.compute_dancer(d, revs)
        return self._omega_map

    def get_omega_timecourse(self, dance: "Dance") -> np.ndarray:
        """Return (N,) array: 1 during omega bend, 0 otherwise."""
        q = np.zeros(dance.n_frames, dtype=np.float32)
        for bend in self._omega_map.get(dance.ID, []):
            q[bend.start_index:bend.end_index] = 1.0
        return q

    def quantifier_count(self) -> int:
        return 3

    def quantifier_title(self, i: int) -> str:
        return ["omega_count", "mean_omega_duration",
                "omega_curvature_peak"][i]

    def compute_dancer_quantity(self, dance: "Dance", index: int) -> float:
        bends = self._omega_map.get(dance.ID, [])
        if index == 0:
            return float(len(bends))
        if index == 1:
            durs = [b.duration for b in bends]
            return float(np.mean(durs)) if durs else 0.0
        if index == 2:
            peaks = [b.peak_curvature for b in bends]
            return float(np.max(peaks)) if peaks else 0.0
        return 0.0

    def to_dataframe(self):
        import pandas as pd
        rows = []
        for wid, bends in self._omega_map.items():
            for b in bends:
                rows.append({
                    'worm_id':    wid,
                    'start_time': b.start_time,
                    'end_time':   b.end_time,
                    'duration':   b.duration,
                    'peak_curv':  b.peak_curvature,
                    'rev_linked': b.is_reversal_linked,
                })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    def _find_episodes(self, dance: "Dance",
                       is_omega: np.ndarray,
                       curv: np.ndarray) -> List[OmegaBend]:
        times = dance.times.astype(float)
        bends = []
        n = len(is_omega)
        in_bend = False
        start_i = 0
        for i in range(n):
            if not in_bend and is_omega[i]:
                in_bend = True
                start_i = i
            elif in_bend and not is_omega[i]:
                in_bend = False
                t0 = float(times[start_i])
                t1 = float(times[i - 1])
                if t1 - t0 >= self.min_duration:
                    peak = float(np.max(curv[start_i:i]))
                    bends.append(OmegaBend(start_i, i, t0, t1, peak))
        if in_bend:
            t0 = float(times[start_i])
            t1 = float(times[-1])
            if t1 - t0 >= self.min_duration:
                peak = float(np.max(curv[start_i:]))
                bends.append(OmegaBend(start_i, n, t0, t1, peak))
        return bends
