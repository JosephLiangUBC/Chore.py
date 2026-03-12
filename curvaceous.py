"""
measures/curvaceous.py
----------------------
Mirrors Curvaceous.java – time-resolved body curvature analysis.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .dance import Dance


class Curvaceous:
    """
    Compute per-frame body curvature metrics.

    Mirrors Curvaceous.java (CustomComputation plugin).

    Parameters
    ----------
    span     : fraction of body length over which curvature is measured
    disrupt  : disruption threshold (frames with curvature > disrupt*mean
               are flagged as disrupted)
    """

    EXTENSION = ".curv"

    def __init__(self, span: float = 0.5, disrupt: float = 3.0):
        self.span    = span
        self.disrupt = disrupt
        self._cache: Dict[int, Dict[str, np.ndarray]] = {}

    def validate_dancer(self, dance: "Dance") -> bool:
        return dance.n_frames >= 3 and any(s is not None for s in dance.spine)

    def compute_dancer(self, dance: "Dance"
                       ) -> Dict[str, np.ndarray]:
        """
        Compute per-frame curvature metrics.

        Returns dict with keys:
          'curvature'  – total body curvature (radians)
          'max_curv'   – maximum local curvature
          'head_curv'  – head-end curvature
          'tail_curv'  – tail-end curvature
          'disrupted'  – boolean mask of disrupted frames
        """
        n = dance.n_frames
        total_curv = np.full(n, np.nan, dtype=np.float32)
        max_curv   = np.full(n, np.nan, dtype=np.float32)
        head_curv  = np.full(n, np.nan, dtype=np.float32)
        tail_curv  = np.full(n, np.nan, dtype=np.float32)

        for i, sp in enumerate(dance.spine):
            if sp is None or sp.size() < 3:
                continue
            pts = sp.points
            angles = self._local_curvatures(pts)
            if len(angles) == 0:
                continue
            total_curv[i] = float(np.sum(np.abs(angles)))
            max_curv[i]   = float(np.max(np.abs(angles)))
            n_span = max(1, int(len(angles) * self.span))
            head_curv[i] = float(np.sum(np.abs(angles[:n_span])))
            tail_curv[i] = float(np.sum(np.abs(angles[-n_span:])))

        mean_c = float(np.nanmean(total_curv))
        disrupted = total_curv > (self.disrupt * mean_c)

        result = {
            'curvature': total_curv,
            'max_curv':  max_curv,
            'head_curv': head_curv,
            'tail_curv': tail_curv,
            'disrupted': disrupted.astype(np.float32),
        }
        self._cache[dance.ID] = result
        return result

    def compute_all(self, dances: Dict[int, "Dance"]) -> None:
        for d in dances.values():
            if self.validate_dancer(d):
                self.compute_dancer(d)

    def quantifier_count(self) -> int:
        return 5

    def quantifier_title(self, i: int) -> str:
        return ['curvature', 'max_curv', 'head_curv',
                'tail_curv', 'disrupted'][i]

    def compute_dancer_quantity(self, dance: "Dance", i: int) -> np.ndarray:
        if dance.ID not in self._cache:
            self.compute_dancer(dance)
        return list(self._cache[dance.ID].values())[i]

    @staticmethod
    def _local_curvatures(pts: np.ndarray) -> np.ndarray:
        """Signed bend angles at each interior spine point."""
        angles = []
        for j in range(1, len(pts) - 1):
            a = pts[j - 1] - pts[j]
            b = pts[j + 1] - pts[j]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na > 0 and nb > 0:
                cross = float(a[0] * b[1] - a[1] * b[0])
                dot   = float(np.dot(a, b))
                angles.append(math.atan2(cross, dot))
        return np.array(angles, dtype=np.float32)
