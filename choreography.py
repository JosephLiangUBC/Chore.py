"""
choreography.py
---------------
The Choreography class – top-level orchestrator that mirrors the
Java Choreography class.

It ties together file I/O, track loading, per-worm quantity computation,
filtering, plugin management, and output writing.

Typical usage
-------------
>>> from chore import Choreography
>>> c = Choreography(directory="20220601_120000", mm_per_pixel=0.025)
>>> c.load()
>>> speed = c.get_quantity("speed")          # dict: id → np.ndarray
>>> stats  = c.summarise("speed")            # dict: id → Statistic
>>> c.write_dat("speed", output_dir="./out")
"""

from __future__ import annotations
import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable, Any, Tuple

from .dance import Dance
from .statistic import Statistic
from .fitter import Fitter
from .io import (load_directory, read_summary, write_dat_file,
                 write_summary_stats, DEFAULT_MM_PER_PIXEL)

# from .measures.reversal  import MeasureReversal
# from .measures.omega     import MeasureOmega
# from .measures.eigenspine import Eigenspine
# from .measures.curvaceous import Curvaceous
# from .measures.spatial   import (Flux, MeasureRadii, Respine,
#                                   Reoutline, Extract, SpinesForward)

from .reversal  import MeasureReversal
from .omega     import MeasureOmega
from .eigenspine import Eigenspine
from .curvaceous import Curvaceous
from .spatial   import (Flux, MeasureRadii, Respine, Reoutline, Extract, SpinesForward)
from .datamap import (
    DataMapper,
    ColorMapper, RainbowMapper, SunsetMapper, SpatterMapper,
    Backgrounder, BlackBackgrounder, GreenGrounder, WhiteGrounder,
    DotPainter, CirclePainter, SpotPainter, IdentityPainter,
    FramePainter, ValuePainter, LinePainter,
)

# ---------------------------------------------------------------------------
# Output type codes  (mirrors Choreography "ftnNpsSlLwWaAmkbcXYVUTCqd1234")
# ---------------------------------------------------------------------------
OUTPUT_CODES = {
    'f': 'frame',
    't': 'time',
    'n': 'number',
    'N': 'id',
    'p': 'path',
    's': 'speed',
    'S': 'angular_speed',
    'l': 'length',
    'L': 'rel_length',
    'w': 'width',
    'W': 'rel_width',
    'a': 'aspect',
    'A': 'rel_aspect',
    'm': 'midline',
    'k': 'kink',
    'b': 'bias',
    'c': 'curve',
    'd': 'dir_change',
    'x': 'loc_x',
    'y': 'loc_y',
    'vx': 'vel_x',
    'vy': 'vel_y',
    'T': 'theta',
    'C': 'crab',
    'q': 'qxfw',
    '1': 'stim1',
    '2': 'stim2',
    '3': 'stim3',
    '4': 'stim4',
}

# Statistic specifier codes  (mirrors Choreography output stat symbols)
STAT_CODES = {
    '%': 'mean',
    '^': 'sd',
    '_': 'min',
    '*': 'max',
    '-': 'median',
    '~': 'jitter',
    '1': 'q1',
    '3': 'q3',
}


# ---------------------------------------------------------------------------
# Choreography class
# ---------------------------------------------------------------------------

class Choreography:
    """
    Top-level C. elegans track analysis orchestrator.

    Mirrors the Java Choreography class in functionality while providing
    a Pythonic API for scripted analysis.

    Parameters
    ----------
    directory     : path to the MWT output directory or .zip archive
    mm_per_pixel  : spatial calibration (mm per pixel)
    speed_window  : time window for speed calculations (seconds)
    min_time      : discard track data before this time (seconds)
    max_time      : discard track data after  this time (seconds)
    min_move_mm   : minimum displacement to count as a moving worm (mm)
    min_move_body : same threshold in body lengths
    min_duration  : minimum track duration to include
    quiet         : suppress progress messages
    """

    # Default values match Choreography.java defaults
    DEFAULT_PIXEL_SIZE:   float = 0.025   # mm/pixel
    DEFAULT_SPEED_WINDOW: float = 0.5     # seconds
    DANCERS_PER_FILE:     int   = 1
    MAP_COLOR_PRESETS = {
        "rainbow": RainbowMapper,
        "sunset": SunsetMapper,
        "spatter": SpatterMapper,
        "grayscale": ColorMapper,
        "gray": ColorMapper,
    }
    MAP_BACKGROUND_PRESETS = {
        "green": GreenGrounder,
        "white": WhiteGrounder,
        "black": BlackBackgrounder,
        "none": BlackBackgrounder,
        "transparent": BlackBackgrounder,
    }
    MAP_PAINTER_PRESETS = {
        "line": LinePainter,
        "dot": DotPainter,
        "circle": CirclePainter,
        "spot": SpotPainter,
        "identity": IdentityPainter,
        "frame": FramePainter,
        "value": ValuePainter,
    }

    def __init__(self,
                 directory:       Optional[Union[str, Path]] = None,
                 mm_per_pixel:    float = DEFAULT_PIXEL_SIZE,
                 speed_window:    float = DEFAULT_SPEED_WINDOW,
                 min_time:        float = 0.0,
                 max_time:        float = float('inf'),
                 min_move_mm:     float = 0.0,
                 min_move_body:   float = 0.0,
                 min_duration:    float = 0.0,
                 quiet:           bool  = False,
                 stimulus_events: Optional[np.ndarray] = None):
        self.directory    = Path(directory) if directory else None
        self.mm_per_pixel = mm_per_pixel
        self.speed_window = speed_window
        self.min_time     = min_time
        self.max_time     = max_time
        self.min_move_mm  = min_move_mm
        self.min_move_body = min_move_body
        self.min_duration  = min_duration
        self.quiet         = quiet

        # Stimulus event table: (n_events, 2) → [on_time, off_time]
        # One per stimulation channel (up to 4 channels)
        self.stimulus_events: List[Optional[np.ndarray]] = [
            stimulus_events, None, None, None
        ]

        # Loaded data
        self.dances: Dict[int, Dance] = {}

        # Registered plugins (CustomComputation / CustomOutputModification)
        self._plugins: List[Any] = []

        # Per-worm statistics cache
        self._stats_cache: Dict[str, Dict[int, Statistic]] = {}

        # Global (population) statistics
        self.area_stat:          Statistic = Statistic()
        self.speed_stat:         Statistic = Statistic()
        self.length_stat:        Statistic = Statistic()
        self.angular_speed_stat: Statistic = Statistic()
        self.bias_stat:          Statistic = Statistic()
        self.curve_stat:         Statistic = Statistic()
        self._datamapper: Optional[DataMapper] = None

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load(self, directory: Optional[Union[str, Path]] = None,
             **kwargs) -> "Choreography":
        """
        Load track data from the MWT directory.

        Parameters
        ----------
        directory : override self.directory
        **kwargs  : forwarded to io.load_directory
        """
        d = Path(directory) if directory else self.directory
        if d is None:
            raise ValueError("No directory specified.")
        self.dances = load_directory(
            d,
            mm_per_pixel=self.mm_per_pixel,
            min_time=self.min_time,
            max_time=self.max_time,
            quiet=self.quiet,
            **kwargs
        )
        self._filter_dancers()
        return self

    def load_from_dict(self, dances: Dict[int, Dance]) -> "Choreography":
        """Directly set the dance dictionary (for programmatic use)."""
        self.dances = dances
        self._filter_dancers()
        return self

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def _filter_dancers(self) -> None:
        """Remove worms that don't meet quality criteria."""
        remove = []
        for wid, d in self.dances.items():
            if self.min_duration > 0 and d.duration() < self.min_duration:
                remove.append(wid)
                continue
            if self.min_move_mm > 0:
                total_path = d.path_length(0, d.n_frames - 1) * self.mm_per_pixel
                if total_path < self.min_move_mm:
                    remove.append(wid)
                    continue
        for wid in remove:
            del self.dances[wid]
        if not self.quiet and remove:
            print(f"Filtered out {len(remove)} worms.")

    def compute_and_check(self, dance: Dance) -> bool:
        """
        Compute basic statistics and return True if the worm is valid.
        Mirrors Choreography.computeAndCheck(Dance).
        """
        ok = dance.calc_basic_statistics()
        dance.calc_position_noise()
        return ok

    def take_attendance(self) -> Dict[int, List[int]]:
        """
        Return a dict mapping frame_number → list of worm IDs present.
        Mirrors Choreography.takeAttendance.
        """
        attendance: Dict[int, List[int]] = {}
        for wid, d in self.dances.items():
            for frame in d.frames:
                fi = int(frame)
                if fi not in attendance:
                    attendance[fi] = []
                attendance[fi].append(wid)
        return attendance

    # ------------------------------------------------------------------
    # Quantity computation  (mirrors Choreography.loadDancerWithData)
    # ------------------------------------------------------------------

    def get_quantity(self, code: str, **kwargs) -> Dict[int, np.ndarray]:
        """
        Compute a named quantity for all loaded worms.

        Parameters
        ----------
        code   : quantity name or single-letter code (see OUTPUT_CODES)
        kwargs : forwarded to the relevant Dance.quantityIs*() method

        Returns
        -------
        dict mapping worm_id → (N,) float32 array
        """
        name = OUTPUT_CODES.get(code, code)
        result = {}
        for wid, d in self.dances.items():
            q = self._compute_quantity(d, name, **kwargs)
            if q is not None:
                result[wid] = q
        return result

    def _compute_quantity(self, dance: Dance,
                          name: str, **kwargs) -> Optional[np.ndarray]:
        """Dispatch quantity computation for one Dance."""
        mpp = kwargs.pop('mm_per_pixel', self.mm_per_pixel)
        win = kwargs.pop('speed_window', self.speed_window)

        dispatch = {
            'time':        lambda: dance.quantityIsTime(**kwargs),
            'frame':       lambda: dance.quantityIsFrame(),
            'area':        lambda: dance.quantityIsArea(**kwargs),
            'speed':       lambda: dance.quantityIsSpeed(mpp, win, **kwargs),
            'angular_speed': lambda: dance.quantityIsAngularSpeed(mpp, win, **kwargs),
            'length':      lambda: dance.quantityIsLength(**kwargs),
            'rel_length':  lambda: dance.quantityIsLength(relative=True),
            'width':       lambda: dance.quantityIsWidth(**kwargs),
            'rel_width':   lambda: dance.quantityIsWidth(relative=True),
            'aspect':      lambda: dance.quantityIsAspect(**kwargs),
            'rel_aspect':  lambda: dance.quantityIsAspect(relative=True),
            'midline':     lambda: dance.quantityIsMidline(**kwargs),
            'kink':        lambda: dance.quantityIsKink(**kwargs),
            'bias':        lambda: dance.quantityIsBias(mpp, win, **kwargs),
            'path':        lambda: dance.quantityIsPath(mpp, **kwargs),
            'curve':       lambda: dance.quantityIsCurve(**kwargs),
            'dir_change':  lambda: dance.quantityIsDirectionChange(mpp, win),
            'loc_x':       lambda: dance.quantityIsX(**kwargs),
            'loc_y':       lambda: dance.quantityIsY(**kwargs),
            'vel_x':       lambda: dance.quantityIsVx(mpp, win, **kwargs),
            'vel_y':       lambda: dance.quantityIsVy(mpp, win, **kwargs),
            'theta':       lambda: dance.quantityIsTheta(**kwargs),
            'crab':        lambda: dance.quantityIsCrab(mpp, win, **kwargs),
            'qxfw':        lambda: dance.quantityIsQxfw(**kwargs),
            'stim1':       lambda: dance.quantityIsStim(self.stimulus_events[0], 0),
            'stim2':       lambda: dance.quantityIsStim(self.stimulus_events[1], 1),
            'stim3':       lambda: dance.quantityIsStim(self.stimulus_events[2], 2),
            'stim4':       lambda: dance.quantityIsStim(self.stimulus_events[3], 3),
        }
        fn = dispatch.get(name)
        if fn is None:
            warnings.warn(f"Unknown quantity '{name}'")
            return None
        return fn()

    # ------------------------------------------------------------------
    # Statistical summaries
    # ------------------------------------------------------------------

    def summarise(self, code: str,
                  robust: bool = False,
                  **kwargs) -> Dict[int, Statistic]:
        """
        Compute per-worm statistics for quantity *code*.

        Returns dict mapping worm_id → Statistic.
        """
        cache_key = f"{code}_{robust}"
        if cache_key in self._stats_cache:
            return self._stats_cache[cache_key]

        quantities = self.get_quantity(code, **kwargs)
        result = {}
        for wid, q in quantities.items():
            s = Statistic()
            finite = q[np.isfinite(q)]
            if len(finite) > 0:
                if robust:
                    s.robust_compute(finite)
                else:
                    s.compute(finite)
            result[wid] = s

        self._stats_cache[cache_key] = result
        return result

    def population_statistic(self, code: str,
                              stat: str = 'mean',
                              **kwargs) -> Statistic:
        """
        Compute a single population-level statistic by first taking
        per-worm means, then computing statistics across worms.

        Parameters
        ----------
        code : quantity name / code
        stat : which per-worm statistic to aggregate
               ('mean', 'median', 'sd', 'min', 'max')
        """
        per_worm = self.summarise(code, **kwargs)
        values = []
        for s in per_worm.values():
            if s.n > 0:
                v = getattr(s, {'mean': 'average', 'sd': 'deviation',
                                'median': 'median', 'min': 'minimum',
                                'max': 'maximum'}.get(stat, stat), None)
                if v is not None:
                    values.append(v)
        pop = Statistic()
        if values:
            pop.compute(np.array(values, dtype=float))
        return pop

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def write_dat(self, code: str,
                  output_dir: Union[str, Path] = '.',
                  prefix: str = '',
                  **kwargs) -> None:
        """
        Write per-worm time-course files (.dat) for quantity *code*.

        Files are named: <prefix><code>_<worm_id>.dat
        Mirrors Choreography's -o output.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        quantities = self.get_quantity(code, **kwargs)
        for wid, q in quantities.items():
            fname = output_dir / f"{prefix}{code}_{wid:05d}.dat"
            write_dat_file(fname, self.dances[wid], q)

    def write_summary_table(self, output_path: Union[str, Path],
                            codes: Optional[List[str]] = None,
                            **kwargs) -> pd.DataFrame:
        """
        Write a tab-delimited summary table (one row per worm, mean
        of each quantity per worm).

        Returns the DataFrame.
        """
        if codes is None:
            codes = ['speed', 'angular_speed', 'length', 'width',
                     'aspect', 'curve', 'bias', 'path']
        rows = []
        for wid, d in self.dances.items():
            row = {
                'worm_id':  wid,
                'duration': d.duration(),
                'n_frames': d.n_frames,
            }
            for code in codes:
                q = self._compute_quantity(d, OUTPUT_CODES.get(code, code), **kwargs)
                if q is not None:
                    finite = q[np.isfinite(q)]
                    if len(finite):
                        row[f"{code}_mean"]   = float(np.mean(finite))
                        row[f"{code}_sd"]     = float(np.std(finite, ddof=1))
                        row[f"{code}_median"] = float(np.median(finite))
                    else:
                        row[f"{code}_mean"] = row[f"{code}_sd"] = \
                            row[f"{code}_median"] = np.nan
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(output_path, sep='\t', index=False)
        return df

    def to_dataframe(self, codes: Optional[List[str]] = None,
                     **kwargs) -> pd.DataFrame:
        """
        Build a long-format DataFrame with columns:
        [worm_id, frame, time, <code1>, <code2>, …]
        """
        if codes is None:
            codes = ['speed', 'length', 'width', 'curve']
        rows = []
        for wid, d in self.dances.items():
            quantities = {}
            for code in codes:
                name = OUTPUT_CODES.get(code, code)
                q = self._compute_quantity(d, name, **kwargs)
                quantities[code] = q if q is not None else np.full(d.n_frames, np.nan)
            for i in range(d.n_frames):
                row = {'worm_id': wid,
                       'frame':   int(d.frames[i]),
                       'time':    float(d.times[i])}
                for code in codes:
                    row[code] = float(quantities[code][i])
                rows.append(row)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Plugin management
    # ------------------------------------------------------------------

    def load_plugin(self, plugin: Any,
                    run: bool = True) -> "Choreography":
        """
        Register and optionally run a computation plugin.

        Compatible plugins: MeasureReversal, MeasureOmega, Eigenspine,
        Curvaceous, Flux, MeasureRadii, Respine, Reoutline, Extract,
        SpinesForward, or any object with compute_all(dances).
        """
        self._plugins.append(plugin)
        if run and hasattr(plugin, 'compute_all'):
            plugin.compute_all(self.dances)
        return self

    def get_plugin(self, plugin_type: type) -> Optional[Any]:
        """Return the first registered plugin of the given type."""
        for p in self._plugins:
            if isinstance(p, plugin_type):
                return p
        return None

    # ------------------------------------------------------------------
    # Data map rendering
    # ------------------------------------------------------------------

    def get_datamapper(self, refresh: bool = False) -> DataMapper:
        """
        Return a DataMapper configured with this Choreography's calibration.
        """
        if refresh or self._datamapper is None:
            self._datamapper = DataMapper(mm_per_pixel=self.mm_per_pixel)
        return self._datamapper

    @classmethod
    def _resolve_color_mapper(cls, mapper: Optional[Union[str, ColorMapper]]) -> Optional[ColorMapper]:
        if mapper is None or isinstance(mapper, ColorMapper):
            return mapper
        if isinstance(mapper, str):
            factory = cls.MAP_COLOR_PRESETS.get(mapper.strip().lower())
            if factory is None:
                raise ValueError(f"Unknown color_mapper preset: {mapper}")
            return factory()
        return mapper

    @classmethod
    def _resolve_backgrounder(cls, backgrounder: Optional[Union[str, Backgrounder]]) -> Optional[Backgrounder]:
        if backgrounder is None or isinstance(backgrounder, Backgrounder):
            return backgrounder
        if isinstance(backgrounder, str):
            factory = cls.MAP_BACKGROUND_PRESETS.get(backgrounder.strip().lower())
            if factory is None:
                raise ValueError(f"Unknown backgrounder preset: {backgrounder}")
            return factory()
        return backgrounder

    @classmethod
    def _resolve_dot_painter(cls, dot_painter: Optional[Union[str, DotPainter]]) -> Optional[DotPainter]:
        if dot_painter is None or isinstance(dot_painter, DotPainter):
            return dot_painter
        if isinstance(dot_painter, str):
            factory = cls.MAP_PAINTER_PRESETS.get(dot_painter.strip().lower())
            if factory is None:
                raise ValueError(f"Unknown dot_painter preset: {dot_painter}")
            return factory()
        return dot_painter

    def render_map(self, **kwargs):
        """
        Render a composite data map for the currently loaded worms.

        All keyword arguments are forwarded to `DataMapper.render()`
        except that `dances` is supplied automatically from `self.dances`.
        """
        kwargs["color_mapper"] = self._resolve_color_mapper(kwargs.get("color_mapper"))
        kwargs["backgrounder"] = self._resolve_backgrounder(kwargs.get("backgrounder"))
        kwargs["dot_painter"] = self._resolve_dot_painter(kwargs.get("dot_painter"))
        return self.get_datamapper().render(self.dances, **kwargs)

    def render_map_at_time(self, t: float, trail_s: float = 5.0, **kwargs):
        """
        Render a time-localized snapshot of the current dataset.
        """
        kwargs["color_mapper"] = self._resolve_color_mapper(kwargs.get("color_mapper"))
        kwargs["backgrounder"] = self._resolve_backgrounder(kwargs.get("backgrounder"))
        kwargs["dot_painter"] = self._resolve_dot_painter(kwargs.get("dot_painter"))
        return self.get_datamapper().render_at_time(
            self.dances,
            t=t,
            trail_s=trail_s,
            **kwargs,
        )

    def save_map_gif(self,
                     output_path: Union[str, Path],
                     fps: float = 10.0,
                     trail_s: float = 2.0,
                     **kwargs) -> None:
        """
        Save an animated GIF for the currently loaded worms.
        """
        kwargs["color_mapper"] = self._resolve_color_mapper(kwargs.get("color_mapper"))
        kwargs["backgrounder"] = self._resolve_backgrounder(kwargs.get("backgrounder"))
        kwargs["dot_painter"] = self._resolve_dot_painter(kwargs.get("dot_painter"))
        self.get_datamapper().save_gif(
            self.dances,
            output_path=output_path,
            fps=fps,
            trail_s=trail_s,
            **kwargs,
        )

    def add_map_colorbar(self,
                         img,
                         color_mapper: Union[str, ColorMapper],
                         v_min: float = 0.0,
                         v_max: float = 1.0,
                         label: str = "",
                         width: int = 24,
                         margin: int = 6):
        """
        Append a colorbar to an image returned by `render_map()`.
        """
        color_mapper = self._resolve_color_mapper(color_mapper)
        return self.get_datamapper().add_colorbar_to_image(
            img,
            color_mapper=color_mapper,
            v_min=v_min,
            v_max=v_max,
            label=label,
            width=width,
            margin=margin,
        )

    def show_map(self,
                 quantity=None,
                 color_mapper: Optional[Union[str, Any]] = None,
                 backgrounder: Optional[Union[str, Any]] = None,
                 dot_painter: Optional[Union[str, Any]] = None,
                 width_px: int = 900,
                 height_px: int = 700,
                 title: str = "Choreography Data Map",
                 colorbar: bool = True,
                 **kwargs) -> None:
        """
        Show an interactive matplotlib viewer for the current dataset.
        """
        color_mapper = self._resolve_color_mapper(color_mapper)
        backgrounder = self._resolve_backgrounder(backgrounder)
        dot_painter = self._resolve_dot_painter(dot_painter)
        self.get_datamapper().show(
            self.dances,
            quantity=quantity,
            color_mapper=color_mapper,
            backgrounder=backgrounder,
            dot_painter=dot_painter,
            width_px=width_px,
            height_px=height_px,
            title=title,
            colorbar=colorbar,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Built-in analysis shortcuts
    # ------------------------------------------------------------------

    def run_reversal_analysis(self,
                              min_duration: float = 0.2,
                              min_distance: float = 0.01,
                              **kwargs) -> MeasureReversal:
        """Run reversal detection and return the MeasureReversal plugin."""
        mr = MeasureReversal(
            mm_per_pixel=self.mm_per_pixel,
            window=self.speed_window,
            min_duration=min_duration,
            min_distance=min_distance,
            **kwargs
        )
        mr.compute_all(self.dances)
        self._plugins.append(mr)
        return mr

    def run_omega_analysis(self, **kwargs) -> MeasureOmega:
        """Run omega-bend detection."""
        mo = MeasureOmega(**kwargs)
        mo.compute_all(self.dances)
        self._plugins.append(mo)
        return mo

    def run_eigenspine(self, n_components: int = 3,
                       n_spine_points: int = 11,
                       **kwargs) -> Eigenspine:
        """Fit and project eigenspine PCA."""
        es = Eigenspine(n_components=n_components,
                        n_spine_points=n_spine_points, **kwargs)
        es.fit_transform(self.dances)
        self._plugins.append(es)
        return es

    def run_curvature_analysis(self, **kwargs) -> Curvaceous:
        """Run detailed curvature analysis."""
        cv = Curvaceous(**kwargs)
        cv.compute_all(self.dances)
        self._plugins.append(cv)
        return cv

    def respine(self, n_points: int = 11, **kwargs) -> "Choreography":
        """Resample all spines to *n_points* points in-place."""
        rs = Respine(n_points=n_points, **kwargs)
        rs.compute_all(self.dances)
        self._plugins.append(rs)
        return self

    def reoutline(self, **kwargs) -> "Choreography":
        """Smooth all outlines in-place."""
        ro = Reoutline(**kwargs)
        ro.compute_all(self.dances)
        self._plugins.append(ro)
        return self

    def orient_spines(self, **kwargs) -> "Choreography":
        """Ensure all spines are head-first."""
        sf = SpinesForward(**kwargs)
        sf.compute_all(self.dances)
        self._plugins.append(sf)
        return self

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def n_worms(self) -> int:
        return len(self.dances)

    def get_dance(self, worm_id: int) -> Dance:
        return self.dances[worm_id]

    def worm_ids(self) -> List[int]:
        return sorted(self.dances.keys())

    def times(self) -> Dict[int, np.ndarray]:
        """Return time arrays for all worms."""
        return {wid: d.times for wid, d in self.dances.items()}

    def centroids(self) -> Dict[int, np.ndarray]:
        """Return centroid arrays (N, 2) for all worms."""
        return {wid: d.centroid for wid, d in self.dances.items()}

    def filter_by_duration(self, min_s: float = 0.0,
                           max_s: float = float('inf')) -> "Choreography":
        """Remove worms whose duration is outside [min_s, max_s] in place."""
        remove = [wid for wid, d in self.dances.items()
                  if not (min_s <= d.duration() <= max_s)]
        for wid in remove:
            del self.dances[wid]
        return self

    def filter_by_displacement(self, min_mm: float) -> "Choreography":
        """Remove worms that move less than *min_mm* total (mm)."""
        remove = [wid for wid, d in self.dances.items()
                  if d.path_length(0, d.n_frames - 1) * self.mm_per_pixel < min_mm]
        for wid in remove:
            del self.dances[wid]
        return self

    def apply_filter(self,
                     fn: Callable[[Dance], bool]) -> "Choreography":
        """
        Keep only worms for which fn(dance) returns True.
        """
        remove = [wid for wid, d in self.dances.items() if not fn(d)]
        for wid in remove:
            del self.dances[wid]
        return self

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (f"Choreography(n_worms={self.n_worms}, "
                f"mm_per_pixel={self.mm_per_pixel}, "
                f"speed_window={self.speed_window}s)")
