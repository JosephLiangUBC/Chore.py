# `chore` – Python Port of the Choreography C. elegans Analysis Library

A Python 3 port of major **Choreography** (Chore.jar) analysis functionality by
Rex Kerr, focused on scriptable access to the Multi-Worm Tracker (MWT)
analysis pipeline.

This repository is made with the assistance of ChatGPT Codex and Anthropic Claude.

Current scope note: the core analysis library is substantially ported, but the
Java application's full CLI, plugin-loading, GUI, and some helper APIs are not
yet feature-complete. See [PARITY_AUDIT.md](/Users/Joseph/Desktop/Chore/choreography/PARITY_AUDIT.md).

---

## Installation

```bash
pip install -r requirements.txt
```

This installs:

- `numpy==2.2.6`
- `pandas==2.2.3`
- `scipy==1.15.3`
- `Pillow==11.2.1`
- `matplotlib==3.10.3`

Then place the `chore/` folder on your Python path or install as a package.

---

## Quick Start

```python
from chore import Choreography

# Load a full MWT output directory
c = Choreography(
    directory="20220601_120000",
    mm_per_pixel=0.025,       # spatial calibration (mm per pixel)
    speed_window=0.5,         # speed smoothing window (seconds)
    min_duration=5.0,         # discard tracks shorter than this
    min_move_mm=0.1,          # discard worms that barely move
)
c.load()

# Compute quantities for all worms
speeds    = c.get_quantity("speed")          # dict: worm_id → np.ndarray (mm/s)
lengths   = c.get_quantity("length")         # body length (pixels)
curves    = c.get_quantity("curve")          # body curvature (radians)

# Per-worm summary statistics
stats = c.summarise("speed")                 # dict: id → Statistic
pop   = c.population_statistic("speed")     # one Statistic across all worms

# Export
df = c.to_dataframe(["speed", "length", "curve", "bias"])
df.to_csv("results.csv", index=False)
c.write_dat("speed", output_dir="./out")     # one .dat file per worm
c.write_summary_table("summary.tsv")

# Render a spatial map directly from Choreography
img = c.render_map(
    quantity="speed",
    color_mapper="rainbow",
    backgrounder="green",
    dot_painter="line",
    width_px=1024,
    height_px=768,
)
img.save("map_speed.png")
```

---

## Spatial Map (`--map` equivalent)

`Choreography` now exposes the `DataMapper` workflow directly, so most map
rendering can happen from the same object you use for loading and analysis.
`DataMapper` is still available if you want lower-level control.

```python
from chore import Choreography

# 1. Load the MWT output directory (same as any other analysis)
c = Choreography(
    directory="20220601_120000",   # path to folder containing .summary / .blob files
    mm_per_pixel=0.025,
    min_duration=5.0,
    min_move_mm=0.1,
)
c.load()

# 2. Render all tracks coloured by speed
img = c.render_map(
    quantity="speed",              # any output code: 'speed', 'time', 'curve', etc.
    color_mapper="grayscale",        # "rainbow" or "sunset", "spatter", "grayscale"
    backgrounder="black",          # "green" or "white", "black"
    dot_painter="line",            # "line" or "dot", "circle", "spot", "identity"
    width_px=1024,
    height_px=768,
)
img.save("map_speed.png")

# 3. Add a colour scale bar
img_with_bar = c.add_map_colorbar(
    img,
    "rainbow",
    v_min=0.0,
    v_max=0.5,
    label="speed (mm/s)",
)
img_with_bar.save("map_speed_bar.png")

# 4. Snapshot at a specific time (e.g. 30 s into the recording, 5 s trail)
img_t = c.render_map_at_time(
    t=30.0,
    trail_s=5.0,
    quantity="speed",
    color_mapper="grayscale",
    backgrounder="black",
    width_px=1024,
    height_px=768,
)
img_t.save("map_t30.png")

# 5. Save an animated GIF of the full recording
c.save_map_gif(
    "map_animated.gif",
    fps=10.0,
    trail_s=2.0,
    quantity="speed",
    color_mapper="grayscale",
    backgrounder="black",
    width_px=800,
    height_px=600,
)

# 6. Interactive matplotlib viewer (equivalent to the Java GUI)
c.show_map(quantity="speed", colorbar=True)

# Optional: access the underlying DataMapper instance directly
dm = c.get_datamapper()
```

**Colour maps** — pass any of these preset strings as `color_mapper`:

| Class | Description |
|---|---|
| `"rainbow"` | Red → green → blue spectral ramp |
| `"sunset"` | Deep blue → warm orange/red |
| `"spatter"` | Distinct hue per worm (good for identity) |
| `"grayscale"` | Grayscale ramp |

**Backgrounds** — pass any of these preset strings as `backgrounder`:

| Class | Description |
|---|---|
| `"green"` | Dark green (default, mimics agar plate) |
| `"white"` | White |
| `"black"` | Black |

**Painters** — pass any of these preset strings as `dot_painter`:

| Class | Description |
|---|---|
| `"line"` | Connected trajectory lines with arrowheads |
| `"dot"` | Single pixel per frame |
| `"circle"` | Filled circle of fixed size |
| `"spot"` | Circle scaled by the quantity value |
| `"identity"` | Circle plus worm ID |
| `"frame"` | Circle plus frame number |
| `"value"` | Circle plus value label |

If you need non-default constructor arguments, you can still access `DataMapper`
directly through `c.get_datamapper()` and pass explicit mapper/painter objects.
The default map painter is `DotPainter()`, and `DotPainter()` is invisible by
default. Use `LinePainter()` explicitly if you want connected trajectory
segments, or `DotPainter(alpha=100)` if you want faint points.

> **Rendering tiers**: at high zoom (`pixel_size` < 2 screen pixels per data pixel) the
> body outline or spine is drawn; at medium zoom the bearing axis is shown; at low
> zoom a crosshair dot is used — exactly matching the Java `--map` behaviour.

---

## Module Reference

### `Choreography` — Main orchestrator

```python
Choreography(
    directory       = None,        # path to MWT output folder or .zip
    mm_per_pixel    = 0.025,       # mm per pixel
    speed_window    = 0.5,         # speed-smoothing window (s)
    min_time        = 0.0,         # discard data before this time (s)
    max_time        = inf,         # discard data after this time (s)
    min_move_mm     = 0.0,         # minimum total travel (mm)
    min_duration    = 0.0,         # minimum track length (s)
    quiet           = False,
    stimulus_events = None,        # (N,2) array of [on_time, off_time]
)
```

#### Loading
| Method | Description |
|--------|-------------|
| `c.load(directory=None)` | Load from MWT directory |
| `c.load_from_dict(dances)` | Load from `{worm_id: Dance}` dict |

#### Quantities  (mirrors `-o` output in original Chore)

All return `dict[worm_id → np.ndarray]`.

```python
c.get_quantity(code, **kwargs)
```

| Code | Name | Description |
|------|------|-------------|
| `'f'` | `frame` | Frame index |
| `'t'` | `time` | Time (seconds) |
| `'s'` | `speed` | Centroid speed (mm/s) |
| `'S'` | `angular_speed` | Angular speed (rad/s) |
| `'l'` | `length` | Body length (pixels) |
| `'L'` | `rel_length` | Relative body length |
| `'w'` | `width` | Body width (pixels) |
| `'W'` | `rel_width` | Relative body width |
| `'a'` | `aspect` | Aspect ratio (L/W) |
| `'A'` | `rel_aspect` | Relative aspect ratio |
| `'m'` | `midline` | Spine (midline) length |
| `'k'` | `kink` | End-wiggle / kink angle (rad) |
| `'b'` | `bias` | Directional bias (−1 to +1) |
| `'p'` | `path` | Cumulative path length (mm) |
| `'c'` | `curve` | Body curvature (rad) |
| `'d'` | `dir_change` | Direction change rate (rad/s) |
| `'x'` | `loc_x` | X position (pixels) |
| `'y'` | `loc_y` | Y position (pixels) |
| `'vx'`| `vel_x` | X velocity (mm/s) |
| `'vy'`| `vel_y` | Y velocity (mm/s) |
| `'T'` | `theta` | Body orientation (rad) |
| `'C'` | `crab` | Lateral (crab) speed (mm/s) |
| `'q'` | `qxfw` | Head orientation indicator (+1/−1) |
| `'1'`–`'4'` | `stim1`–`stim4` | Stimulus on/off mask |

#### Statistics

```python
c.summarise("speed")                    # per-worm → Statistic
c.population_statistic("speed", "mean") # single population Statistic
```

#### Filtering

```python
c.filter_by_duration(min_s=5.0, max_s=600.0)
c.filter_by_displacement(min_mm=0.5)
c.apply_filter(lambda d: d.n_frames > 100)
```

#### Plugins

```python
mr = c.run_reversal_analysis(min_duration=0.2, min_distance=0.01)
mo = c.run_omega_analysis(curvature_threshold=2.5)
es = c.run_eigenspine(n_components=3, n_spine_points=11)
cv = c.run_curvature_analysis(span=0.5)
c.respine(n_points=11)          # resample spines in-place
c.reoutline()                   # smooth outlines in-place
c.orient_spines()               # flip spines to head-first
```

#### Data maps

```python
img = c.render_map(quantity="speed")
img_t = c.render_map_at_time(t=30.0, trail_s=5.0, quantity="speed")
c.save_map_gif("map.gif", quantity="speed", fps=10.0)
c.show_map(quantity="speed", colorbar=True)
dm = c.get_datamapper()         # optional low-level access
```

---

### `Dance` — Per-worm track

```python
d = Dance(worm_id=1)
# Key arrays:
d.times     # (N,) float32 – timestamps (s)
d.frames    # (N,) int32   – frame indices
d.area      # (N,) float32 – body area (px²)
d.centroid  # (N,2) float32 – centroid (x, y) in px
d.bearing   # (N,2) float32 – heading unit vector
d.extent    # (N,2) float32 – (major, minor) half-axes (px)
d.spine     # list[SpineData | None]
d.outline   # list[OutlineData | None]
```

Computed quantity methods (same as `Choreography.get_quantity` but for one worm):

```python
d.quantityIsSpeed(mm_per_pixel, window, signed=False)
d.quantityIsAngularSpeed(mm_per_pixel, window)
d.quantityIsLength(relative=False)
d.quantityIsWidth(relative=False)
d.quantityIsAspect(relative=False)
d.quantityIsMidline(relative=False)
d.quantityIsKink()
d.quantityIsBias(mm_per_pixel, window, min_speed=0)
d.quantityIsPath(mm_per_pixel, window, min_displacement=0)
d.quantityIsCurve(relative=False)
d.quantityIsDirectionChange(mm_per_pixel, window)
d.quantityIsX(mm=False)
d.quantityIsY(mm=False)
d.quantityIsVx(mm_per_pixel, window)
d.quantityIsVy(mm_per_pixel, window)
d.quantityIsTheta(radians=True)
d.quantityIsCrab(mm_per_pixel, window)
d.quantityIsQxfw(forward_fraction=0.5)
d.quantityIsStim(events, stim_index)

# Reversal extraction
revs = d.extract_reversals(mm_per_pixel, window, require_forward_after=False)
```

---

### `Statistic` — Descriptive statistics

```python
s = Statistic(data)          # compute from array
s.average; s.deviation; s.median
s.first_quartile; s.last_quartile
s.minimum; s.maximum; s.n; s.jitter

# Robust (outlier-trimmed) compute
s.robust_compute(data, threshold=2.5)

# Statistical distribution functions (all also available as module-level functions)
Statistic.cdf_normal(x)
Statistic.icdf_normal(p)
Statistic.cdf_t_stat(t, df)
Statistic.cdf_f_stat(f, df1, df2)
Statistic.cdf_chi_sq(df, x)
Statistic.erf(x); Statistic.erf_inv(x)
Statistic.lngamma(x); Statistic.gamma(x)
```

---

### `Fitter` — Geometric fitting

```python
f = Fitter()
f.addL(x, y)        # accumulate point (linear sums)
f.addC(x, y)        # accumulate point (centred, automove)
f.subL(x, y)        # remove a point

lp  = f.fit_line()      # LinearParameters(slope, intercept, r_squared)
cp  = f.fit_circle()    # CircularParameters(cx, cy, r, residual)
sp  = f.fit_spot()      # SpotParameters(cx, cy, amplitude, sigma)
v1, v2, l1, l2 = f.principal_axes()  # eigenvectors + eigenvalues

# From a numpy array directly
lp = Fitter.line_fit(xy)       # xy is (N,2) array
cp = Fitter.circle_fit(xy)
v1, v2, l1, l2 = Fitter.eigen_axes(xy)
```

---

### Measurement Plugins

#### `MeasureReversal`
```python
mr = MeasureReversal(
    mm_per_pixel=0.025, window=0.5,
    min_duration=0.2,   min_distance=0.01,
    require_fwd=False,  time_range=None
)
results = mr.compute_all(dances)       # dict: id → ReversalResult
df      = mr.to_dataframe()            # per-worm summary
df_ev   = mr.to_events_dataframe()     # one row per event
tc      = mr.get_reversal_timecourse(dance)  # (N,) 0/1 array
```

#### `MeasureOmega`
```python
mo = MeasureOmega(
    curvature_threshold=2.5,    # radians
    straightness_threshold=0.5,
    min_duration=0.1
)
omega_map = mo.compute_all(dances, reversal_map=mr._reversal_map)
df        = mo.to_dataframe()
```

#### `Eigenspine`
```python
es = Eigenspine(n_components=3, n_spine_points=11)
es.fit(dances)                         # compute PCs from all spines
scores = es.transform(dance)           # EigenSpineResult
results = es.fit_transform(dances)     # fit + transform in one step
df = es.to_dataframe()
print(es.explained_variance_ratio())   # fraction variance per PC
```

#### `Curvaceous`
```python
cv = Curvaceous(span=0.5, disrupt=3.0)
data = cv.compute_dancer(dance)
# data keys: 'curvature', 'max_curv', 'head_curv', 'tail_curv', 'disrupted'
cv.compute_all(dances)
```

#### `Flux` (region crossing / gating)
```python
fl = Flux()
fl.add_rect(x0, y0, x1, y1)    # or add_circle(cx, cy, r)
fl.compute_all(dances)
inside = fl.get_inside_timecourse(dance)
gated  = fl.modify_quantity(dance, speed_array)   # zero outside region
df     = fl.to_dataframe()
```

#### `MeasureRadii`
```python
mr = MeasureRadii(n_points=11, internal=True)
mr.compute_all(dances)
width_matrix = mr.get_width_profile(dance)  # (n_frames, n_points)
```

#### Morphology plugins
```python
Respine(n_points=11).compute_all(dances)       # resample spines
Reoutline(despike=2, convex=0.3).compute_all(dances)  # smooth outlines
SpinesForward().compute_all(dances)            # head-first orientation

ex = Extract(extract_spine=True, extract_outline=True, extract_path=True)
spine_data = ex.extract_spine_data(dance)      # dict: frame → (N,3) array
ex.save_path(dance, "path.tsv", mm_per_pixel=0.025)
```

---

### I/O

```python
from chore.io import load_directory, read_summary, read_blob_file

dances = load_directory(
    "path/to/mwt_dir",
    mm_per_pixel=0.025,
    min_time=0.0, max_time=600.0,
    load_blobs=True,
    compute_statistics=True,
)

# Low-level readers
dances = read_summary("20220601_120000.summary")
dance  = read_blob_file("20220601_120000_00001.blob")
```

---

## File Format Notes

| Extension | Description |
|-----------|-------------|
| `.summary` | Tab-delimited: `id frame time x y area pob perim orient major minor` |
| `.blob`    | Per-worm: `% frame time` header + `cx cy area pob perim orient major minor [spine_n sx sy w…] [! outline_n ox oy…]` |
| `.zip`     | ZIP archive containing `.summary` + `.blob` files |
| `.dat`     | Output: two columns `time\tvalue`, one file per worm per metric |

---

## Dependencies

| Package | Usage |
|---------|-------|
| `numpy` | All array operations |
| `scipy` | Statistical distributions, Gaussian smoothing |
| `pandas` | DataFrame outputs |
| `Pillow` | Data map image rendering |
| `matplotlib` | Interactive `show_map()` viewer |

---

## `DataMapper` — the `--map` rendering pipeline

Full Python port of `DataMapper.java`. Renders worm tracks onto a 2-D spatial
canvas, colouring each position by any scalar quantity.  Replaces the
interactive Java `DataMapVisualizer` GUI.

```python
from chore import (DataMapper, RainbowMapper, SunsetMapper, SpatterMapper,
                   WhiteGrounder, GreenGrounder, BlackBackgrounder, LinePainter)

dm = DataMapper(mm_per_pixel=0.025)

# ── Full track map coloured by speed ─────────────────────────────────────────
img = dm.render(
    dances,                         # dict: worm_id → Dance
    quantity='speed',               # 'time' | 'speed' | 'length' | 'curve' | …
    v_min=0.0, v_max=0.5,           # normalisation range (None = auto)
    color_mapper=RainbowMapper(),   # RainbowMapper | SunsetMapper | SpatterMapper
    backgrounder=WhiteGrounder(),   # WhiteGrounder | GreenGrounder | BlackBackgrounder
    dot_painter=LinePainter(),      # LinePainter | CirclePainter | SpotPainter | …
    width_px=1024, height_px=768,
)
img.save("speed_map.png")

# ── Snapshot at a specific time (with history trail) ─────────────────────────
img = dm.render_at_time(dances, t=30.0, trail_s=5.0,
                         color_mapper=SunsetMapper(),
                         width_px=800, height_px=800)

# ── Animated outputs ──────────────────────────────────────────────────────────
dm.save_gif(dances, "tracks.gif", fps=10.0, trail_s=2.0)
dm.save_frames(dances, "frames/", fps=10.0, trail_s=2.0)   # PNG series
dm.save_video(dances, "tracks.mp4", fps=10.0, trail_s=2.0) # requires ffmpeg

# ── Colourbar ─────────────────────────────────────────────────────────────────
img_cb = dm.add_colorbar_to_image(img, RainbowMapper(),
                                   v_min=0.0, v_max=0.5, label="mm/s")

# ── Overlays (outline / spine / bearing arrows) ───────────────────────────────
view = dm.render(dances, ...)          # or ViewRequest.auto_bounds(dances, w, h)
img = dm.overlay_outlines(img, dances, view, t=30.0)
img = dm.overlay_spines(img, dances, view, t=30.0)
img = dm.overlay_bearings(img, dances, view, t=30.0)

# ── Interactive matplotlib viewer (replaces DataMapVisualizer) ────────────────
dm.show(dances, quantity='speed', color_mapper=RainbowMapper(),
        backgrounder=GreenGrounder(), title="Speed Map")
```

### Color mappers

| Class | Java equivalent | Description |
|-------|----------------|-------------|
| `RainbowMapper` | `RainbowMapper` | red→yellow→green→cyan→blue→magenta cycle |
| `SunsetMapper` | `SunsetMapper` | deep blue→magenta→orange |
| `SpatterMapper(entries)` | `SpatterMapper` | pseudo-random distinct hues (good for per-worm ID) |
| `ColorMapper` | `ColorMapper` | base grayscale mapper |

### Backgrounds

| Class | Java equivalent | Colors |
|-------|----------------|--------|
| `BlackBackgrounder` | `Backgrounder` | black bg, white highlights |
| `GreenGrounder` | `Greengrounder` | dark green bg (MWT-classic look) |
| `WhiteGrounder` | `Whitegrounder` | white bg, black highlights |
| `ImageGrounder(path)` | `Imagegrounder` | PNG/JPEG backdrop |
| `DimImageGrounder(path)` | `Dimimagegrounder` | dimmed backdrop |

### Dot painters

| Class | Java equivalent | Draws |
|-------|----------------|-------|
| `LinePainter` | `LinePainter` | connected path + arrowheads |
| `CirclePainter(d)` | `CirclePainter` | solid circle of diameter d |
| `SpotPainter(d)` | `SpotPainter` | circle scaled by area |
| `IdentityPainter(d)` | `IdentityPainter` | circle + worm ID number |
| `FramePainter(d)` | `FramePainter` | circle + frame index |
| `ValuePainter(d)` | `ValuePainter` | circle + numeric value |
| `DotPainter` | `DotPainter` | single pixel (base) |

### Value / quantity codes

All codes from the main `Choreography.get_quantity()` table are supported:
`'time'`, `'speed'`, `'angular_speed'`, `'length'`, `'width'`, `'aspect'`,
`'curve'`, `'bias'`, `'path'`, `'kink'`, `'midline'`, `'dir_change'`,
`'loc_x'`, `'loc_y'`, `'vel_x'`, `'vel_y'`, `'theta'`, `'crab'`, `'qxfw'`,
`'area'`, `'frame'`.  A callable `(Dance) → np.ndarray` is also accepted.
