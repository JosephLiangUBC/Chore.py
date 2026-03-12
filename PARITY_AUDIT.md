# Chore.jar Parity Audit

This repository is a substantial Python 3 port of the analysis-facing parts of `Chore.jar`, but it is not yet a full feature-for-feature replica of the Java application.

Status date: 2026-03-12

## What is working

- Package import under Python 3.
- Core objects: `Choreography`, `Dance`, `Statistic`, `Fitter`.
- Primary per-frame quantities exposed by `Choreography.get_quantity()` and `Dance.quantityIs*()`.
- Built-in analysis classes by name: `MeasureReversal`, `MeasureOmega`, `Eigenspine`, `Curvaceous`, `Flux`, `MeasureRadii`, `Respine`, `Reoutline`, `Extract`, `SpinesForward`.
- Basic file loading for `.summary`, `.blob`, and ZIP-backed input in [io.py](/Users/Joseph/Desktop/Chore/choreography/io.py).
- Python-side rendering support in [datamap.py](/Users/Joseph/Desktop/Chore/choreography/datamap.py).

## Confirmed gaps vs `Choreography.class`

These public Java entry points are not currently replicated in [choreography.py](/Users/Joseph/Desktop/Chore/choreography/choreography.py):

- CLI parsing and help: `parseInput`, `parseOutputSpecification`, `printHelpMessage`, `main`
- Plugin loading by name: `requirePlugins`, `loadComputationPlugin`
- File-discovery workflow: `findFiles`, `targetDir`, `loadData`, `loadSummaryToArrays`
- Stimulus / genealogy ingestion: `loadTriggerTimes`, `loadGeneologyTable`
- Statistics/output pipeline: `recomputeOnlineStatistics`, `recomputeCustomOnlineStatistics`, `computeDataSkipJunk`, `loadDancerWithData`, `prepareSingle`, `preparePrinting`, `writeStatistics`, `showStatistics`, `showDataMap`
- Helper methods: `minTravel`, `indexNear`, `makeDataSpecifier`, `nfprintf`

## Confirmed gaps vs `Dance.class`

These public Java entry points are not currently replicated in [dance.py](/Users/Joseph/Desktop/Chore/choreography/dance.py):

- Native Java parser path: `readInputStream`
- Spine/orientation helpers: `alignSpines`, `alignAllSpines`
- Track trimming and ancestry: `trimData`, `findOriginsFates`
- Noise / QC helpers: `loadMinMax`, `loadNoise`, `findStatisticalWeirdness`, `estimateNoise`, `maximumExcursion`, `findFirstBeyond`
- Interpolation helpers: `seekNearT`, `seekTimeIndices`, `fracLoc`, `fracQuant`
- Segment helpers: `indexToSegment`, `getSegmentedDirection`
- Detailed directional analysis variants: `findDirectionBiasSegmented`, `findDirectionBiasUnsegmented`, `findDirectionChangeAtScale`, `findDirectionChangeUnsegmented`, `findDirectionChangeSegmented`
- Posture/confusion analysis: `findPostureConfusion`, `findDirectionConfusion`
- Spine/outline helpers: `findSpineLength`, `normalizeSpineLength`, `findOutlineWidth`, `getBodyAngles`, `getBestEndpoints`
- Reversal helpers used by Java plugins: `addRetro`, `hasBackwards`
- Quantity-state API from Java: `allUnload`, `quantityMult`, `quantityIs`, `quantityAlreadyIsCustom`

## Partial parity

- `DataMapper` is present, but the Java GUI/event model is not mirrored. The Python version is a rendering-oriented API, not a Swing application.
- Plugin classes exist, but the Java plugin lifecycle methods such as `initialize`, `desiredExtension`, and `computeDancerSpecial` are not consistently mirrored.
- Several implemented analysis methods appear intentionally simplified relative to the Java bytecode. Functional names match, but algorithmic equivalence has not yet been verified frame-for-frame against `Chore.jar`.
- `Dance.ready_multiscale()` exists, but Java method names and some quantity-state behavior differ.

## Python 3 compatibility fixes applied

- Fixed broken relative imports in:
  - [reversal.py](/Users/Joseph/Desktop/Chore/choreography/reversal.py)
  - [omega.py](/Users/Joseph/Desktop/Chore/choreography/omega.py)
  - [eigenspine.py](/Users/Joseph/Desktop/Chore/choreography/eigenspine.py)
  - [curvaceous.py](/Users/Joseph/Desktop/Chore/choreography/curvaceous.py)
  - [spatial.py](/Users/Joseph/Desktop/Chore/choreography/spatial.py)

Without these changes, `import choreography` failed under Python 3 with `ImportError: attempted relative import beyond top-level package`.

## Recommended next steps

1. Decide target scope: analysis-library parity, or full parity including CLI/plugin loading and GUI workflows.
2. Add fixture-based comparisons against `Chore.jar` outputs for representative datasets.
3. Implement the missing `Choreography` output/CLI pipeline before claiming full replication.
4. Fill in the missing `Dance` interpolation/QC helpers needed by the Java plugin surface.
