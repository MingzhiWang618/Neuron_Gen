# Birth–Death ArborFlow

Birth–Death ArborFlow is a generative model for variable-size, rooted 3D neuronal
trees. It represents morphology as a dynamically growing tree of branch primitives,
combining continuous geometry flow with discrete `EXTEND`, `SPLIT`, and `STOP`
events.

The repository is being implemented milestone by milestone. Milestones 1–4 now
provide an invertible branch representation, legal randomized destruction/growth
trajectories, analytic oracle replay, and a geometry flow trained with oracle events.

## Current status: Milestone 4 — geometry flow with oracle events

Implemented:

- strict SWC parsing and deterministic writing;
- structural checks for IDs, parents, roots, connectivity, and cycles;
- geometry checks for non-finite values, radii, duplicate coordinates, and
  zero-length edges;
- conservative removal of zero-length parent–child samples with explicit audit logs;
- dataset filters for dimensionality, node count, and real branch count;
- a batch preparation CLI and unit tests.
- maximal-path branch decomposition with exact SWC provenance;
- deterministic multifurcation normalization using removable virtual branches;
- parent-relative cubic Bézier fitting with error/length-driven continuation splits;
- SWC reconstruction with virtual branches removed;
- dependency-free XY/XZ/YZ SVG comparison plots.
- legal single-continuation and terminal-sibling pruning actions;
- configurable 40/30/20/10 uniform/deep/short/long trajectory sampling;
- exact reversal into `EXTEND`, `SPLIT`, and `STOP` events;
- continuous event-time resampling with strict parent-before-child order;
- compact per-trajectory JSON and dependency-free SVG snapshots.
- near-zero isotropic birth seeds at the current parent endpoint;
- normalized branch age and analytic parent-relative geometry interpolation;
- exact constant velocity targets for Bézier controls and radii;
- variable-size `EmbeddedTreeState` snapshots with append-only dynamic indices;
- event-boundary replay with continuous parent-child attachment;
- exact final Bézier recovery plus topology-preserving SWC export;
- JSON, SWC, and dependency-free SVG oracle replay artifacts.
- variable-length PyTorch batches with masks and append-only parent indices;
- branch/type/depth/child-position/root-path embeddings;
- shortest-path and ancestor/descendant/sibling attention biases;
- a full-attention Tree Transformer and per-branch geometry velocity head;
- the single flow-matching velocity objective from the task book;
- random SO(3) rotation, gradient clipping, AMP support, checkpoints, and metrics.

Topology errors are never repaired automatically. Ambiguous duplicate coordinates
that are not a parent–child pair are reported as errors instead of being silently
merged.

## Quick start

The core data layer only needs NumPy and Python 3.10+. Milestone 4 training additionally
requires the `model` extra, which installs PyTorch and the planned experiment stack.

```bash
pip install -e .
pip install -e '.[model]'
python -m unittest discover -s tests -v
python scripts/prepare_data.py input_swcs/ --output data/cleaned
python scripts/inspect_data.py data/cleaned/swcs/example.swc \
  --output outputs/milestone1/example
python scripts/build_trajectories.py data/cleaned/swcs/example.swc \
  --output outputs/milestone2/example \
  --num-trajectories 8 \
  --seed 0
python scripts/replay_oracle.py data/cleaned/swcs/example.swc \
  --output outputs/milestone3/example \
  --num-trajectories 8 \
  --seed 0
CUDA_VISIBLE_DEVICES=4 conda run -n BCI python scripts/train_geometry_flow.py \
  data/m1_excitatory/ \
  --output outputs/milestone4/geometry_flow \
  --max-samples 40 \
  --epochs 40 \
  --device cuda
```

On this workstation, physical GPU 4 is the validated Milestone 4 device. Setting
`CUDA_VISIBLE_DEVICES=4` maps it to logical `cuda:0` inside PyTorch. Supported CUDA
devices use BF16 autocast; otherwise training falls back to FP16 autocast, while
`--no-mixed-precision` selects FP32.

The command writes cleaned SWCs under `data/cleaned/swcs/`, one JSON audit record per
input under `data/cleaned/logs/`, and a dataset-level `manifest.jsonl`. Invalid files
remain untouched and receive a failed audit record.

Useful options:

```bash
python scripts/prepare_data.py input_swcs/ \
  --output data/cleaned \
  --min-nodes 30 \
  --min-real-branches 5 \
  --max-real-branches 1000 \
  --require-3d
```

Use `--no-repair-zero-length` to make zero-length edges fatal. `--require-3d`
requires the samples' affine coordinate span to have rank three; it is disabled by
default in the CLI so small synthetic fixtures remain useful.

## Development order

1. SWC cleaning (complete)
2. branch decomposition, binary normalization, and Bézier primitives (Milestone 1)
3. pruning/growth trajectories (Milestone 2 complete)
4. oracle replay (Milestone 3 complete)
5. dynamic batching, Tree Transformer, and geometry flow (Milestone 4 complete)
6. event model with oracle geometry (next)
7. joint sampling

The recorded real-data gate is available in
[`docs/milestone1_acceptance.md`](docs/milestone1_acceptance.md).
The trajectory gate is recorded in
[`docs/milestone2_acceptance.md`](docs/milestone2_acceptance.md).
The oracle replay gate is recorded in
[`docs/milestone3_acceptance.md`](docs/milestone3_acceptance.md).
The geometry-flow gate is recorded in
[`docs/milestone4_acceptance.md`](docs/milestone4_acceptance.md).
