# Milestone 3 acceptance report

Date: 2026-08-27

Scope: analytic birth seeds, normalized branch age, parent-relative Bézier geometry
paths, constant oracle velocity targets, event-driven dynamic branch insertion, complete
oracle replay, SWC export, and continuous-growth visualization. Neural networks,
optimization, and learned geometry are explicitly out of scope until Milestone 4.

## Oracle contract

- The soma is a permanent root anchor. Before the first event the dynamic state has
  zero branch tokens.
- A branch exists only after its birth event. Existing indices are immutable and new
  branches are appended in event order.
- The birth seed has all control points at the current parent endpoint plus small
  isotropic noise. The root anchor or current parent endpoint remains the exact `P0`,
  so every snapshot is connected.
- Geometry is represented relative to the parent endpoint. For birth time `t_birth`,
  age is `clip((t - t_birth) / (1 - t_birth), 0, 1)` and control/radius parameters
  interpolate linearly from seed to data.
- Oracle velocity is constant in relative parameter space:
  `(data - seed) / (1 - t_birth)`.
- `STOP` changes frontier state but does not freeze the continuous geometry path; all
  born branches reach their data target at global time one.
- Final oracle geometry must equal the fitted Bézier target. Consequently any
  residual error against the source polyline is exactly the Milestone 1 fitting error.

## Real morphology

- Dataset: MorphGrower M1 excitatory source dataset
- File: `20190830_sample_6.SWC`
- Source nodes: 1,834
- Target Bézier primitives: 48
- Virtual normalization branches: 2
- Trajectories: 8
- Base seed: 20260827
- Birth noise standard deviation: 0.01 µm

## Results

| Check | Result |
|---|---:|
| Valid oracle replays | 8/8 |
| Recovered target primitives | 48/48 |
| Primitive topology exact | 8/8 |
| Exported SWC critical topology exact | 8/8 |
| Stable dynamic insertion indices | 8/8 |
| Maximum intermediate continuity error | 0 µm |
| Maximum final oracle control-point error | 2.5121e-15 µm |
| Maximum final oracle radius error | 0 µm |
| Maximum source Bézier RMSE | 1.4111 µm |
| Maximum source Bézier point error | 2.8809 µm |
| Active leaves after final event | 0 |
| MorphIO reload of replayed SWC | pass |
| JSON/SWC/SVG output | pass |

Automated property tests additionally replay 16 trajectories over eight randomly
generated rooted trees. They check exact final geometry, topology recovery, immutable
insertion indices, parent-before-child ownership, event-boundary births, continuous
parent attachment, analytic velocity integration, and frontier termination.

## Go/No-Go

The single-neuron M1-EXC MVP gate is **GO** for Milestone 4: the representation is
compact, fitting errors remain within the configured 1.5 µm RMSE and 3.0 µm maximum
error thresholds, multiple trajectories are stable, and oracle replay introduces no
additional geometric or topological error. This result does not yet claim a
dataset-wide pass; that requires running the same gate over the complete training
corpus before large-scale model training.
