# Milestone 4 acceptance report

Date: 2026-08-27

Scope: Stage-A geometry flow with ground-truth topology events and birth times. This
milestone implements dynamic padding/masks, branch embeddings, rooted-tree positional
biases, a full-attention Tree Transformer, an 11-dimensional velocity head, the sole
velocity flow-matching objective, SO(3) augmentation, guarded training, checkpointing,
and train/validation metrics. Event prediction and joint sampling remain Milestone 5+
work.

## Model contract

- Each currently born branch contributes one token; padding is only to the largest
  current tree in the batch and never defines a fixed topology template.
- Geometry contains 9 parent-relative cubic control offsets and 2 radius values.
- Token features include current geometry, type, depth, age, global time, birth time,
  active/stopped state, parent-relative direction, control-polygon length, binary
  child position, and a root-to-branch path code.
- Full attention receives learned shortest-path, ancestor, descendant, and sibling
  biases. Parent indices remain append-only oracle indices.
- The only optimization loss is branch-average squared L2 oracle velocity error. No
  Chamfer, Sholl, angle, or hand-authored morphology loss was added.
- Final-target error is measured by extrapolating the predicted velocity from the
  sampled state to global time one.

## Dataset admission

- Official source: [MorphGrower M1 excitatory directory in the Brain Image Library](https://download.brainimagelibrary.org/3a/88/3a88a7687ab66069/excitatory/)
- Candidate manifest: `configs/experiment/milestone4_m1_exc_subset.txt`
- Downloaded candidates: 40
- Strictly accepted: 38
- Rejected: 2

The rejected files were `20190506_sample_4.SWC` and `20190722_sample_1.SWC`. Both
contain non-parent duplicate coordinates that cannot be merged without changing
topology, so the existing no-silent-topology-repair policy rejected them explicitly.
The SWCs and checkpoints remain external artifacts and are not committed to Git.

## Experiment A: fixed 32-neuron overfit

Configuration: 32 accepted real neurons, 3,668 total target primitives, two fixed
continuous snapshots per neuron, two available oracle trajectories, no epoch
resampling, no rotation, 80 epochs, 64 hidden dimensions, two Transformer layers,
and CPU execution.

| Check | Result |
|---|---:|
| Initial train velocity loss | 33.0543 |
| Best train velocity loss | 2.8437 |
| Train loss ratio | 0.0860 |
| Initial target control RMSE | 19.3868 µm |
| Best target control RMSE | 4.7332 µm |
| Maximum prediction / oracle-target scale | 1.0049 |
| Non-finite loss/gradient/prediction | 0 |
| 32-sample overfit gate (ratio ≤ 0.25) | pass |

## Experiment B: held-out validation with SO(3)

Configuration: 30 training neurons and 8 file-disjoint validation neurons, 3,509
training primitives, random legal trajectory/time resampling every epoch, independent
SO(3) rotation per tree, 40 epochs, the same 64-dimensional two-layer model, and CPU
execution.

| Check | Result |
|---|---:|
| Initial validation velocity loss | 10.8223 |
| Best validation velocity loss | 3.0744 |
| Initial validation control RMSE | 14.6292 µm |
| Best validation control RMSE | 8.7914 µm |
| Validation control error ratio | 0.6009 |
| Maximum prediction / oracle-target scale | 0.0700 |
| Non-finite loss/gradient/prediction | 0 |
| SO(3) augmentation exercised | yes |
| Validation geometry error decreased | pass |

Gradient norms are checked before clipping and every optimizer update is clipped to
1.0. Relative-coordinate outputs never predict branch starts: each child `P0` is
derived from its current parent endpoint, so geometry integration cannot disconnect a
child from its parent.

PyTorch 2.8 was available in a Python 3.10 environment. The execution container did
not expose a CUDA device, so the CPU path was used and AMP was disabled automatically;
CUDA selection and AMP/gradient-scaling paths are implemented but not claimed as
hardware-validated in this report.

## Go/No-Go

Milestone 4 is **GO** for Milestone 5. The model has sufficient capacity to reduce
fixed 32-neuron loss by more than 90%, improves file-disjoint validation geometry
under trajectory and SO(3) augmentation, preserves continuity by construction, and
shows no numerical failure. This does not yet demonstrate unconditional generation:
topology events remain oracle-provided until the event model is implemented.
