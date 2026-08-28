# Milestone 5 acceptance report

Date: 2026-08-28

Scope: Stage-B event learning with fixed oracle branch geometry. This milestone adds
frontier-only `WAIT`, `EXTEND`, `SPLIT`, and `STOP` prediction, weighted event cross
entropy, dynamic topology rollout, finite safety limits, classification metrics, and
topology-size/depth checks. Geometry velocity is not optimized and joint training is
deliberately deferred to Milestone 6.

## Supervision contract

- A sample is the complete partial tree immediately before one legal oracle event.
- Every present primitive uses its final ground-truth Bézier geometry; target velocity
  is zero and the geometry head is absent from the event model.
- The actual event target leaf receives `EXTEND`, `SPLIT`, or `STOP`; every other
  currently active frontier leaf receives `WAIT`. Non-frontier and padded tokens are
  excluded from the loss.
- The sole optimization objective is class-balanced event cross entropy. No geometry,
  Chamfer, Sholl, branch-angle, or morphology-statistic loss is used.
- The shared dynamic branch embedding, rooted-tree relative attention, and Tree
  Transformer architecture are reused, followed by a four-logit event head.

## Dataset and runtime

- Candidate SWCs: 40 M1 excitatory neurons from the fixed Milestone 4 manifest.
- Strictly accepted: 38; rejected: the same two non-parent duplicate-coordinate files
  recorded in the Milestone 4 report.
- File-disjoint split: 30 train / 8 validation.
- Training snapshots: 32 sampled event decisions per neuron and four legal randomized
  trajectories per neuron, resampled across epochs.
- Model: 64 hidden dimensions, two full-attention Tree Transformer layers, 100,064
  trainable parameters.
- Runtime: `BCI` Python 3.10, PyTorch 2.8.0+cu128, physical GPU 4 (RTX 5090), BF16 AMP,
  40 epochs, seed 0.

Training frontier-token support was `WAIT=10,393`, `EXTEND=575`, `SPLIT=174`, and
`STOP=211`. Inverse-frequency weights were normalized to mean one. This prevents the
event head from obtaining a misleadingly good result by always predicting `WAIT`.

## Held-out event classification

| Check | Result |
|---|---:|
| Majority class | WAIT |
| Majority-class macro-F1 | 0.2372 |
| Event-model macro-F1 | 0.3791 |
| Absolute macro-F1 margin | +0.1418 |
| Event-model accuracy | 0.6542 |
| WAIT F1 | 0.7880 |
| EXTEND F1 | 0.1609 |
| SPLIT F1 | 0.1587 |
| STOP F1 | 0.4086 |
| Non-finite loss/gradient/logit | 0 |
| Macro-F1 margin gate (at least +0.02) | pass |

Validation confusion matrix (rows are targets, columns are predictions in
`WAIT/EXTEND/SPLIT/STOP` order):

```text
[[1606, 417, 266,  92],
 [  69,  51,  17,  10],
 [  17,  16,  30,   0],
 [   3,   3,   2,  38]]
```

All four classes have true positives. `EXTEND` and `SPLIT` remain the weakest classes,
so Milestone 6 should monitor them separately instead of relying on aggregate accuracy.

## Dynamic oracle-geometry rollout

The topology rollout starts from one root primitive. At every discrete step, the
trained model samples an event on the current frontier. `EXTEND` inserts one token,
`SPLIT` inserts two, `STOP` closes a leaf, and `WAIT` leaves topology unchanged. New
branch geometry comes from an empirical training-set oracle-geometry bank; therefore
this is an event-model test, not an unconditional joint generator.

Inverse-frequency training changes the effective class prior. Before free rollout,
the logits are converted back to the empirical frontier-event prior. Without this
standard calibration, the same checkpoint over-grew to 2.19 times the held-out branch
count and 82.8% of samples reached `max_steps`. Calibration changes no model weights.

Post-calibration results for 64 seeded rollouts:

| Check | Generated | Held-out target | Ratio |
|---|---:|---:|---:|
| Mean branch count | 75.0625 | 97.3750 | 0.7709 |
| Branch-count standard deviation | 23.0528 | 29.3809 | 0.7846 |
| Mean maximum depth | 18.4531 | 22.2500 | 0.8294 |
| Maximum-depth standard deviation | 4.1380 | 3.3448 | 1.2371 |

- Natural `all_stopped` termination: 62/64.
- `max_steps=200` termination: 2/64.
- Forced-termination rate: 3.125% (reported diagnostically; the below-5% hard gate
  belongs to Milestone 6).
- Every rollout terminated in at most 200 steps; no loop was unbounded.
- Branch/depth mean-ratio gate `[0.5, 2.0]`: pass.

The sampler also enforces `max_branches=1000` and `max_depth=64`. These are emergency
limits only; the state representation remains variable-size and does not allocate
fixed topology slots.

## Go/No-Go

Milestone 5 is **GO** for Milestone 6. Held-out macro-F1 exceeds the majority baseline
by 0.1418, all event classes are represented in correct predictions, calibrated
dynamic rollouts have reasonable branch-count and depth distributions, and all
samples terminate finitely. The result does not establish unconditional generation:
joint geometry/event teacher forcing and scheduled sampling remain Milestone 6 work.
