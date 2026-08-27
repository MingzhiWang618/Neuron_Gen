# Milestone 2 acceptance report

Date: 2026-08-27

Scope: legal leaf-pruning destruction, multiple randomized trajectories, exact growth
reversal into `EXTEND`, `SPLIT`, and `STOP`, continuous event-time resampling, compact
JSON export, and SVG trajectory visualization. Geometry interpolation and oracle replay
remain Milestone 3 work.

## Event contract

- A single terminal child can be removed only when it is its parent's sole child. Its
  reverse event is `EXTEND`.
- Two branches can be removed together only when they are terminal siblings with the
  same parent. Their reverse event is `SPLIT`.
- Every target terminal primitive receives exactly one `STOP` event.
- The soma/root anchor is permanent and is not a generated primitive. A root birth
  event therefore uses `parent_branch_id: null`.
- Child sets are atomic: an intermediate state contains either all children created by
  an event or none of them. This prevents orphan branches and makes virtual binary
  normalization branches removable without ambiguous partial splits.

## Real morphology

- Dataset: MorphGrower M1 excitatory source dataset
- File: `20190830_sample_6.SWC`
- Target Bézier primitives: 48
- Virtual normalization branches: 2
- Base seed: 20260827

## Results

| Check | Result |
|---|---:|
| Requested trajectories | 8 |
| Valid trajectories | 8/8 |
| Unique pruning orders | 8/8 |
| Pruning steps per trajectory | 40 |
| Growth events per trajectory | 49 |
| EXTEND events per trajectory | 32 |
| SPLIT events per trajectory | 8 |
| STOP events per trajectory | 9 |
| Orphan branches in any state | 0 |
| Invalid deletion actions | 0 |
| Recovered target primitives | 48/48 |
| Active leaves after final STOP | 0 |
| Same-seed reproducibility | pass |
| Parent-before-child event times | pass |
| SVG trajectory rendering | pass |

Across all eight trajectories, strategy labels were sampled 127 uniform, 93 deep, 62
short, and 38 long times. This is consistent with the configured 40/30/20/10 mixture
over 320 pruning steps.

Automated property tests additionally cover 12 randomly generated rooted trees with
three trajectories each. Tampered internal-branch deletion is explicitly rejected.

