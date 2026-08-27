# Milestone 1 acceptance report

Date: 2026-08-27

Scope: SWC cleaning, branch decomposition, binary normalization, cubic Bézier
primitives, exact/approximate SWC reconstruction, and original-versus-reconstruction
visualization. Pruning trajectories and learned models are explicitly out of scope.

## Real morphology

- Dataset: MorphGrower M1 excitatory source dataset
- File: `20190830_sample_6.SWC`
- Official source:
  `https://download.brainimagelibrary.org/3a/88/3a88a7687ab66069/excitatory/20190830_sample_6.SWC`
- Source nodes: 1,834
- Source real branches: 15
- Coordinate affine rank: 3

## Results

| Check | Result |
|---|---:|
| SWC validation | pass |
| Exact SWC → branch → SWC round-trip | pass |
| Binary normalization invariants | pass |
| Binary denormalization exact round-trip | pass |
| Critical topology preserved after Bézier export | pass |
| Real bifurcation count preserved | pass |
| Parent-child continuity error | 1.90e-15 µm |
| Maximum primitive polyline length | ≤ 80 µm |
| Maximum Bézier RMSE | 1.4111 µm |
| Maximum Bézier point error | 2.8809 µm |
| Reconstructed real branches | 15 |
| MorphIO 3.4.0 reload | pass |

The configured limits are 1.5 µm RMSE, 3.0 µm maximum point error, and 80 µm
maximum primitive length. The morphology produced 15 real branches, 2 virtual
normalization branches, and 48 Bézier primitives. Virtual branches were removed at
SWC export.

## Automated coverage

The test suite additionally covers malformed SWCs, zero-length repairs, exact field
preservation, internal and root multifurcations, deterministic normalization, SWC type
transitions, length/error continuation splitting, trees deeper than 1,000 branch
levels, and 20 seeded random rooted trees.

