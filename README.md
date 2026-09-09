# Proseg

This reference wraps [Proseg](https://github.com/dcjones/proseg) 3.2.0
(`e7df1eace923ce4c6ec70b2c597c5d126aa3db88`) for the Iomix `segmentation`
task. The upstream crate, its locked dependency graph, Rust 1.88.0, and the
Linux binary wheel are pinned. Proseg is run as its multithreaded Rust CLI; the
wrapper only supplies candidate-visible inputs and converts its output to the
task contract.

Proseg requires an initial cell count and approximate locations. When a nuclear
image is available, the wrapper applies one fixed threshold-and-components
initializer and supplies the resulting mask. Otherwise it initializes cells
from fixed-scale, gene-agnostic transcript-density maxima. It never uses vendor
cell assignments, evaluator truth, source comparator outputs, or expression
references for segmentation. Labeled-reference scopes use the authorized
reference only after segmentation to provide the labels required by the task.

The task accepts whole-cell polygons and derives transcript assignments from
them. Proseg's own assignments are therefore counted in runtime diagnostics but
are not substituted for the task-derived assignments. Multipart output retains
the largest connected component, and deterministic clipping and overlap
resolution enforce the task's disjoint 2D polygon contract.

The initialization fixes the number of cells, so transcript-poor cells and
missed nuclear components cannot be recovered reliably. Transcript-density
initialization is a task-generic wrapper policy rather than an upstream-validated
Proseg mode. The current integration assumes native transcript and image
coordinates are micrometers. Proseg does not expose a random-seed option, so the
initializer is deterministic but inference itself is stochastic.
