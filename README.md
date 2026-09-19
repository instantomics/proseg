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
references for segmentation.

The task accepts whole-cell polygons and derives scoring coverage from them.
Typing uses Proseg's own foreground transcript assignments, joined to the
retained polygons by Proseg cell identity, even when these differ from geometric
coverage. Multipart output retains
the largest connected component, and deterministic clipping and overlap
resolution enforce the task's disjoint 2D polygon contract.

For labeled-reference fields, a post-segmentation Dirichlet-multinomial classifier
supplies explicit cell-type probabilities. Shared genes use the task's reference
name matching, including aggregation of duplicate reference gene names. Within
each type, reference cells contribute equally after shared-panel library-size
normalization; one uniform pseudo-cell smooths each mean profile. Equal type
priors avoid treating reference sampling proportions as tissue abundance. A fixed
concentration of 20 allows extra-multinomial variation rather than treating every
RNA as independent evidence for a hard label. These are wrapper modelling choices,
not an upstream Proseg typing method or empirically calibrated probabilities.

Empty cells and cells without usable shared-gene assignments retain the uniform
type prior. Fewer than two expressed shared genes, or a visible type lacking any
reference counts on that panel, also yields the prior: composition cannot support
a comparison in those cases. Sparse cells retain probabilistic uncertainty.
Missing genes and low counts do not establish a novel biological type, so the
wrapper adds no `UNKNOWN` class. Typing uses only the supplied labeled reference
and Proseg assignments; it cannot recognize absent reference types, correct
cross-platform expression bias, or resolve types indistinguishable on the panel.
The reference never influences geometry or transcript reassignment.

The initialization fixes the number of cells, so transcript-poor cells and
missed nuclear components cannot be recovered reliably. Transcript-density
initialization is a task-generic wrapper policy rather than an upstream-validated
Proseg mode. The current integration assumes native transcript and image
coordinates are micrometers. Proseg does not expose a random-seed option, so the
initializer is deterministic but inference itself is stochastic.

The maintained preset uses a deliberately short sampling schedule so all release
fields fit the task phase budget. It is a speed-oriented reference configuration,
not evidence that the sampler has converged; longer schedules may improve output
at additional computational cost.
