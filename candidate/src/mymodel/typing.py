from __future__ import annotations

import csv
import gzip
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.special import gammaln, softmax


# Shared across fields; no fitting to evaluator outcomes.
_PROFILE_CONCENTRATION = 20.0


def assigned_counts(path: Path, transcripts, cells) -> tuple[sparse.csr_matrix, int]:
    """Join Proseg foreground assignments by exported row ID, never by position."""
    size = len(transcripts.transcript_ids)
    seen = np.zeros(size, dtype=bool)
    owners = np.full(size, -1, dtype=np.int64)
    cell_rows = {cell.instance_id: index for index, cell in enumerate(cells)}
    assignment_count = 0
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"transcript_id", "gene", "assignment", "background"}.issubset(
            reader.fieldnames or ()
        ):
            raise ValueError("Proseg transcript metadata lacks assignment columns")
        for row in reader:
            index = int(row["transcript_id"])
            if not 0 <= index < size or seen[index]:
                raise ValueError("Proseg transcript IDs must be unique input row IDs")
            seen[index] = True
            if row["gene"] != transcripts.gene_ids[transcripts.gene_index[index]]:
                raise ValueError("Proseg transcript gene does not match its input row ID")
            background = row["background"].strip().lower()
            if background not in {"true", "false"}:
                raise ValueError("Proseg transcript background must be a boolean")
            assignment = row["assignment"].strip()
            if not assignment or background == "true":
                continue
            cell_id = int(assignment)
            if cell_id < 0:
                raise ValueError("Proseg cell IDs must be nonnegative")
            assignment_count += 1
            owners[index] = cell_rows.get(f"proseg-cell-{cell_id}", -1)
    if not seen.all():
        raise ValueError("Proseg transcript metadata is missing input row IDs")
    selected = owners >= 0
    counts = sparse.csr_matrix(
        (
            np.ones(int(selected.sum()), dtype=np.int64),
            (owners[selected], np.asarray(transcripts.gene_index)[selected]),
        ),
        shape=(len(cells), len(transcripts.gene_ids)),
    )
    return counts, assignment_count


def type_probabilities(counts: sparse.csr_matrix, gene_ids, reference) -> list[dict[str, float]]:
    """Dirichlet-multinomial type posteriors conditional on the shared gene panel."""
    labels = tuple(sorted(set(reference.cell_type_labels or ())))
    if not labels:
        raise ValueError("Proseg typing requires candidate-visible reference labels")
    probabilities = np.full((counts.shape[0], len(labels)), 1.0 / len(labels))
    matched = reference.matched_expression(gene_ids)
    reference_counts = matched.counts.astype(np.float64)
    expressed = np.asarray(reference_counts.sum(axis=0)).ravel() > 0
    # A single expressed shared gene has no compositional information.
    if expressed.sum() >= 2:
        reference_counts = reference_counts[:, expressed]
        depth = np.asarray(reference_counts.sum(axis=1)).ravel()
        reference_labels = np.asarray(reference.cell_type_labels)
        supported = [np.flatnonzero((reference_labels == label) & (depth > 0)) for label in labels]
        # Do not rule out a visible type merely because its reference has no
        # usable panel counts. Without comparable profiles retain the type prior.
        if all(len(rows) for rows in supported):
            normalized = sparse.diags(1.0 / np.maximum(depth, 1.0)) @ reference_counts
            ngenes = reference_counts.shape[1]
            profiles = np.stack(
                [
                    (np.asarray(normalized[rows].sum(axis=0)).ravel() + 1.0 / ngenes)
                    / (len(rows) + 1.0)
                    for rows in supported
                ]
            )
            spatial_columns = {gene: index for index, gene in enumerate(gene_ids)}
            columns = [
                spatial_columns[gene]
                for gene, keep in zip(matched.gene_ids, expressed, strict=True)
                if keep
            ]
            observed = counts[:, columns].tocsr()
            # The multinomial coefficient and total-concentration terms cancel
            # between types. Sparse evaluation avoids cells x types x genes state.
            log_likelihood = np.zeros_like(probabilities)
            for index, profile in enumerate(profiles):
                alpha = _PROFILE_CONCENTRATION * profile
                contributions = observed.copy().astype(np.float64)
                a = alpha[observed.indices]
                contributions.data = gammaln(a + observed.data) - gammaln(a)
                log_likelihood[:, index] = np.asarray(contributions.sum(axis=1)).ravel()
            probabilities = softmax(log_likelihood, axis=1)
    return [dict(zip(labels, map(float, row), strict=True)) for row in probabilities]
