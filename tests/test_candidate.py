from __future__ import annotations

import csv
import gzip
import json
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from scipy import sparse
from shapely.geometry import Polygon

CANDIDATE_SOURCE = Path(__file__).parents[1] / "candidate" / "src"
sys.path.insert(0, str(CANDIDATE_SOURCE))


@dataclass(frozen=True)
class PolygonInstance:
    instance_id: str
    vertices: np.ndarray
    parent_cell_id: str | None = None
    type_probabilities: dict[str, float] | None = None


@dataclass(frozen=True)
class SegmentationPrediction:
    cells: tuple[PolygonInstance, ...]
    nuclei: tuple[PolygonInstance, ...]


@dataclass(frozen=True)
class TranscriptTable:
    transcript_ids: tuple[str, ...]
    gene_ids: tuple[str, ...]
    gene_index: np.ndarray
    coordinates: np.ndarray


@dataclass(frozen=True)
class ImageChannel:
    channel_id: str
    role: str
    image: np.ndarray
    origin_um: tuple[float, float]
    pixel_size_um: tuple[float, float]


segmentation = ModuleType("segmentation")
segmentation.__path__ = []
schema = ModuleType("segmentation.schema")
schema.MAX_POLYGON_VERTICES = 1_024
for _name, _value in {
    "ImageChannel": ImageChannel,
    "PolygonInstance": PolygonInstance,
    "SegmentationPrediction": SegmentationPrediction,
    "TranscriptTable": TranscriptTable,
}.items():
    setattr(schema, _name, _value)
segmentation.schema = schema
sys.modules["segmentation"] = segmentation
sys.modules["segmentation.schema"] = schema

from mymodel import method  # noqa: E402
from mymodel.typing import assigned_counts, type_probabilities  # noqa: E402


def _transcripts(coordinates: np.ndarray) -> TranscriptTable:
    count = len(coordinates)
    return TranscriptTable(
        tuple(f"tx-{index}" for index in range(count)),
        ("gene,quoted", "gene-b"),
        np.arange(count, dtype=np.int64) % 2,
        np.asarray(coordinates, dtype=np.float64),
    )


def _argument_values(command: tuple[str, ...], name: str, count: int = 1) -> list[str]:
    index = command.index(name)
    return list(command[index + 1 : index + 1 + count])


def test_float_argument_normalizes_signed_zero() -> None:
    assert method._float_argument(-0.0) == "0"


def test_image_initialization_builds_generic_command_and_exact_local_affine(
    tmp_path: Path,
) -> None:
    bounds = (1_000_000.0, 2_000_000.0, 1_000_020.0, 2_000_020.0)
    transcripts = _transcripts(
        np.asarray([[1_000_002.25, 2_000_003.5], [1_000_004.0, 2_000_005.0]])
    )
    image = np.zeros((9, 9), dtype=np.float32)
    image[2:7, 2:7] = 10
    channel = ImageChannel(
        "nuclear",
        "nuclear",
        image,
        (999_999.6, 1_999_999.8),
        (0.5, 0.25),
    )
    field = SimpleNamespace(
        nuclear_image=object(),
        load_nuclear_image=lambda: channel,
    )
    config = replace(method.ProsegConfig(), minimum_component_pixels=4, nthreads=7)
    local = method._local_float32_coordinates(transcripts.coordinates, bounds)

    invocation = method._prepare_invocation(
        field, transcripts, local, bounds, config, tmp_path, Path("/opt/proseg")
    )

    command = invocation.command
    assert command[0] == "/opt/proseg"
    assert not any(flag in command for flag in ("--xenium", "--cosmx", "--merfish"))
    assert "--ignore-z-coord" in command
    assert "--enforce-connectivity" in command
    assert _argument_values(command, "--voxel-layers") == ["1"]
    assert _argument_values(command, "--nthreads") == ["7"]
    assert _argument_values(command, "--cellpose-x-transform", 3) == ["0.5", "0", "0.1"]
    assert _argument_values(command, "--cellpose-y-transform", 3) == ["0", "0.25", "0.05"]
    mask_path = Path(_argument_values(command, "--cellpose-masks")[0])
    mask = np.load(mask_path, allow_pickle=False)
    assert mask.dtype == np.uint32
    assert mask.shape == (8, 8)
    assert set(np.unique(mask)) == {0, 1}

    transcript_path = Path(command[-1])
    with gzip.open(transcript_path, "rt", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["gene"] == "gene,quoted"
    assert float(rows[0]["x"]) == pytest.approx(2.25)
    assert float(rows[0]["y"]) == pytest.approx(3.5)
    assert {row["cell_id"] for row in rows} == {"0"}


def test_density_initialization_is_deterministic_and_assigns_separate_peaks() -> None:
    coordinates = np.asarray(
        [
            [1.8, 2.0],
            [2.0, 2.1],
            [2.2, 1.9],
            [17.8, 18.0],
            [18.0, 18.1],
            [18.2, 17.9],
        ],
        dtype=np.float32,
    )
    config = replace(
        method.ProsegConfig(),
        transcript_bin_size_um=2.0,
        transcript_smoothing_um=1.0,
        transcript_density_quantile=0.0,
        initial_assignment_radius_um=4.0,
    )

    first = method._density_initial_cell_ids(coordinates, (0.0, 0.0, 20.0, 20.0), config)
    second = method._density_initial_cell_ids(coordinates, (0.0, 0.0, 20.0, 20.0), config)

    np.testing.assert_array_equal(first, second)
    assert first[0] == first[1] == first[2] != 0
    assert first[3] == first[4] == first[5] != 0
    assert first[0] != first[3]


def test_subprocess_timeout_is_finite_and_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, object] = {}

    def expire(*args, **kwargs):
        observed.update(kwargs)
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(method.subprocess, "run", expire)
    with pytest.raises(method.ProsegExecutionError, match="17 second timeout"):
        method._run_command(
            ("proseg", "input.csv.gz"),
            cwd=tmp_path,
            timeout_seconds=17,
            environment={"PROSEG_TEST": "1"},
            log_path=tmp_path / "proseg.log",
        )

    assert observed["timeout"] == 17
    assert observed["env"] == {"PROSEG_TEST": "1"}
    assert observed["check"] is False


def test_geojson_conversion_restores_origin_clips_simplifies_and_disjoins(
    tmp_path: Path,
) -> None:
    angles = np.linspace(0, 2 * np.pi, 1_300, endpoint=False)
    detailed_ring = np.column_stack((4 + 3 * np.cos(angles), 5 + 3 * np.sin(angles)))
    detailed_ring = np.vstack((detailed_ring, detailed_ring[0])).tolist()
    document = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"cell": 1},
                "geometry": {"type": "Polygon", "coordinates": [detailed_ring]},
            },
            {
                "type": "Feature",
                "properties": {"cell": 2},
                "geometry": {
                    "type": "MultiPolygon",
                    "coordinates": [
                        [[[-2, 3], [10, 3], [10, 9], [-2, 9], [-2, 3]]],
                        [[[9, 9], [9.2, 9], [9.2, 9.2], [9, 9.2], [9, 9]]],
                    ],
                },
            },
            {
                "type": "Feature",
                "properties": {"cell": 3},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[20, 20], [21, 20], [21, 21], [20, 20]]],
                },
            },
        ],
    }
    path = tmp_path / "cells.geojson.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(document, handle)

    cells = method._load_geojson_cells(path, (100.0, 200.0, 110.0, 210.0))

    assert len(cells) == 2
    polygons = [Polygon(cell.vertices) for cell in cells]
    assert all(polygon.is_valid and polygon.area > 0 for polygon in polygons)
    assert all(len(cell.vertices) <= 1_024 for cell in cells)
    assert all(100 <= x <= 110 and 200 <= y <= 210 for cell in cells for x, y in cell.vertices)
    assert polygons[0].intersection(polygons[1]).area == pytest.approx(0.0)
    assert not np.array_equal(cells[0].vertices[0], cells[0].vertices[-1])


def _reference(counts, labels, genes=("gene,quoted", "gene-b")):
    """Stand-in for the task's already-matched reference expression view."""
    matrix = sparse.csr_matrix(counts)

    def matched_expression(spatial_genes):
        shared = tuple(gene for gene in genes if gene in spatial_genes)
        return SimpleNamespace(
            counts=matrix[:, [genes.index(gene) for gene in shared]], gene_ids=shared
        )

    return SimpleNamespace(cell_type_labels=tuple(labels), matched_expression=matched_expression)


def test_typing_retains_sparse_uncertainty_and_normalizes_reference_cells_equally():
    reference = _reference([[90, 10], [9000, 1000], [1, 9]], ["A", "A", "B"])
    counts = sparse.csr_matrix([[0, 0], [1, 0], [20, 0], [0, 20], [10**8, 0]])
    result = type_probabilities(counts, ("gene,quoted", "gene-b"), reference)

    assert result[0] == {"A": 0.5, "B": 0.5}
    # One RNA has probability proportional to the smoothed mean proportions,
    # rather than to library depth or the number of reference cells of a type.
    assert result[1]["A"] == pytest.approx((23 / 30) / (23 / 30 + 3 / 10))
    assert 0.5 < result[1]["A"] < result[2]["A"] < 1
    assert result[3]["B"] > 0.95
    assert result[4]["A"] > 0.99
    for probability in result:
        assert sum(probability.values()) == pytest.approx(1, abs=1e-8)
        assert all(np.isfinite(value) and 0 <= value <= 1 for value in probability.values())


@pytest.mark.parametrize(
    "reference",
    [
        _reference([[1], [3]], ["A", "B"], ("unshared",)),
        _reference([[1, 0], [3, 0]], ["A", "B"]),
        _reference([[1, 3], [0, 0]], ["A", "B"]),
        _reference([[1, 3], [1, 3]], ["A", "B"]),
    ],
)
def test_uninformative_panel_does_not_assert_unknown_or_drop_a_visible_type(reference):
    result = type_probabilities(
        sparse.csr_matrix([[50, 0], [0, 50], [0, 0]]), ("gene,quoted", "gene-b"), reference
    )
    assert result == [{"A": 0.5, "B": 0.5}] * 3


def test_typing_aligns_gene_axes_and_ignores_unmatched_transcripts():
    reference = _reference([[9, 1], [1, 9]], ["A", "B"])
    expected = type_probabilities(
        sparse.csr_matrix([[7, 1], [0, 0]]), ("gene,quoted", "gene-b"), reference
    )
    reordered = type_probabilities(
        sparse.csr_matrix([[1, 1_000_000, 7], [0, 1_000_000, 0]]),
        ("gene-b", "unshared", "gene,quoted"),
        reference,
    )
    assert reordered == expected


def _write_assignments(path, rows):
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("transcript_id", "gene", "assignment", "background"))
        writer.writerows(rows)


@pytest.mark.parametrize("defect", ["duplicate", "missing", "wrong_gene", "out_of_range"])
def test_assignment_join_rejects_silent_identity_corruption(tmp_path, defect):
    transcripts = _transcripts(np.zeros((2, 2)))
    rows = [[0, "gene,quoted", 0, "false"], [1, "gene-b", 0, "false"]]
    if defect == "duplicate":
        rows.append(rows[0])
    elif defect == "missing":
        rows.pop()
    elif defect == "wrong_gene":
        rows[0][1] = "gene-b"
    else:
        rows[0][0] = 2
    path = tmp_path / "assignments.csv.gz"
    _write_assignments(path, rows)
    with pytest.raises(ValueError, match="Proseg transcript"):
        assigned_counts(path, transcripts, [])


def test_segment_field_types_proseg_assignments_without_changing_geometry(tmp_path, monkeypatch):
    transcripts = _transcripts(np.full((12, 2), 0.5))
    reference = _reference([[90, 10], [10, 90]], ["A", "B"])
    field = SimpleNamespace(
        field_handle="test-field",
        field_bounds=(0.0, 0.0, 10.0, 10.0),
        nuclear_image=None,
        load_transcripts=lambda: transcripts,
        load_reference=lambda: reference,
    )

    def run(command, **kwargs):
        # Unsorted IDs and an out-of-field polygon exercise identity preservation
        # through sorting and dropping. Cell zero is a real Proseg output cell.
        features = []
        for cell_id, x in [(7, 3), (1, 20), (0, 0), (9, 6)]:
            features.append(
                {
                    "type": "Feature",
                    "properties": {"cell": cell_id},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[x, 0], [x + 1, 0], [x + 1, 1], [x, 1], [x, 0]]],
                    },
                }
            )
        polygon_path = Path(_argument_values(command, "--output-cell-polygons")[0])
        with gzip.open(polygon_path, "wt", encoding="utf-8") as handle:
            json.dump({"type": "FeatureCollection", "features": features}, handle)
        path = Path(_argument_values(command, "--output-transcript-metadata")[0])
        # All coordinates are in cell 0, but Proseg owns the expression partition.
        rows = [
            [index, transcripts.gene_ids[index % 2], 0 if index % 2 == 0 else 7, "false"]
            for index in range(8)
        ]
        rows.extend(
            [
                [8, "gene,quoted", 7, "true"],  # Ambient RNA even though assigned.
                [9, "gene-b", "", "true"],
                [10, "gene,quoted", 1, "false"],  # Discarded polygon.
                [11, "gene-b", 0, "true"],
            ]
        )
        _write_assignments(path, list(reversed(rows)))

    monkeypatch.setattr(method, "_proseg_runtime", lambda: (Path("/opt/proseg"), {}))
    monkeypatch.setattr(method, "_local_scratch_root", lambda: str(tmp_path))
    monkeypatch.setattr(method, "_run_command", run)
    prediction = method.segment_field(field, method.ProsegConfig())

    assert [cell.instance_id for cell in prediction.cells] == [
        "proseg-cell-0",
        "proseg-cell-7",
        "proseg-cell-9",
    ]
    expected = type_probabilities(
        sparse.csr_matrix([[4, 0], [0, 4], [0, 0]]), transcripts.gene_ids, reference
    )
    assert [cell.type_probabilities for cell in prediction.cells] == expected
    assert expected[0]["A"] > 0.9
    assert expected[1]["B"] > 0.9
    assert expected[2] == {"A": 0.5, "B": 0.5}
    assert prediction.nuclei == ()

    field.load_reference = lambda: None
    untyped = method.segment_field(field, method.ProsegConfig())
    for typed_cell, plain_cell in zip(prediction.cells, untyped.cells, strict=True):
        assert typed_cell.instance_id == plain_cell.instance_id
        np.testing.assert_array_equal(typed_cell.vertices, plain_cell.vertices)
        assert plain_cell.type_probabilities is None
