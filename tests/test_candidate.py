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
from shapely.geometry import Point, Polygon

CANDIDATE_SOURCE = Path(__file__).parents[1] / "candidate" / "src"
sys.path.insert(0, str(CANDIDATE_SOURCE))


@dataclass(frozen=True)
class PolygonInstance:
    instance_id: str
    vertices: np.ndarray
    parent_cell_id: str | None = None
    cell_type_label: str | None = None


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


@dataclass(frozen=True)
class ReferenceExpression:
    counts: sparse.csr_matrix
    cell_ids: tuple[str, ...]
    gene_ids: tuple[str, ...]
    cell_type_labels: tuple[str, ...] | None = None

    def matched_expression(self, gene_ids):
        indices = [self.gene_ids.index(gene) for gene in gene_ids if gene in self.gene_ids]
        return SimpleNamespace(
            counts=self.counts[:, indices],
            gene_ids=tuple(self.gene_ids[index] for index in indices),
        )


def _assign_points(coordinates, polygons):
    result = np.full(len(coordinates), None, dtype=object)
    for index, coordinate in enumerate(coordinates):
        point = Point(coordinate)
        for polygon in polygons:
            if Polygon(polygon.vertices).covers(point):
                result[index] = polygon.instance_id
                break
    return result


segmentation = ModuleType("segmentation")
segmentation.__path__ = []
geometry = ModuleType("segmentation.geometry")
geometry.assign_points = _assign_points
schema = ModuleType("segmentation.schema")
schema.MAX_POLYGON_VERTICES = 1_024
for _name, _value in {
    "ImageChannel": ImageChannel,
    "PolygonInstance": PolygonInstance,
    "ReferenceExpression": ReferenceExpression,
    "SegmentationPrediction": SegmentationPrediction,
    "TranscriptTable": TranscriptTable,
}.items():
    setattr(schema, _name, _value)
segmentation.geometry = geometry
segmentation.schema = schema
sys.modules["segmentation"] = segmentation
sys.modules["segmentation.geometry"] = geometry
sys.modules["segmentation.schema"] = schema

from mymodel import method  # noqa: E402


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
        (1_000_001.0, 2_000_002.0),
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
    assert _argument_values(command, "--cellpose-x-transform", 3) == ["0.5", "0", "1"]
    assert _argument_values(command, "--cellpose-y-transform", 3) == ["0", "0.25", "2"]
    mask_path = Path(_argument_values(command, "--cellpose-masks")[0])
    mask = np.load(mask_path, allow_pickle=False)
    assert mask.dtype == np.uint32
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


def test_labeled_reference_transfer_uses_only_legal_reference_labels() -> None:
    transcripts = TranscriptTable(
        ("t0", "t1", "t2", "t3"),
        ("gene-a", "gene-b"),
        np.asarray([0, 0, 1, 1], dtype=np.int64),
        np.asarray([[1, 1], [2, 2], [11, 1], [12, 2]], dtype=np.float64),
    )
    reference = ReferenceExpression(
        sparse.csr_matrix(np.asarray([[10, 0], [0, 10]], dtype=np.int64)),
        ("reference-a", "reference-b"),
        ("gene-a", "gene-b"),
        ("type-a", "type-b"),
    )
    cells = [
        PolygonInstance("cell-a", np.asarray([[0, 0], [5, 0], [5, 5], [0, 5]], dtype=float)),
        PolygonInstance("cell-b", np.asarray([[10, 0], [15, 0], [15, 5], [10, 5]], dtype=float)),
    ]

    labeled = method._label_cells(transcripts, reference, cells)

    assert [cell.cell_type_label for cell in labeled] == ["type-a", "type-b"]
