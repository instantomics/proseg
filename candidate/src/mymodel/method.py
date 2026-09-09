from __future__ import annotations

import csv
import gzip
import json
import logging
import math
import os
import resource
import subprocess
import tempfile
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from segmentation.geometry import assign_points
from segmentation.schema import (
    MAX_POLYGON_VERTICES,
    PolygonInstance,
    ReferenceExpression,
    SegmentationPrediction,
    TranscriptTable,
)
from shapely import STRtree, make_valid
from shapely.affinity import translate
from shapely.errors import GEOSException
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box, shape

LOGGER = logging.getLogger(__name__)

_TASK_PHASE_SECONDS = 600.0
_MAX_INITIAL_CELLS = 250_000
_MAX_DENSITY_BINS = 1_048_576
_MAX_DENSITY_TRANSCRIPTS = 2_000_000

# The image policy is intentionally fixed across platforms and assays.
_IMAGE_BACKGROUND_QUANTILE = 0.5
_IMAGE_HIGH_QUANTILE = 0.995
_IMAGE_THRESHOLD_FRACTION = 0.35
_IMAGE_STRUCTURE = np.ones((3, 3), dtype=bool)


class ProsegExecutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProsegConfig:
    nthreads: int = 4
    timeout_seconds: float = 540.0
    burnin_samples: int = 10
    samples: int = 10
    recorded_samples: int = 5
    hillclimb: int = 5
    morphology_steps_per_iter: int = 250
    ncomponents: int = 10
    cell_compactness: float = 0.04
    burnin_voxel_size_um: float = 2.0
    voxel_size_um: float = 1.0
    minimum_component_pixels: int = 12
    maximum_initial_cells: int = 4_096
    maximum_density_bins: int = 262_144
    maximum_density_transcripts: int = 1_500_000
    transcript_bin_size_um: float = 4.0
    transcript_density_quantile: float = 0.75
    transcript_smoothing_um: float = 6.0
    initial_assignment_radius_um: float = 12.0


@dataclass(frozen=True)
class ProsegInvocation:
    command: tuple[str, ...]
    polygon_path: Path
    transcript_metadata_path: Path
    log_path: Path


def load_config(path: str | Path) -> ProsegConfig:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise TypeError("candidate parameterization must be a JSON object")
    reference = document.get("reference")
    if not isinstance(reference, Mapping):
        raise TypeError("candidate parameterization must contain reference metadata")
    parameters = reference.get("parameters")
    if not isinstance(parameters, Mapping):
        raise TypeError("reference parameters must be a JSON object")

    allowed = set(ProsegConfig.__dataclass_fields__)
    unknown = set(parameters).difference(allowed)
    if unknown:
        raise ValueError(f"unknown Proseg parameters: {sorted(unknown)!r}")
    config = ProsegConfig(**dict(parameters))
    _validate_config(config)
    return config


def _validate_config(config: ProsegConfig) -> None:
    bounded_integers = {
        "nthreads": (1, 256),
        "burnin_samples": (1, 10_000),
        "samples": (1, 10_000),
        "recorded_samples": (1, 10_000),
        "hillclimb": (0, 10_000),
        "morphology_steps_per_iter": (1, 1_000_000),
        "ncomponents": (1, 1_024),
        "minimum_component_pixels": (1, 10_000_000),
        "maximum_initial_cells": (1, _MAX_INITIAL_CELLS),
        "maximum_density_bins": (1, _MAX_DENSITY_BINS),
        "maximum_density_transcripts": (1, _MAX_DENSITY_TRANSCRIPTS),
    }
    for name, (minimum, maximum) in bounded_integers.items():
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if not minimum <= value <= maximum:
            raise ValueError(f"{name} must be in [{minimum}, {maximum}]")

    positive_floats = (
        "timeout_seconds",
        "cell_compactness",
        "burnin_voxel_size_um",
        "voxel_size_um",
        "transcript_bin_size_um",
        "transcript_smoothing_um",
        "initial_assignment_radius_um",
    )
    for name in positive_floats:
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number")
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if config.timeout_seconds >= _TASK_PHASE_SECONDS:
        raise ValueError("timeout_seconds must leave time for task broker handling")
    if config.recorded_samples > config.samples:
        raise ValueError("recorded_samples must not exceed samples")
    if config.cell_compactness > 10:
        raise ValueError("cell_compactness must not exceed 10")
    quantile = config.transcript_density_quantile
    if (
        isinstance(quantile, bool)
        or not isinstance(quantile, (int, float))
        or not math.isfinite(quantile)
        or not 0 <= quantile < 1
    ):
        raise ValueError("transcript_density_quantile must be finite and in [0, 1)")
    ratio = config.burnin_voxel_size_um / config.voxel_size_um
    rounded_ratio = round(ratio)
    if (
        not math.isclose(ratio, rounded_ratio, rel_tol=0.0, abs_tol=1e-7)
        or rounded_ratio < 1
        or rounded_ratio & (rounded_ratio - 1)
    ):
        raise ValueError("burnin_voxel_size_um / voxel_size_um must be a positive power of two")


def posterior_method(inputs, outputs) -> None:
    fields = inputs.data.list_fields()
    try:
        config = load_config(inputs.parameterization_path)
    except (OSError, UnicodeError, TypeError, ValueError) as error:
        for field in fields:
            outputs.fail(field, error)
        return

    for field in fields:
        try:
            prediction = segment_field(field, config)
        # Candidate-side failures are arbitrary; report one result and continue.
        except Exception as error:
            LOGGER.exception(
                "proseg_field_failed %s",
                json.dumps(
                    {
                        "field_handle": field.field_handle,
                        "thread_count": config.nthreads,
                    },
                    sort_keys=True,
                ),
            )
            outputs.fail(field, error)
        else:
            outputs.submit(field, prediction)


def segment_field(field, config: ProsegConfig) -> SegmentationPrediction:
    transcripts = field.load_transcripts()
    if not len(transcripts.transcript_ids):
        raise ValueError("Proseg requires at least one transcript")
    bounds = _validated_bounds(field.field_bounds)
    local_coordinates = _local_float32_coordinates(transcripts.coordinates, bounds)
    scratch = _local_scratch_root()

    with tempfile.TemporaryDirectory(prefix="iomix-proseg-", dir=scratch) as temporary:
        workdir = Path(temporary)
        executable_path, environment = _proseg_runtime()
        invocation = _prepare_invocation(
            field,
            transcripts,
            local_coordinates,
            bounds,
            config,
            workdir,
            executable_path,
        )
        child_rss_before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        started = time.monotonic()
        _run_command(
            invocation.command,
            cwd=workdir,
            timeout_seconds=config.timeout_seconds,
            environment=environment,
            log_path=invocation.log_path,
        )
        inference_seconds = time.monotonic() - started
        child_rss_after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        cells = _load_geojson_cells(invocation.polygon_path, bounds)
        assignment_count = _count_assignments(invocation.transcript_metadata_path)

    if field.information_condition == "labeled_reference":
        reference = field.load_reference()
        if reference is None or reference.cell_type_labels is None:
            raise ValueError("labeled_reference field did not expose labeled reference data")
        cells = _label_cells(transcripts, reference, cells)

    count = len(transcripts.transcript_ids)
    log_record: dict[str, Any] = {
        "assignment_count": assignment_count,
        "cell_count": len(cells),
        "field_handle": field.field_handle,
        "inference_seconds": inference_seconds,
        "peak_child_rss_bytes": int(max(child_rss_before, child_rss_after)) * 1024,
        "thread_count": config.nthreads,
        "transcript_count": count,
        "transcripts_per_second": count / max(inference_seconds, 1e-9),
    }
    LOGGER.info("proseg_field %s", json.dumps(log_record, sort_keys=True))
    return SegmentationPrediction(tuple(cells), nuclei=())


def _proseg_runtime() -> tuple[Path, dict[str, str]]:
    from proseg_bin import executable, subprocess_environment

    return Path(executable()), dict(subprocess_environment())


def _local_scratch_root() -> str | None:
    scratch = os.environ.get("SCRATCHDIR")
    return scratch if scratch and Path(scratch).is_dir() else None


def _validated_bounds(values: Sequence[float]) -> tuple[float, float, float, float]:
    bounds = tuple(float(value) for value in values)
    if len(bounds) != 4 or not np.isfinite(bounds).all():
        raise ValueError("field bounds must contain four finite coordinates")
    if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
        raise ValueError("field bounds must have positive width and height")
    return bounds


def _local_float32_coordinates(
    coordinates: np.ndarray, bounds: tuple[float, float, float, float]
) -> np.ndarray:
    values = np.asarray(coordinates, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError("transcript coordinates must be a finite [transcripts, 2] array")
    local = values - np.asarray(bounds[:2], dtype=np.float64)
    local = local.astype(np.float32)
    if not np.isfinite(local).all():
        raise ValueError("field-local transcript coordinates exceed float32 range")
    return local


def _prepare_invocation(
    field,
    transcripts: TranscriptTable,
    local_coordinates: np.ndarray,
    bounds: tuple[float, float, float, float],
    config: ProsegConfig,
    workdir: Path,
    executable_path: Path,
) -> ProsegInvocation:
    transcript_path = workdir / "transcripts.csv.gz"
    polygon_path = workdir / "cell-polygons.geojson.gz"
    metadata_path = workdir / "transcript-metadata.csv.gz"
    log_path = workdir / "proseg.log"
    initialization_arguments: list[str]

    if field.nuclear_image is not None:
        channel = field.load_nuclear_image()
        mask = _component_mask(
            channel.image,
            minimum_component_pixels=config.minimum_component_pixels,
            maximum_components=config.maximum_initial_cells,
        )
        if not np.any(mask):
            raise ValueError("fixed nuclear-image policy found no initialization components")
        mask_path = workdir / "nuclear-components.npy"
        np.save(mask_path, mask, allow_pickle=False)
        initial_cell_ids = np.zeros(len(local_coordinates), dtype=np.uint32)
        x_offset = float(channel.origin_um[0]) - bounds[0]
        y_offset = float(channel.origin_um[1]) - bounds[1]
        initialization_arguments = [
            "--cellpose-masks",
            str(mask_path),
            "--cellpose-x-transform",
            _float_argument(channel.pixel_size_um[0]),
            "0",
            _float_argument(x_offset),
            "--cellpose-y-transform",
            "0",
            _float_argument(channel.pixel_size_um[1]),
            _float_argument(y_offset),
        ]
    else:
        local_bounds = (0.0, 0.0, bounds[2] - bounds[0], bounds[3] - bounds[1])
        initial_cell_ids = _density_initial_cell_ids(local_coordinates, local_bounds, config)
        if not np.any(initial_cell_ids):
            raise ValueError("transcript-density policy found no initialization cells")
        initialization_arguments = ["--use-cell-initialization"]

    _write_transcript_csv(transcript_path, transcripts, local_coordinates, initial_cell_ids)
    command = _build_command(
        executable_path,
        transcript_path,
        polygon_path,
        metadata_path,
        workdir / "proseg-output.zarr",
        config,
        initialization_arguments,
    )
    return ProsegInvocation(command, polygon_path, metadata_path, log_path)


def _component_mask(
    image: np.ndarray, *, minimum_component_pixels: int, maximum_components: int
) -> np.ndarray:
    values = np.asarray(image, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("nuclear image must be a finite two-dimensional array")
    background = float(np.quantile(values, _IMAGE_BACKGROUND_QUANTILE))
    high = float(np.quantile(values, _IMAGE_HIGH_QUANTILE))
    if high <= background:
        return np.zeros(values.shape, dtype=np.uint32)
    threshold = background + _IMAGE_THRESHOLD_FRACTION * (high - background)
    foreground = values >= threshold
    foreground = ndimage.binary_opening(foreground, structure=_IMAGE_STRUCTURE)
    foreground = ndimage.binary_closing(foreground, structure=_IMAGE_STRUCTURE)
    foreground = ndimage.binary_fill_holes(foreground)
    labels, count = ndimage.label(foreground, structure=_IMAGE_STRUCTURE)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    selected = [
        label for label in range(1, count + 1) if int(sizes[label]) >= minimum_component_pixels
    ]
    selected = sorted(
        sorted(selected, key=lambda label: (-int(sizes[label]), label))[:maximum_components]
    )
    result = np.zeros(values.shape, dtype=np.uint32)
    for new_label, old_label in enumerate(selected, start=1):
        result[labels == old_label] = new_label
    return result


def _density_initial_cell_ids(
    coordinates: np.ndarray,
    bounds: tuple[float, float, float, float],
    config: ProsegConfig,
) -> np.ndarray:
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if not len(coordinates):
        return np.zeros(0, dtype=np.uint32)
    sampled = _evenly_spaced_sample(coordinates, config.maximum_density_transcripts)
    x_bins, y_bins = _density_grid_shape(
        bounds,
        bin_size_um=config.transcript_bin_size_um,
        maximum_bins=config.maximum_density_bins,
    )
    histogram, _, _ = np.histogram2d(
        sampled[:, 1],
        sampled[:, 0],
        bins=(y_bins, x_bins),
        range=((bounds[1], bounds[3]), (bounds[0], bounds[2])),
    )
    x_bin_size = (bounds[2] - bounds[0]) / x_bins
    y_bin_size = (bounds[3] - bounds[1]) / y_bins
    sigma = (
        min(8.0, config.transcript_smoothing_um / y_bin_size),
        min(8.0, config.transcript_smoothing_um / x_bin_size),
    )
    density = ndimage.gaussian_filter(histogram, sigma=sigma, mode="constant", truncate=2.0)
    positive = density[density > 0]
    if not positive.size:
        return np.zeros(len(coordinates), dtype=np.uint32)
    threshold = float(np.quantile(positive, config.transcript_density_quantile))
    neighborhood = tuple(2 * max(1, math.ceil(value)) + 1 for value in sigma)
    maxima = (density >= threshold) & (
        density == ndimage.maximum_filter(density, size=neighborhood, mode="constant")
    )
    plateau_labels, _ = ndimage.label(maxima, structure=_IMAGE_STRUCTURE)
    representatives: list[int] = []
    seen: set[int] = set()
    for flat_index in np.flatnonzero(maxima):
        plateau = int(plateau_labels.ravel()[flat_index])
        if plateau not in seen:
            seen.add(plateau)
            representatives.append(int(flat_index))
    representatives.sort(key=lambda index: (-float(density.ravel()[index]), index))
    representatives = representatives[: config.maximum_initial_cells]
    if not representatives:
        return np.zeros(len(coordinates), dtype=np.uint32)
    rows, columns = np.unravel_index(representatives, density.shape)
    centers = np.column_stack(
        (
            bounds[0] + (columns + 0.5) * x_bin_size,
            bounds[1] + (rows + 0.5) * y_bin_size,
        )
    )
    distances, nearest = cKDTree(centers).query(coordinates, k=1, workers=1)
    assignments = np.zeros(len(coordinates), dtype=np.uint32)
    assigned = np.asarray(distances) <= config.initial_assignment_radius_um
    assignments[assigned] = np.asarray(nearest[assigned], dtype=np.uint32) + 1
    return assignments


def _density_grid_shape(
    bounds: tuple[float, float, float, float],
    *,
    bin_size_um: float,
    maximum_bins: int,
) -> tuple[int, int]:
    x_bins = max(1, math.ceil((bounds[2] - bounds[0]) / bin_size_um))
    y_bins = max(1, math.ceil((bounds[3] - bounds[1]) / bin_size_um))
    if x_bins * y_bins <= maximum_bins:
        return x_bins, y_bins
    scale = math.sqrt(maximum_bins / (x_bins * y_bins))
    x_bins = max(1, math.floor(x_bins * scale))
    y_bins = max(1, math.floor(y_bins * scale))
    while x_bins * y_bins > maximum_bins:
        if x_bins >= y_bins:
            x_bins -= 1
        else:
            y_bins -= 1
    return x_bins, y_bins


def _evenly_spaced_sample(values: np.ndarray, maximum: int) -> np.ndarray:
    if len(values) <= maximum:
        return values
    indices = np.linspace(0, len(values) - 1, maximum, dtype=np.int64)
    return values[indices]


def _write_transcript_csv(
    path: Path,
    transcripts: TranscriptTable,
    coordinates: np.ndarray,
    cell_ids: np.ndarray,
) -> None:
    gene_index = np.asarray(transcripts.gene_index, dtype=np.int64)
    if len(coordinates) != len(gene_index) or len(cell_ids) != len(gene_index):
        raise ValueError("transcript CSV columns are not aligned")
    genes = tuple(transcripts.gene_ids)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("transcript_id", "gene", "x", "y", "z", "cell_id"))
        for index, ((x, y), gene, cell_id) in enumerate(
            zip(coordinates, gene_index, cell_ids, strict=True)
        ):
            if gene < 0 or gene >= len(genes):
                raise ValueError("transcript gene index is out of range")
            writer.writerow(
                (
                    index,
                    genes[int(gene)],
                    _float_argument(x),
                    _float_argument(y),
                    "0",
                    int(cell_id),
                )
            )


def _build_command(
    executable_path: Path,
    transcript_path: Path,
    polygon_path: Path,
    metadata_path: Path,
    spatialdata_path: Path,
    config: ProsegConfig,
    initialization_arguments: Sequence[str],
) -> tuple[str, ...]:
    return (
        str(executable_path),
        "--gene-column",
        "gene",
        "--transcript-id-column",
        "transcript_id",
        "--x-column",
        "x",
        "--y-column",
        "y",
        "--z-column",
        "z",
        "--cell-id-column",
        "cell_id",
        "--cell-id-unassigned",
        "0",
        "--ignore-z-coord",
        "--voxel-layers",
        "1",
        "--enforce-connectivity",
        "--nthreads",
        str(config.nthreads),
        "--burnin-samples",
        str(config.burnin_samples),
        "--samples",
        str(config.samples),
        "--recorded-samples",
        str(config.recorded_samples),
        "--hillclimb",
        str(config.hillclimb),
        "--morphology-steps-per-iter",
        str(config.morphology_steps_per_iter),
        "--ncomponents",
        str(config.ncomponents),
        "--cell-compactness",
        _float_argument(config.cell_compactness),
        "--burnin-voxel-size",
        _float_argument(config.burnin_voxel_size_um),
        "--voxel-size",
        _float_argument(config.voxel_size_um),
        "--output-spatialdata",
        str(spatialdata_path),
        "--exclude-spatialdata-transcripts",
        "--output-cell-polygons",
        str(polygon_path),
        "--output-transcript-metadata",
        str(metadata_path),
        "--output-transcript-metadata-fmt",
        "csv-gz",
        *tuple(initialization_arguments),
        str(transcript_path),
    )


def _float_argument(value: float) -> str:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Proseg numeric arguments must be finite")
    return format(value, ".9g")


def _run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    timeout_seconds: float,
    environment: Mapping[str, str],
    log_path: Path,
) -> None:
    with log_path.open("wb") as log:
        try:
            completed = subprocess.run(
                list(command),
                cwd=cwd,
                env=dict(environment),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            raise ProsegExecutionError(
                f"Proseg exceeded its {timeout_seconds:g} second timeout"
            ) from error
    if completed.returncode:
        detail = _log_excerpt(log_path)
        suffix = f": {detail}" if detail else ""
        raise ProsegExecutionError(f"Proseg exited with status {completed.returncode}{suffix}")


def _log_excerpt(path: Path, maximum_bytes: int = 4_096) -> str:
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - maximum_bytes))
        return " ".join(handle.read().decode("utf-8", errors="replace").split())


def _load_geojson_cells(
    path: Path, bounds: tuple[float, float, float, float]
) -> list[PolygonInstance]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, Mapping) or document.get("type") != "FeatureCollection":
        raise ValueError("Proseg polygon output is not a GeoJSON FeatureCollection")
    features = document.get("features")
    if not isinstance(features, list):
        raise TypeError("Proseg GeoJSON features must be an array")

    ordered: list[tuple[tuple[int, Any, int], Mapping[str, Any]]] = []
    for index, feature in enumerate(features):
        if not isinstance(feature, Mapping):
            continue
        properties = feature.get("properties")
        cell = properties.get("cell") if isinstance(properties, Mapping) else index
        if isinstance(cell, int) and not isinstance(cell, bool):
            key = (0, cell, index)
        elif isinstance(cell, str) and cell.isdecimal():
            key = (0, int(cell), index)
        else:
            key = (1, str(cell), index)
        ordered.append((key, feature))
    ordered.sort(key=lambda item: item[0])

    field_box = box(*bounds)
    origin_x, origin_y = bounds[:2]
    prepared: list[Polygon] = []
    for _key, feature in ordered:
        geometry_value = feature.get("geometry")
        if not isinstance(geometry_value, Mapping):
            continue
        try:
            geometry = shape(geometry_value)
            geometry = _valid_geometry(translate(geometry, xoff=origin_x, yoff=origin_y))
            polygon = _largest_polygon(_valid_geometry(geometry.intersection(field_box)))
        except (TypeError, ValueError, OverflowError, GEOSException):
            continue
        if polygon is None:
            continue
        # The task polygon schema cannot represent holes.
        polygon = _largest_polygon(_valid_geometry(Polygon(polygon.exterior)))
        polygon = _simplify_to_limit(polygon, MAX_POLYGON_VERTICES)
        if polygon is None:
            continue
        prepared.append(polygon)

    tree = STRtree(prepared)
    resolved: list[Polygon | None] = [None] * len(prepared)
    result: list[PolygonInstance] = []
    for index, polygon in enumerate(prepared):
        previous = _intersecting_previous(polygon, tree, resolved, index)
        polygon = _subtract_accepted(polygon, previous)
        if polygon is None:
            continue
        if _vertex_count(polygon) > MAX_POLYGON_VERTICES:
            polygon = _simplify_to_limit(polygon, MAX_POLYGON_VERTICES)
            if polygon is None:
                continue
            previous = _intersecting_previous(polygon, tree, resolved, index)
            polygon = _subtract_accepted(polygon, previous)
        if (
            polygon is None
            or _vertex_count(polygon) > MAX_POLYGON_VERTICES
            or not polygon.is_valid
            or polygon.area <= 0
        ):
            continue
        vertices = np.asarray(polygon.exterior.coords[:-1], dtype=np.float64)
        if len(vertices) < 3 or not np.isfinite(vertices).all():
            continue
        resolved[index] = polygon
        result.append(PolygonInstance(f"proseg-cell-{len(result)}", vertices))
    return result


def _intersecting_previous(
    polygon: Polygon,
    tree: STRtree,
    resolved: Sequence[Polygon | None],
    index: int,
) -> list[Polygon]:
    indices = sorted(
        int(candidate)
        for candidate in tree.query(polygon, predicate="intersects")
        if int(candidate) < index and resolved[int(candidate)] is not None
    )
    previous = []
    for candidate in indices:
        accepted = resolved[candidate]
        if accepted is not None:
            previous.append(accepted)
    return previous


def _valid_geometry(geometry):
    if geometry.is_empty:
        return geometry
    return geometry if geometry.is_valid else make_valid(geometry)


def _largest_polygon(geometry) -> Polygon | None:
    if isinstance(geometry, Polygon):
        return geometry if not geometry.is_empty and geometry.area > 0 else None
    if isinstance(geometry, (MultiPolygon, GeometryCollection)):
        polygons: list[Polygon] = []
        for item in geometry.geoms:
            polygon = _largest_polygon(item)
            if polygon is not None:
                polygons.append(polygon)
        if polygons:
            return max(enumerate(polygons), key=lambda item: (item[1].area, -item[0]))[1]
    return None


def _subtract_accepted(polygon: Polygon, accepted: Sequence[Polygon]) -> Polygon | None:
    geometry = polygon
    for previous in accepted:
        if geometry.intersects(previous):
            geometry = geometry.difference(previous)
            geometry = _valid_geometry(geometry)
            if geometry.is_empty:
                return None
    polygon = _largest_polygon(geometry)
    if polygon is None:
        return None
    # Exterior-only conversion is safe here because earlier cells have already been removed.
    exterior = Polygon(polygon.exterior)
    for previous in accepted:
        if exterior.intersection(previous).area > 1e-12:
            exterior = _largest_polygon(_valid_geometry(exterior.difference(previous)))
            if exterior is None:
                return None
    return exterior


def _vertex_count(polygon: Polygon) -> int:
    return max(0, len(polygon.exterior.coords) - 1)


def _simplify_to_limit(polygon: Polygon, maximum_vertices: int) -> Polygon | None:
    if _vertex_count(polygon) <= maximum_vertices:
        return polygon
    min_x, min_y, max_x, max_y = polygon.bounds
    high = max(max_x - min_x, max_y - min_y) / 1_000_000.0
    candidate: Polygon | None = None
    for _ in range(48):
        simplified = _largest_polygon(
            _valid_geometry(polygon.simplify(high, preserve_topology=True))
        )
        if simplified is not None and _vertex_count(simplified) <= maximum_vertices:
            candidate = simplified
            break
        high *= 2.0
    if candidate is None:
        return None
    low = 0.0
    for _ in range(32):
        middle = (low + high) / 2.0
        simplified = _largest_polygon(
            _valid_geometry(polygon.simplify(middle, preserve_topology=True))
        )
        if simplified is not None and _vertex_count(simplified) <= maximum_vertices:
            candidate = simplified
            high = middle
        else:
            low = middle
    return candidate


def _count_assignments(path: Path) -> int | None:
    try:
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "assignment" not in reader.fieldnames:
                return None
            return sum(
                bool((row.get("assignment") or "").strip())
                and (row.get("background") or "false").strip().lower() not in {"true", "1"}
                for row in reader
            )
    except (OSError, UnicodeError, csv.Error):
        return None


def _label_cells(
    transcripts: TranscriptTable,
    reference: ReferenceExpression,
    cells: list[PolygonInstance],
) -> list[PolygonInstance]:
    labels = reference.cell_type_labels
    if labels is None or not labels:
        raise ValueError("labeled reference must contain cell type labels")
    matched = reference.matched_expression(transcripts.gene_ids)
    labels_array = np.asarray(labels)
    label_order = tuple(sorted(set(labels)))
    targets = np.vstack(
        [
            np.asarray(matched.counts[labels_array == label].mean(axis=0)).ravel()
            for label in label_order
        ]
    )
    targets = _normalize_profiles(targets)
    spatial_lookup = {gene: index for index, gene in enumerate(transcripts.gene_ids)}
    spatial_columns = np.asarray(
        [spatial_lookup[gene] for gene in matched.gene_ids], dtype=np.int64
    )
    spatial_to_matched = np.full(len(transcripts.gene_ids), -1, dtype=np.int64)
    spatial_to_matched[spatial_columns] = np.arange(len(spatial_columns))
    assignments = assign_points(transcripts.coordinates, tuple(cells))
    gene_index = np.asarray(transcripts.gene_index, dtype=np.int64)
    frequencies = Counter(labels)
    fallback = min(label_order, key=lambda label: (-frequencies[label], label))

    labeled: list[PolygonInstance] = []
    for cell in cells:
        indices = np.flatnonzero(assignments == cell.instance_id)
        mapped = spatial_to_matched[gene_index[indices]]
        mapped = mapped[mapped >= 0]
        if mapped.size and matched.gene_ids:
            profile = np.bincount(mapped, minlength=len(matched.gene_ids))[None, :]
            similarities = targets @ _normalize_profiles(profile)[0]
            label = label_order[int(np.argmax(similarities))]
        else:
            label = fallback
        labeled.append(replace(cell, cell_type_label=label))
    return labeled


def _normalize_profiles(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals > 0)
    normalized = np.log1p(normalized * 10_000.0)
    norms = np.linalg.norm(normalized, axis=1, keepdims=True)
    return np.divide(normalized, norms, out=np.zeros_like(normalized), where=norms > 0)


__all__ = ["load_config", "posterior_method", "segment_field"]
