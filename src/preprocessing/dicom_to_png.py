"""
dicom_to_png.py
===============
Convert a folder of DICOM files into 2D axial PNG slices.

Pipeline
--------
1. Recursively scan the input directory for all DICOM files.
2. Group files by SeriesInstanceUID so each imaging series is handled
   independently (one patient may have multiple series).
3. Sort slices within each series by ImagePositionPatient z-coordinate
   (axial position) so the volume is assembled in anatomical order.
4. Extract the 2D pixel array from each slice.
5. Apply per-slice z-score normalisation: subtract the slice mean and
   divide by the slice standard deviation, then rescale to [0, 255].
6. Write PNG files to the output directory with the naming convention:
       <SeriesInstanceUID>_slice<NNNN>.png

Usage
-----
Command line:
    python dicom_to_png.py \\
        --input  path/to/dicom/folder \\
        --output path/to/data/processed/anomaly/images \\
        [--series-prefix]

From another module:
    from src.preprocessing.dicom_to_png import convert_series_to_png

Dependencies
------------
- pydicom  : DICOM file I/O
- numpy    : array math
- Pillow   : PNG writing
- pathlib  : path handling (stdlib)
- argparse : CLI (stdlib)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pydicom
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# DICOM discovery and grouping
# ─────────────────────────────────────────────────────────────────────────────


def find_dicom_files(directory: Path) -> List[Path]:
    """Recursively find all DICOM files under *directory*.

    A file is considered DICOM if it can be opened by pydicom without error.
    Files with the extension ``.dcm`` are tried first; other files are also
    checked so that DICOM data stored without extensions is not missed.

    Args:
        directory: Root folder to search.

    Returns:
        Sorted list of Path objects pointing to valid DICOM files.

    Raises:
        FileNotFoundError: If *directory* does not exist.
    """
    if not directory.exists():
        raise FileNotFoundError(f"Input directory not found: {directory}")

    candidates: List[Path] = []

    # Prioritise *.dcm files, then check everything else
    dcm_files = sorted(directory.rglob("*.dcm"))
    other_files = sorted(
        p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() != ".dcm"
    )
    all_candidates = dcm_files + other_files

    for path in all_candidates:
        try:
            pydicom.dcmread(str(path), stop_before_pixels=True)
            candidates.append(path)
        except Exception:  # noqa: BLE001  (broad — intentional: skip non-DICOM)
            continue

    logger.info("Found %d DICOM files in %s", len(candidates), directory)
    return candidates


def group_by_series(
    dicom_paths: List[Path],
) -> Dict[str, List[Tuple[float, Path]]]:
    """Group DICOM file paths by SeriesInstanceUID.

    Within each group the slices are stored as (z_position, path) tuples so
    they can later be sorted into anatomical order along the axial axis.

    Args:
        dicom_paths: List of paths to valid DICOM files.

    Returns:
        Dictionary mapping SeriesInstanceUID → list of (z_position, path).
        If a file lacks ImagePositionPatient its z_position defaults to 0.0.
    """
    series: Dict[str, List[Tuple[float, Path]]] = {}

    for path in dicom_paths:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True)

        series_uid: str = getattr(ds, "SeriesInstanceUID", "UNKNOWN_SERIES")

        # z-coordinate from ImagePositionPatient (third element = axial position)
        try:
            z_pos = float(ds.ImagePositionPatient[2])
        except (AttributeError, IndexError, TypeError):
            z_pos = 0.0

        series.setdefault(series_uid, []).append((z_pos, path))

    # Sort each series by z-position (inferior → superior)
    for uid in series:
        series[uid].sort(key=lambda t: t[0])

    logger.info("Grouped into %d series", len(series))
    return series


# ─────────────────────────────────────────────────────────────────────────────
# Pixel extraction and normalisation
# ─────────────────────────────────────────────────────────────────────────────


def extract_pixel_array(ds: pydicom.Dataset) -> np.ndarray:
    """Extract the raw 2D pixel array from a DICOM dataset.

    Applies RescaleSlope / RescaleIntercept if present so the values
    represent Hounsfield Units (CT) or scanner-specific intensity units (MRI).

    Args:
        ds: Fully loaded pydicom Dataset (pixel data must be present).

    Returns:
        2-D float32 NumPy array of shape (rows, cols).

    Raises:
        AttributeError: If the dataset has no PixelData.
    """
    arr: np.ndarray = ds.pixel_array.astype(np.float32)

    slope: float = float(getattr(ds, "RescaleSlope", 1.0))
    intercept: float = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept

    return arr


def zscore_normalise(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Apply per-slice z-score normalisation and rescale to uint8 [0, 255].

    Z-score formula:   z = (x - μ) / (σ + ε)

    After normalisation the values are clipped to ±3σ and linearly rescaled
    to [0, 255] so that 99.7 % of the intensity range is preserved without
    saturation artefacts.

    Args:
        arr: 2-D float32 array (one MRI/CT slice).
        eps: Small constant added to the standard deviation to avoid
             division by zero for uniform slices.

    Returns:
        2-D uint8 NumPy array suitable for PNG export.
    """
    mean: float = float(np.mean(arr))
    std: float = float(np.std(arr))

    z: np.ndarray = (arr - mean) / (std + eps)

    # Clip to ±3 σ to reduce the effect of outlier intensities
    z = np.clip(z, -3.0, 3.0)

    # Rescale from [-3, 3] → [0, 255]
    z_min, z_max = z.min(), z.max()
    if z_max - z_min < eps:
        # Completely uniform slice → grey image
        normalised = np.full_like(z, 128, dtype=np.uint8)
    else:
        normalised = ((z - z_min) / (z_max - z_min) * 255).astype(np.uint8)

    return normalised


# ─────────────────────────────────────────────────────────────────────────────
# Per-series conversion
# ─────────────────────────────────────────────────────────────────────────────


def convert_series_to_png(
    series_uid: str,
    slices: List[Tuple[float, Path]],
    output_dir: Path,
    series_prefix: bool = False,
) -> List[Path]:
    """Convert one DICOM series to a set of normalised axial PNG slices.

    Args:
        series_uid: The SeriesInstanceUID string used for output filenames.
        slices:     List of (z_position, dicom_path) tuples, sorted by z.
        output_dir: Destination directory for PNG files.
        series_prefix: If True, use the full SeriesInstanceUID in the filename;
                       if False, truncate to the last 12 characters for brevity.

    Returns:
        List of Path objects pointing to the PNG files that were written.

    Raises:
        RuntimeError: If a slice cannot be read or converted.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    uid_label: str = series_uid if series_prefix else series_uid[-12:]
    written: List[Path] = []

    for idx, (z_pos, dicom_path) in enumerate(slices):
        try:
            ds: pydicom.Dataset = pydicom.dcmread(str(dicom_path))
            arr: np.ndarray = extract_pixel_array(ds)
            normalised: np.ndarray = zscore_normalise(arr)

            filename: str = f"{uid_label}_slice{idx:04d}.png"
            out_path: Path = output_dir / filename

            Image.fromarray(normalised, mode="L").save(str(out_path))
            written.append(out_path)

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Skipping slice %d (z=%.2f) in series %s — %s",
                idx,
                z_pos,
                series_uid,
                exc,
            )
            continue

    logger.info(
        "Series %s — wrote %d / %d slices to %s",
        series_uid[-12:],
        len(written),
        len(slices),
        output_dir,
    )
    return written


# ─────────────────────────────────────────────────────────────────────────────
# Top-level batch conversion
# ─────────────────────────────────────────────────────────────────────────────


def convert_dicom_folder(
    input_dir: Path,
    output_dir: Path,
    series_prefix: bool = False,
) -> Dict[str, List[Path]]:
    """Convert an entire folder of DICOM files to PNG slices.

    This is the main entry-point when calling from another module.

    Args:
        input_dir:     Directory (scanned recursively) containing DICOM files.
        output_dir:    Root output directory. PNGs are written directly into
                       this folder (one flat namespace per run).
        series_prefix: Passed through to :func:`convert_series_to_png`.

    Returns:
        Dictionary mapping SeriesInstanceUID → list of written PNG Paths.

    Raises:
        FileNotFoundError: If *input_dir* does not exist.
        ValueError:        If no DICOM files are found.
    """
    dicom_paths: List[Path] = find_dicom_files(input_dir)

    if not dicom_paths:
        raise ValueError(f"No DICOM files found under {input_dir}")

    series_map: Dict[str, List[Tuple[float, Path]]] = group_by_series(dicom_paths)

    results: Dict[str, List[Path]] = {}
    for uid, slices in series_map.items():
        written = convert_series_to_png(
            series_uid=uid,
            slices=slices,
            output_dir=output_dir,
            series_prefix=series_prefix,
        )
        results[uid] = written

    total_slices = sum(len(v) for v in results.values())
    logger.info(
        "Conversion complete — %d series, %d PNG slices → %s",
        len(results),
        total_slices,
        output_dir,
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to sys.argv if None).

    Returns:
        Parsed Namespace with fields: input, output, series_prefix.
    """
    parser = argparse.ArgumentParser(
        prog="dicom_to_png",
        description=(
            "Convert a folder of DICOM files to z-score-normalised axial PNG slices. "
            "Slices are grouped by SeriesInstanceUID and sorted by anatomical z-position."
        ),
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        metavar="DIR",
        help="Path to the directory containing DICOM files (searched recursively).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("data/processed/anomaly/images"),
        metavar="DIR",
        help=(
            "Destination directory for PNG slices. "
            "Default: data/processed/anomaly/images (relative to CWD)."
        ),
    )
    parser.add_argument(
        "--series-prefix",
        action="store_true",
        default=False,
        help=(
            "Use the full SeriesInstanceUID in output filenames instead of "
            "the last 12 characters. Useful when series UIDs share a common suffix."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry-point for dicom_to_png.

    Args:
        argv: Argument list (defaults to sys.argv if None).

    Raises:
        SystemExit: With code 1 on fatal error.
    """
    args = _parse_args(argv)

    logger.info("Input directory : %s", args.input.resolve())
    logger.info("Output directory: %s", args.output.resolve())
    logger.info("Series prefix   : %s", args.series_prefix)

    try:
        convert_dicom_folder(
            input_dir=args.input,
            output_dir=args.output,
            series_prefix=args.series_prefix,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Fatal error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
