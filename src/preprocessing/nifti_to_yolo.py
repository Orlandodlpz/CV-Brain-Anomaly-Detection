"""
nifti_to_yolo.py
================
Convert BraTS-style NIfTI volumes + segmentation masks into a YOLOv11-seg
compatible 2D training dataset (axial PNG slices + polygon label files).

BraTS layout (one folder per case)
----------------------------------
    <case>-seg.nii.gz   ← integer segmentation map (label values 1-4)
    <case>-t1c.nii.gz   ← T1 contrast-enhanced (default modality for images)
    <case>-t1n.nii.gz
    <case>-t2f.nii.gz
    <case>-t2w.nii.gz

BraTS 2024 segmentation label semantics
---------------------------------------
    1 = NETC  (necrotic tumor core)
    2 = SNFH  (surrounding non-enhancing FLAIR hyperintensity)
    3 = ET    (enhancing tumor)
    4 = RC    (resection cavity, post-treatment cases only)

Pipeline
--------
1. Discover every <case>-seg.nii.gz file under --input.
2. For each case, load the segmentation volume and the chosen modality volume.
3. Iterate over axial slices along --axis (default Z = axis 2).
4. For each slice:
     a. Build a binary mask per output class according to --label-mode.
     b. Extract polygon contours via OpenCV findContours.
     c. Simplify each polygon with approxPolyDP for compact YOLO labels.
     d. If at least one polygon is kept, write the modality slice as PNG
        and emit a YOLO-seg label file with one line per polygon.
     e. If the slice has no tumor pixels, skip (unless --include-empty).
5. Outputs are written to:
       <output>/images/<case>_slice<NNNN>.png
       <output>/labels/<case>_slice<NNNN>.txt

YOLO-seg label format
---------------------
One polygon per line:

    <class_id> x1 y1 x2 y2 x3 y3 ... xN yN

Coordinates are normalised to [0, 1] relative to image width / height.

Usage
-----
Command line:
    python -m src.preprocessing.nifti_to_yolo \\
        --input      data/raw/training_data1_v2 \\
        --output     data/processed/anomaly \\
        --modality   t1c \\
        --label-mode combined

From another module:
    from src.preprocessing.nifti_to_yolo import convert_case, convert_dataset

Dependencies
------------
- nibabel       : NIfTI file I/O
- numpy         : array math
- opencv-python : contour extraction (cv2.findContours, cv2.approxPolyDP)
- Pillow        : PNG writing
- pathlib       : path handling (stdlib)
- argparse      : CLI (stdlib)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import nibabel as nib
import numpy as np
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
# Class taxonomy
# ─────────────────────────────────────────────────────────────────────────────

# BraTS raw integer labels → human-readable name
BRATS_LABEL_NAMES: Dict[int, str] = {
    1: "NETC",   # necrotic tumor core
    2: "SNFH",   # surrounding non-enhancing FLAIR hyperintensity
    3: "ET",     # enhancing tumor
    4: "RC",     # resection cavity (post-treatment)
}

# label-mode → mapping {raw_brats_label: yolo_class_id}
LABEL_MODES: Dict[str, Dict[int, int]] = {
    # All tumor pixels collapsed to a single "glioma" class (id 0).
    # Use this for the top-level Phase 2 anomaly model where glioma is one of
    # several anomaly types (glioma, meningioma, hemorrhage, ...).
    "combined": {1: 0, 2: 0, 3: 0, 4: 0},
    # Sub-region tumor decomposition. Useful for fine-grained segmentation
    # experiments / Phase 3 grading evidence.
    "subregions": {1: 0, 2: 1, 3: 2, 4: 3},
}

# Class names exported per label-mode (index = yolo_class_id)
LABEL_MODE_NAMES: Dict[str, List[str]] = {
    "combined": ["glioma"],
    "subregions": ["NETC", "SNFH", "ET", "RC"],
}

# Supported BraTS modality suffixes (file is <case>-<modality>.nii.gz)
SUPPORTED_MODALITIES: Tuple[str, ...] = ("t1c", "t1n", "t2f", "t2w")


# ─────────────────────────────────────────────────────────────────────────────
# Slice normalisation (kept local so the module is standalone)
# ─────────────────────────────────────────────────────────────────────────────


def zscore_normalise(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Apply per-slice z-score normalisation and rescale to uint8 [0, 255].

    Mirrors the normalisation used in :mod:`dicom_to_png` so the two
    preprocessing pipelines produce visually-comparable PNGs.

    Args:
        arr: 2-D float array (one MRI slice).
        eps: Small constant added to the standard deviation to avoid
             division by zero for uniform / background-only slices.

    Returns:
        2-D uint8 NumPy array suitable for PNG export.
    """
    arr = arr.astype(np.float32)

    mean: float = float(np.mean(arr))
    std: float = float(np.std(arr))

    z: np.ndarray = (arr - mean) / (std + eps)
    z = np.clip(z, -3.0, 3.0)

    z_min, z_max = float(z.min()), float(z.max())
    if z_max - z_min < eps:
        return np.full_like(z, 128, dtype=np.uint8)

    return ((z - z_min) / (z_max - z_min) * 255.0).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Case discovery
# ─────────────────────────────────────────────────────────────────────────────


def find_brats_cases(root: Path) -> List[Path]:
    """Discover BraTS case folders by looking for ``*-seg.nii.gz`` files.

    A case folder is any directory that contains a single segmentation NIfTI
    of the form ``<case>-seg.nii.gz``.

    Args:
        root: Root directory to scan (recursively).

    Returns:
        Sorted list of case folder paths.

    Raises:
        FileNotFoundError: If *root* does not exist.
    """
    if not root.exists():
        raise FileNotFoundError(f"Input directory not found: {root}")

    seg_files = sorted(root.rglob("*-seg.nii.gz"))
    case_dirs = sorted({p.parent for p in seg_files})

    logger.info("Discovered %d BraTS cases under %s", len(case_dirs), root)
    return case_dirs


def resolve_case_files(case_dir: Path, modality: str) -> Tuple[Path, Path]:
    """Resolve the segmentation and modality NIfTI paths for a single case.

    Args:
        case_dir: Directory containing the BraTS case files.
        modality: One of :data:`SUPPORTED_MODALITIES` (e.g. ``"t1c"``).

    Returns:
        Tuple ``(seg_path, modality_path)``.

    Raises:
        ValueError:        If *modality* is not supported.
        FileNotFoundError: If the expected NIfTI files are missing.
    """
    if modality not in SUPPORTED_MODALITIES:
        raise ValueError(
            f"Unsupported modality '{modality}'. "
            f"Choose one of {SUPPORTED_MODALITIES}."
        )

    case_id: str = case_dir.name
    seg_path: Path = case_dir / f"{case_id}-seg.nii.gz"
    mod_path: Path = case_dir / f"{case_id}-{modality}.nii.gz"

    if not seg_path.exists():
        raise FileNotFoundError(f"Segmentation file not found: {seg_path}")
    if not mod_path.exists():
        raise FileNotFoundError(f"Modality file not found: {mod_path}")

    return seg_path, mod_path


# ─────────────────────────────────────────────────────────────────────────────
# Volume I/O
# ─────────────────────────────────────────────────────────────────────────────


def load_nifti_volume(path: Path) -> np.ndarray:
    """Load a NIfTI file and return its voxel array.

    Args:
        path: Path to a ``.nii`` or ``.nii.gz`` file.

    Returns:
        3-D NumPy array in voxel order as stored on disk.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"NIfTI file not found: {path}")

    img = nib.load(str(path))
    arr = np.asarray(img.dataobj)
    return arr


def iter_axial_slices(
    volume: np.ndarray, axis: int
) -> Sequence[Tuple[int, np.ndarray]]:
    """Yield ``(index, slice)`` pairs along the given axis.

    Args:
        volume: 3-D NumPy array.
        axis:   Axis along which to slice (0, 1, or 2).

    Returns:
        List of ``(slice_index, 2D_array)`` tuples in increasing index order.

    Raises:
        ValueError: If *volume* is not 3-D or *axis* is out of range.
    """
    if volume.ndim != 3:
        raise ValueError(
            f"Expected a 3-D volume, got shape {volume.shape}"
        )
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, or 2 — got {axis}")

    n_slices: int = volume.shape[axis]
    out: List[Tuple[int, np.ndarray]] = []
    for i in range(n_slices):
        sl = np.take(volume, indices=i, axis=axis)
        out.append((i, sl))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Polygon extraction
# ─────────────────────────────────────────────────────────────────────────────


def mask_to_polygons(
    binary_mask: np.ndarray,
    min_area: float = 10.0,
    approx_epsilon_frac: float = 0.005,
) -> List[np.ndarray]:
    """Extract polygon contours from a 2-D binary mask.

    Each returned polygon is an ``(N, 2)`` array of (x, y) pixel coordinates
    after Douglas-Peucker simplification.

    Args:
        binary_mask:           2-D uint8 array (0 = background, >0 = object).
        min_area:              Discard polygons smaller than this many pixels.
        approx_epsilon_frac:   Epsilon for ``cv2.approxPolyDP`` expressed as
                               a fraction of the polygon's perimeter.

    Returns:
        List of polygon vertex arrays (may be empty).
    """
    if binary_mask.dtype != np.uint8:
        binary_mask = binary_mask.astype(np.uint8)

    # RETR_EXTERNAL: only outer contours (we treat holes as filled regions —
    # YOLO-seg has no concept of polygon holes anyway).
    contours, _ = cv2.findContours(
        binary_mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE
    )

    polygons: List[np.ndarray] = []
    for cnt in contours:
        area: float = float(cv2.contourArea(cnt))
        if area < min_area:
            continue

        perimeter: float = float(cv2.arcLength(cnt, closed=True))
        epsilon: float = max(1.0, approx_epsilon_frac * perimeter)
        approx = cv2.approxPolyDP(cnt, epsilon=epsilon, closed=True)

        # YOLO-seg requires at least 3 distinct points to form a polygon.
        if approx.shape[0] < 3:
            continue

        polygons.append(approx.reshape(-1, 2))

    return polygons


def polygon_to_yolo_line(
    polygon: np.ndarray,
    class_id: int,
    image_width: int,
    image_height: int,
) -> str:
    """Convert a pixel-space polygon to a single YOLO-seg label line.

    Args:
        polygon:       (N, 2) array of (x, y) integer pixel coordinates.
        class_id:      YOLO class id (non-negative integer).
        image_width:   Width of the source image in pixels.
        image_height:  Height of the source image in pixels.

    Returns:
        A single-line string in the format
        ``"<class_id> x1 y1 x2 y2 ... xN yN"`` with coordinates normalised
        to [0, 1].
    """
    parts: List[str] = [str(int(class_id))]
    for x, y in polygon:
        xn: float = float(x) / float(image_width)
        yn: float = float(y) / float(image_height)
        # Clamp into [0, 1] just in case of border rounding artefacts.
        xn = min(max(xn, 0.0), 1.0)
        yn = min(max(yn, 0.0), 1.0)
        parts.append(f"{xn:.6f}")
        parts.append(f"{yn:.6f}")
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Per-case conversion
# ─────────────────────────────────────────────────────────────────────────────


def convert_case(
    case_dir: Path,
    output_dir: Path,
    modality: str = "t1c",
    label_mode: str = "combined",
    axis: int = 2,
    include_empty: bool = False,
    min_polygon_area: float = 10.0,
    approx_epsilon_frac: float = 0.005,
) -> List[Path]:
    """Convert a single BraTS case into PNG + YOLO-seg label pairs.

    Args:
        case_dir:            Path to the BraTS case folder.
        output_dir:          Root output directory; ``images/`` and ``labels/``
                             subfolders will be created underneath.
        modality:            Modality used as the image (default ``"t1c"``).
        label_mode:          Either ``"combined"`` or ``"subregions"``.
        axis:                Axis along which to iterate slices (default 2 = Z).
        include_empty:       If True, also write images and empty label files
                             for slices that contain no tumor pixels.
        min_polygon_area:    Discard polygons smaller than this many pixels.
        approx_epsilon_frac: Douglas-Peucker simplification factor.

    Returns:
        List of PNG paths that were written (label files share the same stem).

    Raises:
        ValueError:        On unsupported modality or label_mode.
        FileNotFoundError: If the expected NIfTI files are missing.
    """
    if label_mode not in LABEL_MODES:
        raise ValueError(
            f"Unknown label_mode '{label_mode}'. "
            f"Choose one of {list(LABEL_MODES)}."
        )

    seg_path, mod_path = resolve_case_files(case_dir, modality)
    case_id: str = case_dir.name

    seg_volume = load_nifti_volume(seg_path).astype(np.int16)
    mod_volume = load_nifti_volume(mod_path)

    if seg_volume.shape != mod_volume.shape:
        raise ValueError(
            f"Shape mismatch in case {case_id}: "
            f"seg={seg_volume.shape} vs {modality}={mod_volume.shape}"
        )

    images_dir: Path = output_dir / "images"
    labels_dir: Path = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    label_map: Dict[int, int] = LABEL_MODES[label_mode]
    written_pngs: List[Path] = []
    n_pos_slices: int = 0

    for slice_idx, (mod_slice, seg_slice) in enumerate(
        zip(
            (s for _, s in iter_axial_slices(mod_volume, axis=axis)),
            (s for _, s in iter_axial_slices(seg_volume, axis=axis)),
        )
    ):
        # Collect polygons per output class for this slice
        per_class_polys: List[Tuple[int, np.ndarray]] = []
        for raw_label, yolo_class in label_map.items():
            binary = (seg_slice == raw_label).astype(np.uint8) * 255
            if binary.sum() == 0:
                continue
            polys = mask_to_polygons(
                binary,
                min_area=min_polygon_area,
                approx_epsilon_frac=approx_epsilon_frac,
            )
            for poly in polys:
                per_class_polys.append((yolo_class, poly))

        has_objects: bool = len(per_class_polys) > 0
        if not has_objects and not include_empty:
            continue

        if has_objects:
            n_pos_slices += 1

        # Normalise and save the modality slice
        png_arr: np.ndarray = zscore_normalise(mod_slice)
        png_name: str = f"{case_id}_slice{slice_idx:04d}.png"
        png_path: Path = images_dir / png_name
        Image.fromarray(png_arr, mode="L").save(str(png_path))
        written_pngs.append(png_path)

        # Emit YOLO-seg label file (may be empty for include_empty=True slices)
        height, width = png_arr.shape
        lines: List[str] = [
            polygon_to_yolo_line(
                polygon=poly,
                class_id=yolo_class,
                image_width=width,
                image_height=height,
            )
            for yolo_class, poly in per_class_polys
        ]

        label_path: Path = labels_dir / f"{case_id}_slice{slice_idx:04d}.txt"
        label_path.write_text("\n".join(lines), encoding="utf-8")

    logger.info(
        "Case %s — wrote %d PNG/label pairs (%d positive, %d empty)",
        case_id,
        len(written_pngs),
        n_pos_slices,
        len(written_pngs) - n_pos_slices,
    )
    return written_pngs


# ─────────────────────────────────────────────────────────────────────────────
# Top-level batch conversion
# ─────────────────────────────────────────────────────────────────────────────


def convert_dataset(
    input_dir: Path,
    output_dir: Path,
    modality: str = "t1c",
    label_mode: str = "combined",
    axis: int = 2,
    include_empty: bool = False,
    min_polygon_area: float = 10.0,
    approx_epsilon_frac: float = 0.005,
) -> Dict[str, List[Path]]:
    """Convert an entire BraTS root directory of case folders.

    Args:
        input_dir:           Directory containing BraTS case sub-folders.
        output_dir:          Root output directory (images/ and labels/ will
                             be populated underneath).
        modality:            Modality used as the image (default ``"t1c"``).
        label_mode:          Either ``"combined"`` or ``"subregions"``.
        axis:                Slicing axis (default 2 = axial Z).
        include_empty:       Pass-through to :func:`convert_case`.
        min_polygon_area:    Pass-through to :func:`convert_case`.
        approx_epsilon_frac: Pass-through to :func:`convert_case`.

    Returns:
        Dictionary mapping case id → list of written PNG paths.

    Raises:
        FileNotFoundError: If *input_dir* contains no BraTS cases.
        ValueError:        On unsupported modality or label_mode.
    """
    cases: List[Path] = find_brats_cases(input_dir)
    if not cases:
        raise FileNotFoundError(
            f"No BraTS cases (*-seg.nii.gz) found under {input_dir}"
        )

    results: Dict[str, List[Path]] = {}
    failures: List[Tuple[str, str]] = []

    for case_dir in cases:
        try:
            pngs = convert_case(
                case_dir=case_dir,
                output_dir=output_dir,
                modality=modality,
                label_mode=label_mode,
                axis=axis,
                include_empty=include_empty,
                min_polygon_area=min_polygon_area,
                approx_epsilon_frac=approx_epsilon_frac,
            )
            results[case_dir.name] = pngs
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("Skipping case %s — %s", case_dir.name, exc)
            failures.append((case_dir.name, str(exc)))
            continue

    total_pairs: int = sum(len(v) for v in results.values())
    logger.info(
        "Conversion complete — %d cases, %d PNG/label pairs, %d failures → %s",
        len(results),
        total_pairs,
        len(failures),
        output_dir,
    )
    if failures:
        logger.warning("Failed cases:")
        for name, reason in failures:
            logger.warning("  %s — %s", name, reason)

    # Emit dataset.yaml hint for convenience
    _write_dataset_yaml(output_dir, label_mode)
    return results


def _write_dataset_yaml(output_dir: Path, label_mode: str) -> Path:
    """Write a minimal Ultralytics-style ``dataset.yaml`` next to the outputs.

    The user can edit this file (train/val splits, augmentation) before
    handing it to ``yolo segment train data=...``.

    Args:
        output_dir: Root output directory (containing images/ and labels/).
        label_mode: Label mode used during conversion.

    Returns:
        Path to the written YAML file.
    """
    names: List[str] = LABEL_MODE_NAMES[label_mode]
    names_block: str = "\n".join(f"  {i}: {n}" for i, n in enumerate(names))

    yaml_text: str = (
        f"# Auto-generated by nifti_to_yolo.py — edit splits before training.\n"
        f"path: {output_dir.resolve()}\n"
        f"train: images\n"
        f"val: images   # TODO: replace with a held-out split\n"
        f"names:\n{names_block}\n"
    )
    yaml_path: Path = output_dir / "dataset.yaml"
    yaml_path.write_text(yaml_text, encoding="utf-8")
    logger.info("Wrote dataset stub: %s", yaml_path)
    return yaml_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to sys.argv if None).

    Returns:
        Parsed Namespace.
    """
    parser = argparse.ArgumentParser(
        prog="nifti_to_yolo",
        description=(
            "Convert BraTS-style NIfTI segmentation volumes into a YOLOv11-seg "
            "training dataset: per-slice PNG images + polygon label files."
        ),
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        metavar="DIR",
        help=(
            "Root directory containing BraTS case sub-folders "
            "(e.g. data/raw/training_data1_v2)."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("data/processed/anomaly"),
        metavar="DIR",
        help=(
            "Output root. images/ and labels/ subfolders will be created "
            "underneath. Default: data/processed/anomaly"
        ),
    )
    parser.add_argument(
        "--modality",
        "-m",
        type=str,
        default="t1c",
        choices=SUPPORTED_MODALITIES,
        help="BraTS modality to use as the image (default: t1c).",
    )
    parser.add_argument(
        "--label-mode",
        type=str,
        default="combined",
        choices=list(LABEL_MODES),
        help=(
            "How to map BraTS labels 1-4 to YOLO classes. "
            "'combined' → single glioma class; "
            "'subregions' → NETC / SNFH / ET / RC."
        ),
    )
    parser.add_argument(
        "--axis",
        type=int,
        default=2,
        choices=(0, 1, 2),
        help="Axis along which to slice the volume (default: 2 = axial Z).",
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        default=False,
        help=(
            "Also write images / empty label files for slices with no tumor "
            "pixels. Off by default — keeps the training set focused."
        ),
    )
    parser.add_argument(
        "--min-polygon-area",
        type=float,
        default=10.0,
        metavar="PIXELS",
        help=(
            "Discard polygons smaller than this many pixels. "
            "Filters out single-voxel speckle artefacts. Default: 10."
        ),
    )
    parser.add_argument(
        "--approx-epsilon-frac",
        type=float,
        default=0.005,
        metavar="FRAC",
        help=(
            "Douglas-Peucker simplification factor as a fraction of polygon "
            "perimeter. Smaller → more vertices, larger → coarser. Default: 0.005."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry-point for nifti_to_yolo.

    Args:
        argv: Argument list (defaults to sys.argv if None).

    Raises:
        SystemExit: With code 1 on fatal error.
    """
    args = _parse_args(argv)

    logger.info("Input directory   : %s", args.input.resolve())
    logger.info("Output directory  : %s", args.output.resolve())
    logger.info("Modality          : %s", args.modality)
    logger.info("Label mode        : %s", args.label_mode)
    logger.info("Slice axis        : %d", args.axis)
    logger.info("Include empty     : %s", args.include_empty)
    logger.info("Min polygon area  : %.1f px", args.min_polygon_area)
    logger.info("Approx epsilon    : %.4f * perimeter", args.approx_epsilon_frac)

    try:
        convert_dataset(
            input_dir=args.input,
            output_dir=args.output,
            modality=args.modality,
            label_mode=args.label_mode,
            axis=args.axis,
            include_empty=args.include_empty,
            min_polygon_area=args.min_polygon_area,
            approx_epsilon_frac=args.approx_epsilon_frac,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Fatal error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
