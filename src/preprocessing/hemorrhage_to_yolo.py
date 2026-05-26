"""
hemorrhage_to_yolo.py
=====================
Convert the PhysioNet Hssayeni intracranial hemorrhage CT dataset into a
YOLOv11-seg compatible 2D training dataset (PNG slices + polygon label files).

Dataset layout (relative to the dataset root)
----------------------------------------------
    hemorrhage_diagnosis.csv               ← per-slice multi-label annotations
    patient_demographics.csv               ← (not used here)
    Patients_CT/<patient_id>/brain/<slice_num>.jpg        ← brain-windowed CT slice
    Patients_CT/<patient_id>/bone/<slice_num>.jpg         ← bone-windowed CT (unused)
    Patients_CT/<patient_id>/brain/<slice_num>_HGE_Seg.jpg ← hemorrhage mask
                                                              (present only if hemorrhage)

hemorrhage_diagnosis.csv columns
---------------------------------
    PatientNumber   : integer patient id  → folder name under Patients_CT/
    SliceNumber     : integer slice id    → image file stem (0-padded in folder)
    Intraventricular: 0 / 1
    Intraparenchymal: 0 / 1
    Subarachnoid    : 0 / 1
    Epidural        : 0 / 1
    Subdural        : 0 / 1
    No_Hemorrhage   : 0 / 1              (1 = negative scan)
    Fracture_Yes_No : 0 / 1              (not used for segmentation)

Label modes
-----------
    combined  — all hemorrhage pixels → a single configurable class id
                (default 1, so glioma=0 and hemorrhage=1 in the unified dataset)
    subtypes  — use the per-slice CSV annotation to assign a class id per subtype:
                    Intraventricular = 0
                    Intraparenchymal = 1
                    Subarachnoid     = 2
                    Epidural         = 3
                    Subdural         = 4
                Since only one combined mask exists per slice, multi-label slices
                emit one polygon per active subtype (same spatial region, different
                class ids). This is an acknowledged approximation.

Pipeline
--------
1. Load hemorrhage_diagnosis.csv.
2. Filter to rows where No_Hemorrhage == 0 (positive scans).
3. For each (patient, slice):
     a. Locate the brain JPG and the *_HGE_Seg.jpg mask.
     b. Binarise the mask (threshold > 0).
     c. Extract polygon contours with OpenCV (same logic as nifti_to_yolo).
     d. Build the class list from the label-mode.
     e. Write the brain CT slice as PNG + one YOLO-seg label file.
4. Optionally include negative slices (--include-empty).
5. Emit a dataset.yaml stub.

Output structure mirrors nifti_to_yolo.py:
    <output>/images/<patient_id>_slice<NNNN>.png
    <output>/labels/<patient_id>_slice<NNNN>.txt

YOLO-seg label format
---------------------
One polygon per line:
    <class_id> x1 y1 x2 y2 ... xN yN
Coordinates normalised to [0, 1].

Usage
-----
    python -m src.preprocessing.hemorrhage_to_yolo \\
        --input   data/raw/hemorrhage-ct/computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.0.0 \\
        --output  data/processed/anomaly \\
        --label-mode combined \\
        --combined-class-id 1

Dependencies
------------
- numpy         : array math
- opencv-python : contour extraction
- pandas        : CSV parsing
- Pillow        : PNG writing
- pathlib       : path handling (stdlib)
- argparse      : CLI (stdlib)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
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

# Ordered list of subtype column names in the CSV
SUBTYPE_COLUMNS: List[str] = [
    "Intraventricular",
    "Intraparenchymal",
    "Subarachnoid",
    "Epidural",
    "Subdural",
]

# In subtypes mode, each subtype column maps to a YOLO class id.
# The ordering matches SUBTYPE_COLUMNS (index = yolo class id for subtypes mode).
SUBTYPE_CLASS_IDS: Dict[str, int] = {
    col: idx for idx, col in enumerate(SUBTYPE_COLUMNS)
}

# Class names exported per label-mode
LABEL_MODE_NAMES: Dict[str, List[str]] = {
    "combined": ["hemorrhage"],  # updated at write time with the actual id
    "subtypes": SUBTYPE_COLUMNS,
}

SUPPORTED_LABEL_MODES: Tuple[str, ...] = ("combined", "subtypes")


# ─────────────────────────────────────────────────────────────────────────────
# CSV loading
# ─────────────────────────────────────────────────────────────────────────────


def load_diagnosis_csv(dataset_root: Path) -> pd.DataFrame:
    """Load and validate the hemorrhage_diagnosis.csv file.

    Args:
        dataset_root: Root directory of the Hssayeni dataset (contains the CSV).

    Returns:
        DataFrame with columns: PatientNumber, SliceNumber, Intraventricular,
        Intraparenchymal, Subarachnoid, Epidural, Subdural, No_Hemorrhage,
        Fracture_Yes_No.

    Raises:
        FileNotFoundError: If the CSV is not found.
        ValueError:        If expected columns are missing.
    """
    csv_path: Path = dataset_root / "hemorrhage_diagnosis.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Diagnosis CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = {
        "PatientNumber",
        "SliceNumber",
        "No_Hemorrhage",
        *SUBTYPE_COLUMNS,
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"hemorrhage_diagnosis.csv is missing columns: {sorted(missing)}"
        )

    logger.info(
        "Loaded %d rows from %s", len(df), csv_path.name
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Path resolution
# ─────────────────────────────────────────────────────────────────────────────


def resolve_slice_paths(
    dataset_root: Path, patient_id: int, slice_num: int
) -> Tuple[Optional[Path], Optional[Path]]:
    """Resolve the brain JPG and mask JPG for a single (patient, slice) pair.

    The Hssayeni dataset uses bare integer filenames with no zero-padding.
    The mask file ``<slice_num>_HGE_Seg.jpg`` is absent for negative slices.

    Args:
        dataset_root: Root directory of the Hssayeni dataset.
        patient_id:   Integer patient number (maps to a Patients_CT sub-folder).
        slice_num:    Integer slice number (file stem inside the brain/ folder).

    Returns:
        Tuple ``(brain_jpg_path, mask_jpg_path)``.  Either value may be ``None``
        if the corresponding file does not exist.
    """
    brain_dir: Path = (
        dataset_root / "Patients_CT" / f"{patient_id:03d}" / "brain"
    )
    brain_jpg: Path = brain_dir / f"{slice_num}.jpg"
    mask_jpg: Path = brain_dir / f"{slice_num}_HGE_Seg.jpg"

    return (
        brain_jpg if brain_jpg.exists() else None,
        mask_jpg if mask_jpg.exists() else None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Polygon extraction (mirrors nifti_to_yolo logic)
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
        binary_mask:          2-D uint8 array (0 = background, >0 = object).
        min_area:             Discard polygons smaller than this many pixels.
        approx_epsilon_frac:  Epsilon for ``cv2.approxPolyDP`` expressed as
                              a fraction of the polygon's perimeter.

    Returns:
        List of polygon vertex arrays (may be empty).
    """
    if binary_mask.dtype != np.uint8:
        binary_mask = binary_mask.astype(np.uint8)

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
        A string ``"<class_id> x1 y1 x2 y2 ... xN yN"`` with coordinates
        normalised to [0, 1].
    """
    parts: List[str] = [str(int(class_id))]
    for x, y in polygon:
        xn = min(max(float(x) / float(image_width), 0.0), 1.0)
        yn = min(max(float(y) / float(image_height), 0.0), 1.0)
        parts.append(f"{xn:.6f}")
        parts.append(f"{yn:.6f}")
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Per-slice conversion
# ─────────────────────────────────────────────────────────────────────────────


def convert_slice(
    brain_jpg: Path,
    mask_jpg: Optional[Path],
    class_ids: List[int],
    output_dir: Path,
    out_stem: str,
    min_polygon_area: float = 10.0,
    approx_epsilon_frac: float = 0.005,
    include_empty: bool = False,
) -> bool:
    """Convert a single CT slice + mask into a PNG image and YOLO-seg label.

    If ``mask_jpg`` is None or produces no valid polygons, the slice is only
    written when ``include_empty`` is True (empty label file).

    Args:
        brain_jpg:           Path to the brain-windowed CT slice JPEG.
        mask_jpg:            Path to the hemorrhage mask JPEG, or None.
        class_ids:           List of YOLO class ids to emit for this slice.
                             - combined mode: single-element list [combined_id]
                             - subtypes mode: one id per active subtype
                             Each gets the same spatial polygon(s) from the mask.
        output_dir:          Root output directory (images/ and labels/ beneath).
        out_stem:            Output file stem (no extension), e.g.
                             ``"049_slice0023"``.
        min_polygon_area:    Passed to :func:`mask_to_polygons`.
        approx_epsilon_frac: Passed to :func:`mask_to_polygons`.
        include_empty:       Write image + empty label for no-annotation slices.

    Returns:
        True if an image was written, False if the slice was skipped.
    """
    images_dir: Path = output_dir / "images"
    labels_dir: Path = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # ── Load and normalise the brain CT slice ──────────────────────────────
    brain_img = np.array(Image.open(str(brain_jpg)).convert("L"), dtype=np.float32)
    # Normalise brain slice to uint8 [0, 255] via min-max (CT JPEGs already
    # have reasonable window/level baked in — z-score would over-stretch).
    b_min, b_max = brain_img.min(), brain_img.max()
    if b_max - b_min < 1e-8:
        brain_uint8 = np.full_like(brain_img, 128, dtype=np.uint8)
    else:
        brain_uint8 = (
            (brain_img - b_min) / (b_max - b_min) * 255.0
        ).astype(np.uint8)

    height, width = brain_uint8.shape

    # ── Extract polygons from the mask ─────────────────────────────────────
    label_lines: List[str] = []

    if mask_jpg is not None and mask_jpg.exists():
        mask_gray = np.array(
            Image.open(str(mask_jpg)).convert("L"), dtype=np.uint8
        )
        # Binarise: any non-zero pixel is hemorrhage.
        binary_mask = (mask_gray > 0).astype(np.uint8) * 255

        polygons = mask_to_polygons(
            binary_mask,
            min_area=min_polygon_area,
            approx_epsilon_frac=approx_epsilon_frac,
        )

        # Emit one set of polygons per class id.
        for cid in class_ids:
            for poly in polygons:
                label_lines.append(
                    polygon_to_yolo_line(poly, cid, width, height)
                )

    has_objects: bool = len(label_lines) > 0

    if not has_objects and not include_empty:
        return False

    # ── Write PNG ──────────────────────────────────────────────────────────
    png_path: Path = images_dir / f"{out_stem}.png"
    Image.fromarray(brain_uint8, mode="L").save(str(png_path))

    # ── Write label ────────────────────────────────────────────────────────
    label_path: Path = labels_dir / f"{out_stem}.txt"
    label_path.write_text("\n".join(label_lines), encoding="utf-8")

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Top-level batch conversion
# ─────────────────────────────────────────────────────────────────────────────


def convert_dataset(
    dataset_root: Path,
    output_dir: Path,
    label_mode: str = "combined",
    combined_class_id: int = 1,
    include_empty: bool = False,
    min_polygon_area: float = 10.0,
    approx_epsilon_frac: float = 0.005,
) -> Dict[str, int]:
    """Convert the entire Hssayeni hemorrhage CT dataset to YOLO-seg format.

    Args:
        dataset_root:        Root of the Hssayeni dataset (contains the CSV and
                             the ``Patients_CT/`` folder).
        output_dir:          Destination root; ``images/`` and ``labels/``
                             sub-folders will be created underneath.
        label_mode:          ``"combined"`` or ``"subtypes"``.
        combined_class_id:   YOLO class id used for all hemorrhage polygons in
                             ``combined`` mode (default 1; use 0 for single-class
                             hemorrhage-only datasets, 1 to sit alongside glioma=0).
        include_empty:       Write images + empty label files for negative slices.
        min_polygon_area:    Discard polygons smaller than this many pixels.
        approx_epsilon_frac: Douglas-Peucker simplification factor.

    Returns:
        Summary dict with keys ``written``, ``positive``, ``skipped``, ``errors``.

    Raises:
        FileNotFoundError: If the CSV or Patients_CT folder is missing.
        ValueError:        On unsupported label_mode.
    """
    if label_mode not in SUPPORTED_LABEL_MODES:
        raise ValueError(
            f"Unknown label_mode '{label_mode}'. "
            f"Choose one of {SUPPORTED_LABEL_MODES}."
        )

    patients_dir: Path = dataset_root / "Patients_CT"
    if not patients_dir.exists():
        raise FileNotFoundError(
            f"Patients_CT directory not found under {dataset_root}"
        )

    df = load_diagnosis_csv(dataset_root)

    written = 0
    positive = 0
    skipped = 0
    errors = 0

    for _, row in df.iterrows():
        patient_id: int = int(row["PatientNumber"])
        slice_num: int = int(row["SliceNumber"])
        no_hemorrhage: bool = bool(int(row["No_Hemorrhage"]))

        out_stem: str = f"{patient_id:04d}_slice{slice_num:04d}"

        # ── Determine class ids for this slice ─────────────────────────────
        if no_hemorrhage:
            class_ids: List[int] = []  # negative slice
        elif label_mode == "combined":
            class_ids = [combined_class_id]
        else:  # subtypes
            class_ids = [
                SUBTYPE_CLASS_IDS[col]
                for col in SUBTYPE_COLUMNS
                if int(row.get(col, 0)) == 1
            ]

        # ── Resolve file paths ─────────────────────────────────────────────
        try:
            brain_jpg, mask_jpg = resolve_slice_paths(
                dataset_root, patient_id, slice_num
            )
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "Patient %d slice %d — path resolution error: %s",
                patient_id, slice_num, exc,
            )
            errors += 1
            continue

        if brain_jpg is None:
            logger.debug(
                "Patient %d slice %d — brain JPG not found, skipping.",
                patient_id, slice_num,
            )
            skipped += 1
            continue

        # Negative slices have no mask; we still need a Path placeholder.
        # include_empty will decide whether to write them.
        if no_hemorrhage and not include_empty:
            skipped += 1
            continue

        # ── Convert ────────────────────────────────────────────────────────
        try:
            did_write = convert_slice(
                brain_jpg=brain_jpg,
                mask_jpg=mask_jpg if not no_hemorrhage else None,
                class_ids=class_ids,
                output_dir=output_dir,
                out_stem=out_stem,
                min_polygon_area=min_polygon_area,
                approx_epsilon_frac=approx_epsilon_frac,
                include_empty=include_empty,
            )
        except Exception as exc:
            logger.warning(
                "Patient %d slice %d — conversion error: %s",
                patient_id, slice_num, exc,
            )
            errors += 1
            continue

        if did_write:
            written += 1
            if not no_hemorrhage and class_ids:
                positive += 1
        else:
            skipped += 1

    summary = {
        "written": written,
        "positive": positive,
        "skipped": skipped,
        "errors": errors,
    }
    logger.info(
        "Hemorrhage conversion complete — "
        "written=%d (positive=%d), skipped=%d, errors=%d → %s",
        written, positive, skipped, errors, output_dir,
    )

    _write_dataset_yaml(output_dir, label_mode, combined_class_id)
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Dataset YAML
# ─────────────────────────────────────────────────────────────────────────────


def _write_dataset_yaml(
    output_dir: Path, label_mode: str, combined_class_id: int
) -> Path:
    """Write an Ultralytics-style dataset.yaml stub next to the outputs.

    If the file already exists (created by nifti_to_yolo.py for the shared
    output directory), this function does NOT overwrite it — the shared dataset
    yaml must be edited manually to reflect the combined class taxonomy.

    Args:
        output_dir:         Root output directory.
        label_mode:         Label mode used during conversion.
        combined_class_id:  Class id used in combined mode (for the comment).

    Returns:
        Path to the written (or existing) YAML file.
    """
    yaml_path: Path = output_dir / "dataset.yaml"

    if yaml_path.exists():
        logger.info(
            "dataset.yaml already exists — not overwriting. "
            "Manually verify the class taxonomy includes hemorrhage classes."
        )
        return yaml_path

    if label_mode == "combined":
        names_block = f"  {combined_class_id}: hemorrhage"
    else:
        names_block = "\n".join(
            f"  {SUBTYPE_CLASS_IDS[c]}: {c}" for c in SUBTYPE_COLUMNS
        )

    yaml_text: str = (
        f"# Auto-generated by hemorrhage_to_yolo.py — edit before training.\n"
        f"# NOTE: if mixing with BraTS data, reconcile class ids manually.\n"
        f"path: {output_dir.resolve()}\n"
        f"train: images\n"
        f"val: images   # TODO: replace with a held-out split\n"
        f"names:\n{names_block}\n"
    )
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
        prog="hemorrhage_to_yolo",
        description=(
            "Convert the PhysioNet Hssayeni intracranial hemorrhage CT dataset "
            "into a YOLOv11-seg training dataset: PNG slices + polygon labels."
        ),
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        metavar="DIR",
        help=(
            "Root directory of the Hssayeni dataset — the folder that contains "
            "hemorrhage_diagnosis.csv and the Patients_CT/ sub-folder. "
            "e.g. data/raw/hemorrhage-ct/computed-tomography-images-for-"
            "intracranial-hemorrhage-detection-and-segmentation-1.0.0"
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("data/processed/anomaly"),
        metavar="DIR",
        help=(
            "Output root. images/ and labels/ will be created underneath. "
            "Default: data/processed/anomaly (shared with nifti_to_yolo output)."
        ),
    )
    parser.add_argument(
        "--label-mode",
        type=str,
        default="combined",
        choices=list(SUPPORTED_LABEL_MODES),
        help=(
            "'combined' → all hemorrhage as one class (see --combined-class-id). "
            "'subtypes' → per-subtype class ids (Intraventricular=0 … Subdural=4)."
        ),
    )
    parser.add_argument(
        "--combined-class-id",
        type=int,
        default=1,
        metavar="INT",
        help=(
            "YOLO class id for hemorrhage in combined mode. "
            "Set to 1 when combining with BraTS data (glioma=0). "
            "Set to 0 for a standalone hemorrhage-only dataset. Default: 1."
        ),
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        default=False,
        help=(
            "Also write images / empty label files for negative (no-hemorrhage) "
            "slices. Off by default."
        ),
    )
    parser.add_argument(
        "--min-polygon-area",
        type=float,
        default=10.0,
        metavar="PIXELS",
        help="Discard polygons smaller than this many pixels. Default: 10.",
    )
    parser.add_argument(
        "--approx-epsilon-frac",
        type=float,
        default=0.005,
        metavar="FRAC",
        help=(
            "Douglas-Peucker simplification factor as a fraction of polygon "
            "perimeter. Default: 0.005."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry-point for hemorrhage_to_yolo.

    Args:
        argv: Argument list (defaults to sys.argv if None).

    Raises:
        SystemExit: With code 1 on fatal error.
    """
    args = _parse_args(argv)

    logger.info("Input directory      : %s", args.input.resolve())
    logger.info("Output directory     : %s", args.output.resolve())
    logger.info("Label mode           : %s", args.label_mode)
    logger.info("Combined class id    : %d", args.combined_class_id)
    logger.info("Include empty        : %s", args.include_empty)
    logger.info("Min polygon area     : %.1f px", args.min_polygon_area)
    logger.info("Approx epsilon       : %.4f * perimeter", args.approx_epsilon_frac)

    try:
        convert_dataset(
            dataset_root=args.input,
            output_dir=args.output,
            label_mode=args.label_mode,
            combined_class_id=args.combined_class_id,
            include_empty=args.include_empty,
            min_polygon_area=args.min_polygon_area,
            approx_epsilon_frac=args.approx_epsilon_frac,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Fatal error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
