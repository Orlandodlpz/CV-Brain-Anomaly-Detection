"""
visualization.py
================
Overlay YOLOv11-seg polygon labels on their matching PNG slices so the
annotations can be inspected visually.

Use this to sanity-check the output of ``nifti_to_yolo.py`` (and later
``hemorrhage_to_yolo.py``) before kicking off a full training run — if a
polygon doesn't actually trace the tumor edge, training will be wasted.

Pipeline
--------
1. Walk the --images directory (PNG slices).
2. For each PNG, look up the matching ``.txt`` label file in --labels.
3. Parse each polygon line:
       <class_id> x1 y1 x2 y2 ... xN yN     (xi, yi normalised to [0, 1])
4. Denormalise to pixel coords and draw the polygon outline on the slice,
   colour-coded by class.
5. Save the annotated copy to --output (PNG, same filename).

Usage
-----
Command line:
    python -m src.utils.visualization \\
        --images data/processed/anomaly_smoketest/images \\
        --labels data/processed/anomaly_smoketest/labels \\
        --output data/processed/anomaly_smoketest/overlays \\
        --limit 20

From another module:
    from src.utils.visualization import overlay_yolo_seg

Dependencies
------------
- opencv-python : polyline drawing, BGR conversion
- numpy         : array math
- pathlib       : path handling (stdlib)
- argparse      : CLI (stdlib)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

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
# Colour palette (BGR — OpenCV's native order)
# ─────────────────────────────────────────────────────────────────────────────

# Hand-picked high-contrast palette. Cycles by class id; index 0 is "glioma"
# in combined mode and "NETC" in subregions mode.
CLASS_PALETTE_BGR: List[Tuple[int, int, int]] = [
    (0, 255, 255),    # 0 — yellow
    (255, 64, 64),    # 1 — bright blue
    (64, 255, 64),    # 2 — green
    (64, 64, 255),    # 3 — red
    (255, 0, 255),    # 4 — magenta
    (255, 255, 0),    # 5 — cyan
]


def class_color(class_id: int) -> Tuple[int, int, int]:
    """Return a BGR colour for the given class id (cycles through the palette).

    Args:
        class_id: Non-negative integer class id.

    Returns:
        ``(B, G, R)`` tuple of ints in [0, 255].
    """
    return CLASS_PALETTE_BGR[class_id % len(CLASS_PALETTE_BGR)]


# ─────────────────────────────────────────────────────────────────────────────
# Label parsing
# ─────────────────────────────────────────────────────────────────────────────


def parse_yolo_seg_label(
    label_path: Path,
) -> List[Tuple[int, np.ndarray]]:
    """Parse a YOLOv11-seg label file into ``(class_id, polygon)`` pairs.

    Args:
        label_path: Path to a ``.txt`` label file. Each line is
                    ``"<class_id> x1 y1 x2 y2 ... xN yN"`` with coordinates
                    normalised to [0, 1].

    Returns:
        List of ``(class_id, polygon)`` pairs. ``polygon`` is an ``(N, 2)``
        float array of normalised (x, y) coordinates.

    Raises:
        FileNotFoundError: If *label_path* does not exist.
        ValueError:        If a line has fewer than 3 coordinate pairs.
    """
    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    polygons: List[Tuple[int, np.ndarray]] = []
    for line_no, raw in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        tokens = raw.strip().split()
        if not tokens:
            continue  # blank line

        class_id = int(tokens[0])
        coords = np.asarray(tokens[1:], dtype=np.float32)
        if coords.size % 2 != 0:
            raise ValueError(
                f"{label_path}:{line_no} has an odd number of coordinates"
            )
        polygon = coords.reshape(-1, 2)
        if polygon.shape[0] < 3:
            raise ValueError(
                f"{label_path}:{line_no} has only {polygon.shape[0]} vertex/vertices"
            )
        polygons.append((class_id, polygon))

    return polygons


def denormalise_polygon(
    polygon: np.ndarray, image_width: int, image_height: int
) -> np.ndarray:
    """Convert a normalised polygon to integer pixel coordinates.

    Args:
        polygon:      ``(N, 2)`` float array, values in [0, 1].
        image_width:  Source image width in pixels.
        image_height: Source image height in pixels.

    Returns:
        ``(N, 1, 2)`` int32 array in the layout expected by
        ``cv2.polylines`` / ``cv2.fillPoly``.
    """
    pix = polygon.copy()
    pix[:, 0] *= image_width
    pix[:, 1] *= image_height
    return pix.round().astype(np.int32).reshape(-1, 1, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Overlay rendering
# ─────────────────────────────────────────────────────────────────────────────


def overlay_yolo_seg(
    image: np.ndarray,
    polygons: Sequence[Tuple[int, np.ndarray]],
    thickness: int = 1,
    alpha: float = 0.0,
) -> np.ndarray:
    """Draw YOLO-seg polygons on top of an image.

    Args:
        image:     2-D uint8 grayscale or 3-D uint8 BGR image.
        polygons:  Sequence of ``(class_id, normalised_polygon)`` pairs as
                   produced by :func:`parse_yolo_seg_label`.
        thickness: Outline thickness in pixels. Pass a negative value to fill.
        alpha:     If > 0, also blend a filled mask over the image with this
                   opacity (0 = outline only, 1 = fully opaque fill).

    Returns:
        3-D uint8 BGR image with overlays drawn.
    """
    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 3:
        bgr = image.copy()
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    height, width = bgr.shape[:2]

    # Optional filled-mask blend pass (drawn first so outline goes on top)
    if alpha > 0:
        fill_layer = bgr.copy()
        for class_id, poly in polygons:
            color = class_color(class_id)
            pts = denormalise_polygon(poly, width, height)
            cv2.fillPoly(fill_layer, [pts], color)
        bgr = cv2.addWeighted(fill_layer, alpha, bgr, 1.0 - alpha, 0.0)

    # Outline pass
    for class_id, poly in polygons:
        color = class_color(class_id)
        pts = denormalise_polygon(poly, width, height)
        cv2.polylines(
            bgr,
            [pts],
            isClosed=True,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

    return bgr


# ─────────────────────────────────────────────────────────────────────────────
# Batch driver
# ─────────────────────────────────────────────────────────────────────────────


def overlay_directory(
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    limit: Optional[int] = None,
    thickness: int = 1,
    alpha: float = 0.0,
    skip_empty: bool = False,
) -> List[Path]:
    """Generate overlay PNGs for every image that has a matching label file.

    Args:
        images_dir:  Directory containing source PNG slices.
        labels_dir:  Directory containing matching ``.txt`` label files
                     (same stem as the image).
        output_dir:  Destination for overlay PNGs. Created if missing.
        limit:       If set, process at most this many images.
        thickness:   Polygon outline thickness (pixels).
        alpha:       Filled-mask blend opacity (0 = outlines only).
        skip_empty:  If True, do not write overlays for slices whose label
                     file is empty (no polygons).

    Returns:
        List of overlay paths that were written.

    Raises:
        FileNotFoundError: If *images_dir* does not exist or is empty.
    """
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths: List[Path] = sorted(images_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No PNG images found under {images_dir}")

    if limit is not None:
        image_paths = image_paths[:limit]

    written: List[Path] = []
    n_skipped: int = 0
    n_missing_label: int = 0

    for img_path in image_paths:
        label_path: Path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            n_missing_label += 1
            continue

        polygons = parse_yolo_seg_label(label_path)
        if not polygons and skip_empty:
            n_skipped += 1
            continue

        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            logger.warning("Could not read image: %s", img_path)
            continue

        overlay = overlay_yolo_seg(
            image=image,
            polygons=polygons,
            thickness=thickness,
            alpha=alpha,
        )
        out_path: Path = output_dir / img_path.name
        cv2.imwrite(str(out_path), overlay)
        written.append(out_path)

    logger.info(
        "Overlay complete — wrote %d / %d images (%d skipped empty, %d missing label) → %s",
        len(written),
        len(image_paths),
        n_skipped,
        n_missing_label,
        output_dir,
    )
    return written


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
        prog="visualization",
        description=(
            "Overlay YOLOv11-seg polygon labels on their matching PNG slices "
            "and save the annotated copies to an output directory."
        ),
    )
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        metavar="DIR",
        help="Directory of source PNG slices (e.g. .../images).",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        metavar="DIR",
        help="Directory of YOLO-seg .txt label files (e.g. .../labels).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        metavar="DIR",
        help="Destination directory for overlay PNGs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process at most N images (default: all).",
    )
    parser.add_argument(
        "--thickness",
        type=int,
        default=1,
        metavar="PX",
        help="Polygon outline thickness in pixels (default: 1).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        metavar="FLOAT",
        help=(
            "If > 0, also blend a filled mask over the image with this opacity "
            "(0.0 = outline only, 1.0 = fully opaque). Default: 0."
        ),
    )
    parser.add_argument(
        "--skip-empty",
        action="store_true",
        default=False,
        help="Do not write overlays for slices whose label file is empty.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry-point for visualization.

    Args:
        argv: Argument list (defaults to sys.argv if None).

    Raises:
        SystemExit: With code 1 on fatal error.
    """
    args = _parse_args(argv)

    logger.info("Images directory  : %s", args.images.resolve())
    logger.info("Labels directory  : %s", args.labels.resolve())
    logger.info("Output directory  : %s", args.output.resolve())
    logger.info("Limit             : %s", args.limit if args.limit else "all")
    logger.info("Outline thickness : %d px", args.thickness)
    logger.info("Fill alpha        : %.2f", args.alpha)
    logger.info("Skip empty labels : %s", args.skip_empty)

    try:
        overlay_directory(
            images_dir=args.images,
            labels_dir=args.labels,
            output_dir=args.output,
            limit=args.limit,
            thickness=args.thickness,
            alpha=args.alpha,
            skip_empty=args.skip_empty,
        )
    except FileNotFoundError as exc:
        logger.error("Fatal error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
