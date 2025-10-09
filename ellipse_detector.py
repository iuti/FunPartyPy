import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def detect_smallest_ellipse(image: np.ndarray, min_contour_points: int = 20, min_axis_length: float = 0.0):
    """Return the smallest ellipse fitted to contours in the image.

    Parameters
    ----------
    image : np.ndarray
        BGR image as loaded by cv2.imread.
    min_contour_points : int, optional
        Minimum number of points required for cv2.fitEllipse.

    Returns
    -------
    tuple | None
        (ellipse, area) where ellipse is ((cx, cy), (major_axis, minor_axis), angle).
        Returns None if no valid ellipse is found.
    """

    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)

    # Try both external and list retrieval to maximize candidates
    contours_external, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_all, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    contours = contours_external + contours_all

    smallest = None
    min_area = math.inf
    candidate_count = 0

    for contour in contours:
        if len(contour) < max(5, min_contour_points):
            continue

        ellipse = cv2.fitEllipse(contour)
        (_, _), (major_axis, minor_axis), _ = ellipse

        if major_axis <= 0 or minor_axis <= 0:
            continue

        if min_axis_length > 0 and (major_axis < min_axis_length or minor_axis < min_axis_length):
            continue

        area = math.pi * (major_axis / 2.0) * (minor_axis / 2.0)

        candidate_count += 1

        if area < min_area:
            min_area = area
            smallest = ellipse

    if smallest is None:
        return None

    return smallest, min_area, candidate_count


def annotate_image(image: np.ndarray, ellipse, color=(0, 255, 0)):
    annotated = image.copy()
    cv2.ellipse(annotated, ellipse, color, 2)
    (cx, cy), (major, minor), angle = ellipse
    label = f"center=({cx:.1f}, {cy:.1f}), axes=({major:.1f}, {minor:.1f}), angle={angle:.1f}"
    cv2.putText(
        annotated,
        label,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        lineType=cv2.LINE_AA,
    )
    return annotated


def main(argv=None):
    parser = argparse.ArgumentParser(description="Detect the smallest ellipse in calibration images.")
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=Path(__file__).parent / "calibImg",
        type=Path,
        help="Directory containing calibration images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "calibImg" / "annotated",
        help="Directory to save annotated images.",
    )
    parser.add_argument(
        "--min-contour-points",
        type=int,
        default=20,
        help="Minimum contour points required to fit an ellipse.",
    )
    parser.add_argument(
        "--min-axis-length",
        type=float,
        default=50.0,
        help="Discard fitted ellipses whose major or minor axis is shorter than this length (pixels).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "calibration_config.json",
        help="Path to calibration configuration JSON file to update.",
    )
    args = parser.parse_args(argv)

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        [
            path
            for path in input_dir.iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        ]
    )

    if not image_paths:
        print(f"No images found in {input_dir}")
        return 1

    global_smallest = None
    global_smallest_area = math.inf
    global_smallest_shape = None
    global_smallest_candidates = 0

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue

        result = detect_smallest_ellipse(image, args.min_contour_points, args.min_axis_length)
        if result is None:
            print(f"No ellipse detected in {image_path.name}")
            continue

        ellipse, area, candidate_count = result
        print(
            f"{image_path.name}: center=({ellipse[0][0]:.2f}, {ellipse[0][1]:.2f}), "
            f"axes=({ellipse[1][0]:.2f}, {ellipse[1][1]:.2f}), angle={ellipse[2]:.2f}, area={area:.2f}"
        )
        print(f"  candidates evaluated: {candidate_count}")

        annotated = annotate_image(image, ellipse)
        output_path = output_dir / f"{image_path.stem}_smallest_ellipse.png"
        cv2.imwrite(str(output_path), annotated)

        if area < global_smallest_area:
            global_smallest_area = area
            global_smallest = (image_path, ellipse)
            global_smallest_shape = image.shape
            global_smallest_candidates = candidate_count

    if global_smallest is not None:
        image_path, ellipse = global_smallest
        print(
            f"\nSmallest ellipse overall: {image_path.name} -> center=({ellipse[0][0]:.2f}, {ellipse[0][1]:.2f}), "
            f"axes=({ellipse[1][0]:.2f}, {ellipse[1][1]:.2f}), angle={ellipse[2]:.2f}, area={global_smallest_area:.2f}"
        )

        config_path: Path = args.config
        config_data = {}
        if config_path.exists():
            try:
                with config_path.open("r", encoding="utf-8") as f:
                    config_data = json.load(f)
            except json.JSONDecodeError as exc:
                print(f"Failed to parse existing config ({config_path}): {exc}. Overwriting.")

        (cx, cy), (major_axis, minor_axis), angle = ellipse
        frame_height, frame_width = global_smallest_shape[:2] if global_smallest_shape else (None, None)

        config_data["timestamp"] = time.time()
        config_data.setdefault("metrics", {})
        config_data.setdefault("source", {})

        config_data["ellipse"] = {
            "center_x": float(cx),
            "center_y": float(cy),
            "major_axis": float(major_axis),
            "minor_axis": float(minor_axis),
            "angle_deg": float(angle),
        }

        if frame_width is not None and frame_height is not None:
            config_data["frame"] = {
                "width": int(frame_width),
                "height": int(frame_height),
            }

        config_data["metrics"]["ellipse_candidates"] = global_smallest_candidates

        config_data["source"] = {
            "type": "image",
            "path": str(image_path.resolve()),
        }

        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)

        print(f"Updated calibration config: {config_path}")
    else:
        print("No ellipses detected in any images.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
