from __future__ import annotations

import shutil
import importlib.util
import sys
from pathlib import Path

import cv2

_COMMON_SPEC = importlib.util.spec_from_file_location("common", Path(__file__).with_name("03_common.py"))
common = importlib.util.module_from_spec(_COMMON_SPEC)
assert _COMMON_SPEC.loader is not None
sys.modules[_COMMON_SPEC.name] = common
_COMMON_SPEC.loader.exec_module(common)

DetectionResult = common.DetectionResult
OUTPUT_DIR = common.OUTPUT_DIR
PATTERN_SIZE = common.PATTERN_SIZE
detect_corners = common.detect_corners
ensure_dirs = common.ensure_dirs
image_files = common.image_files
load_calibration_image = common.load_calibration_image
resize_for_page = common.resize_for_page
write_detection_summary = common.write_detection_summary


def main() -> None:
    ensure_dirs()

    results: list[DetectionResult] = []
    sample_names = {"1.jpg", "4.jpg", "8.jpg", "14.jpg"}

    for image_path in image_files():
        try:
            image, orientation = load_calibration_image(image_path)
        except ValueError as exc:
            results.append(DetectionResult(str(image_path), False, None, message=str(exc)))
            continue

        ok, corners = detect_corners(image, PATTERN_SIZE)
        h, w = image.shape[:2]
        out_path = OUTPUT_DIR / "corners" / f"{image_path.stem}_corners.jpg"

        if ok and corners is not None:
            annotated = image.copy()
            cv2.drawChessboardCorners(annotated, PATTERN_SIZE, corners, ok)
            cv2.imwrite(str(out_path), resize_for_page(annotated))
            results.append(DetectionResult(str(image_path), True, (w, h), str(out_path), orientation))
        else:
            results.append(DetectionResult(str(image_path), False, (w, h), message="Checkerboard corners not found."))

        if image_path.name in sample_names:
            sample_out = OUTPUT_DIR / "samples" / image_path.name
            cv2.imwrite(str(sample_out), resize_for_page(image))

    write_detection_summary(results)
    total = len(results)
    success = sum(item.success for item in results)
    print(f"Detected checkerboard corners in {success}/{total} images.")
    print(f"Annotated corner images: {OUTPUT_DIR / 'corners'}")
    print(f"Detection summary: {OUTPUT_DIR / 'detection_summary.json'}")


if __name__ == "__main__":
    main()
