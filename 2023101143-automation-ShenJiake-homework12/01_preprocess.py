from __future__ import annotations

import shutil

import cv2

from hw13_common import DetectionResult, detect_corners, ensure_dirs, image_files, load_calibration_image, resize_for_page, write_detection_summary
from hw13_common import OUTPUT_DIR, PATTERN_SIZE


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
