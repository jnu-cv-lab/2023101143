from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np

_COMMON_SPEC = importlib.util.spec_from_file_location("common", Path(__file__).with_name("03_common.py"))
common = importlib.util.module_from_spec(_COMMON_SPEC)
assert _COMMON_SPEC.loader is not None
sys.modules[_COMMON_SPEC.name] = common
_COMMON_SPEC.loader.exec_module(common)

OUTPUT_DIR = common.OUTPUT_DIR
PATTERN_SIZE = common.PATTERN_SIZE
SQUARE_SIZE_MM = common.SQUARE_SIZE_MM
detect_corners = common.detect_corners
ensure_dirs = common.ensure_dirs
image_files = common.image_files
load_calibration_image = common.load_calibration_image
object_points = common.object_points
resize_for_page = common.resize_for_page


UNDISTORT_TARGETS = ["1.jpg", "14.jpg", "4.jpg", "8.jpg"]


def calibrate() -> dict:
    ensure_dirs()
    objp = object_points(PATTERN_SIZE, SQUARE_SIZE_MM)
    objpoints: list[np.ndarray] = []
    imgpoints: list[np.ndarray] = []
    used_images: list[str] = []
    image_size: tuple[int, int] | None = None
    orientations: list[str] = []

    for image_path in image_files():
        try:
            image, orientation = load_calibration_image(image_path)
        except ValueError:
            continue
        ok, corners = detect_corners(image, PATTERN_SIZE)
        if not ok or corners is None:
            continue
        h, w = image.shape[:2]
        image_size = (w, h)
        objpoints.append(objp.copy())
        imgpoints.append(corners)
        used_images.append(str(image_path))
        orientations.append(orientation)

    if image_size is None or len(objpoints) < 3:
        raise RuntimeError("Not enough valid checkerboard images for calibration.")

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None,
    )

    per_image_errors: list[float] = []
    total_error_sq = 0.0
    total_points = 0
    for obj, img, rvec, tvec in zip(objpoints, imgpoints, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, camera_matrix, dist_coeffs)
        err = cv2.norm(img, projected, cv2.NORM_L2)
        n = len(projected)
        per_image_errors.append(float(err / np.sqrt(n)))
        total_error_sq += err * err
        total_points += n

    mean_reprojection_error = float(np.sqrt(total_error_sq / total_points))

    np.savez(
        OUTPUT_DIR / "calibration_result.npz",
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rvecs=np.array(rvecs, dtype=object),
        tvecs=np.array(tvecs, dtype=object),
        rms=rms,
        mean_reprojection_error=mean_reprojection_error,
        per_image_errors=np.array(per_image_errors),
        used_images=np.array(used_images),
        image_size=np.array(image_size),
        pattern_size=np.array(PATTERN_SIZE),
        square_size_mm=SQUARE_SIZE_MM,
    )

    undistorted_outputs: list[dict[str, str]] = []
    used_by_name = {Path(path).name: Path(path) for path in used_images}
    for target_name in UNDISTORT_TARGETS:
        target_path = used_by_name.get(target_name)
        if target_path is None:
            continue

        undistort_image, _ = load_calibration_image(target_path)
        h, w = undistort_image.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
        undistorted = cv2.undistort(undistort_image, camera_matrix, dist_coeffs, None, new_camera_matrix)

        x, y, rw, rh = roi
        cropped = undistorted[y : y + rh, x : x + rw] if rw > 0 and rh > 0 else undistorted
        stem = target_path.stem
        original_out = OUTPUT_DIR / "undistorted" / f"original_{stem}.jpg"
        undistorted_out = OUTPUT_DIR / "undistorted" / f"undistorted_{stem}.jpg"
        cropped_out = OUTPUT_DIR / "undistorted" / f"undistorted_{stem}_cropped.jpg"
        cv2.imwrite(str(original_out), resize_for_page(undistort_image))
        cv2.imwrite(str(undistorted_out), resize_for_page(undistorted))
        cv2.imwrite(str(cropped_out), resize_for_page(cropped))
        undistorted_outputs.append(
            {
                "image": target_name,
                "original": str(original_out),
                "undistorted": str(undistorted_out),
                "cropped": str(cropped_out),
            }
        )

    result = {
        "pattern_size": PATTERN_SIZE,
        "square_size_mm": SQUARE_SIZE_MM,
        "image_size": image_size,
        "num_images_total": len(image_files()),
        "num_images_used": len(used_images),
        "used_images": used_images,
        "orientation_normalization": orientations,
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": dist_coeffs.ravel().tolist(),
        "rms_error": float(rms),
        "mean_reprojection_error_px": mean_reprojection_error,
        "per_image_reprojection_error_px": per_image_errors,
        "undistorted_outputs": undistorted_outputs,
    }
    (OUTPUT_DIR / "calibration_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def print_result(result: dict) -> None:
    k = np.array(result["camera_matrix"])
    d = np.array(result["distortion_coefficients"])
    print("Camera calibration finished.")
    print(f"Images used: {result['num_images_used']}/{result['num_images_total']}")
    print(f"Image size: {result['image_size'][0]} x {result['image_size'][1]}")
    print("Camera matrix K:")
    print(k)
    print("Distortion coefficients [k1, k2, p1, p2, k3]:")
    print(d)
    print(f"RMS error: {result['rms_error']:.4f}")
    print(f"Mean reprojection error: {result['mean_reprojection_error_px']:.4f} px")
    print(f"Result JSON: {OUTPUT_DIR / 'calibration_result.json'}")


def main() -> None:
    result = calibrate()
    print_result(result)


if __name__ == "__main__":
    main()
