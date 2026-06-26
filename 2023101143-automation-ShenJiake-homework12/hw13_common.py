from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


CHESSBOARD_DIR = Path("chessboard")
OUTPUT_DIR = Path("outputs")

PATTERN_SIZE = (9, 6)  # inner corners: columns, rows
SQUARE_SIZE_MM = 25.0


@dataclass
class DetectionResult:
    image_path: str
    success: bool
    image_size: tuple[int, int] | None
    corners_path: str | None = None
    message: str = ""


def ensure_dirs() -> None:
    (OUTPUT_DIR / "corners").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "undistorted").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "samples").mkdir(parents=True, exist_ok=True)


def image_files() -> list[Path]:
    files = sorted(
        CHESSBOARD_DIR.glob("*.jpg"),
        key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem,
    )
    if not files:
        raise FileNotFoundError(f"No .jpg calibration images found in {CHESSBOARD_DIR}")
    return files


def object_points(pattern_size: tuple[int, int] = PATTERN_SIZE, square_size: float = SQUARE_SIZE_MM) -> np.ndarray:
    cols, rows = pattern_size
    points = np.zeros((rows * cols, 3), np.float32)
    points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    points *= float(square_size)
    return points


def resize_for_page(image: np.ndarray, max_side: int = 1200) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(max_side / max(h, w), 1.0)
    if scale >= 1.0:
        return image
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def load_calibration_image(path: Path) -> tuple[np.ndarray, str]:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"OpenCV failed to read image: {path}")
    h, w = image.shape[:2]
    if w > h:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), "rotated_90_clockwise"
    return image, "original"


def detect_corners(image: np.ndarray, pattern_size: tuple[int, int] = PATTERN_SIZE) -> tuple[bool, np.ndarray | None]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sb_flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    ok, corners = cv2.findChessboardCornersSB(gray, pattern_size, sb_flags)
    if ok and corners is not None:
        return True, corners.astype(np.float32)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, pattern_size, flags)

    if not ok:
        # A contrast-normalized fallback helps on high-resolution phone photos.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)
        ok, corners = cv2.findChessboardCorners(gray_eq, pattern_size, flags)
        gray = gray_eq

    if not ok or corners is None:
        return False, None

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        50,
        0.001,
    )
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, refined


def write_detection_summary(results: list[DetectionResult], path: Path = OUTPUT_DIR / "detection_summary.json") -> None:
    ensure_dirs()
    path.write_text(
        json.dumps([asdict(item) for item in results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_detection_summary(path: Path = OUTPUT_DIR / "detection_summary.json") -> list[DetectionResult]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [DetectionResult(**item) for item in data]
