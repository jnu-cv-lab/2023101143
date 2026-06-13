"""
骨架可视化

功能：
- 读取 1 个视频
- 使用 MediaPipe Pose 提取若干帧骨架
- 保存 1-2 张骨架关键点可视化图片
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

import hw13_common as hw


def parse_args():
    parser = argparse.ArgumentParser(description="Optional: visualize MediaPipe skeleton")
    parser.add_argument("--video", default="archive/forehand_drive/001.mp4")
    parser.add_argument("--output-dir", default="outputs/skeleton_visualization")
    parser.add_argument("--pose-model", default=None)
    parser.add_argument("--frames", type=int, default=2)
    return parser.parse_args()


def draw_skeleton(frame: np.ndarray, landmarks: np.ndarray, output_path: Path) -> None:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    # MediaPipe Pose 常用连接边，覆盖躯干、手臂和腿部。
    edges = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 23), (12, 24), (23, 24),
        (23, 25), (25, 27), (24, 26), (26, 28),
        (27, 29), (29, 31), (28, 30), (30, 32),
    ]

    plt.figure(figsize=(6, 6))
    plt.imshow(rgb)
    for a, b in edges:
        xa, ya, va = landmarks[a, 0] * w, landmarks[a, 1] * h, landmarks[a, 3]
        xb, yb, vb = landmarks[b, 0] * w, landmarks[b, 1] * h, landmarks[b, 3]
        if va > 0.2 and vb > 0.2:
            plt.plot([xa, xb], [ya, yb], color="#00b4d8", linewidth=2)
    visible = landmarks[:, 3] > 0.2
    plt.scatter(landmarks[visible, 0] * w, landmarks[visible, 1] * h, s=20, c="#ffb703")
    plt.axis("off")
    plt.tight_layout(pad=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160, bbox_inches="tight", pad_inches=0)
    plt.close()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    video_path = Path(args.video)
    pose = hw.try_create_mediapipe_pose(args.pose_model)
    if pose is None:
        raise RuntimeError("MediaPipe Pose 初始化失败，请确认 models/pose_landmarker_lite.task 存在。")

    frames = hw.read_video_frames(video_path, max(args.frames, 1))
    saved = 0
    for i, frame in enumerate(frames):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks = pose.extract_frame(rgb)
        if landmarks is None:
            continue
        draw_skeleton(frame, landmarks, output_dir / f"skeleton_frame_{i + 1}.png")
        saved += 1
        if saved >= args.frames:
            break
    pose.close()

    print(f"骨架可视化完成，保存 {saved} 张图片到 {output_dir.resolve()}")


if __name__ == "__main__":
    main()
