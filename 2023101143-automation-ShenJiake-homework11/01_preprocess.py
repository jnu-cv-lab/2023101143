"""
预处理代码

功能：
- 遍历 archive/ 下 6 个类别文件夹
- 使用 OpenCV 读取视频
- 使用 MediaPipe Pose 提取每帧 33 个关键点
- 重采样为 [30, 132]
- 做髋部中心和肩宽归一化
- 按 test_size=0.2 划分训练/测试集
- 保存 X_train.npy / y_train.npy / X_test.npy / y_test.npy / label_map.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

import hw13_common as hw


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1: preprocess videos to skeleton npy files")
    parser.add_argument("--data-dir", default="archive")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--target-frames", type=int, default=30)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-videos-per-class", type=int, default=None)
    parser.add_argument("--pose-model", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = hw.ExperimentConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        target_frames=args.target_frames,
        test_size=args.test_size,
        seed=args.seed,
        max_videos_per_class=args.max_videos_per_class,
        pose_model=args.pose_model,
    )
    hw.set_seed(config.seed)

    X, y, paths, feature_source = hw.preprocess_dataset(config)
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=config.test_size,
        random_state=config.seed,
        stratify=y,
    )

    np.save(output_dir / "X_train.npy", X[train_idx])
    np.save(output_dir / "y_train.npy", y[train_idx])
    np.save(output_dir / "X_test.npy", X[test_idx])
    np.save(output_dir / "y_test.npy", y[test_idx])

    label_map = {str(i): name for i, name in enumerate(hw.LABELS)}
    (output_dir / "label_map.json").write_text(json.dumps(label_map, ensure_ascii=False, indent=2), encoding="utf-8")

    metadata = {
        "feature_source": feature_source,
        "target_frames": config.target_frames,
        "input_dim": config.input_dim,
        "test_size": config.test_size,
        "seed": config.seed,
        "all_paths": paths,
        "train_indices": train_idx.tolist(),
        "test_indices": test_idx.tolist(),
        "train_paths": [paths[i] for i in train_idx],
        "test_paths": [paths[i] for i in test_idx],
        "class_counts": {hw.LABELS[i]: int((y == i).sum()) for i in range(len(hw.LABELS))},
    }
    (output_dir / "preprocess_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("预处理完成")
    print(f"feature_source: {feature_source}")
    print(f"X_train: {X[train_idx].shape}")
    print(f"y_train: {y[train_idx].shape}")
    print(f"X_test: {X[test_idx].shape}")
    print(f"y_test: {y[test_idx].shape}")
    print(f"输出目录: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
