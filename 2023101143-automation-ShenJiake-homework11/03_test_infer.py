"""
测试与推理代码

功能：
- 加载 X_test.npy / y_test.npy 和 skeleton_transformer.pt
- 输出测试集 accuracy、confusion matrix、classification report
- 对一个测试样本做单样本推理，输出 6 类 logits、softmax 概率、预测类别和置信度
- 保存 experiment_summary.json 和 confusion_matrix.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import hw13_common as hw


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 3: test and infer")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sample-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    X_train = np.load(output_dir / "X_train.npy")
    y_train = np.load(output_dir / "y_train.npy")
    X_test = np.load(output_dir / "X_test.npy")
    y_test = np.load(output_dir / "y_test.npy")

    metadata_path = output_dir / "preprocess_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    train_history_path = output_dir / "train_history.json"
    train_history = json.loads(train_history_path.read_text(encoding="utf-8")) if train_history_path.exists() else {}
    previous_summary_path = output_dir / "experiment_summary.json"
    previous_summary = (
        json.loads(previous_summary_path.read_text(encoding="utf-8"))
        if previous_summary_path.exists()
        else {}
    )

    config_dict = train_history.get("config", previous_summary.get("config", {}))
    config = hw.ExperimentConfig(
        output_dir=str(output_dir),
        target_frames=int(X_test.shape[1]),
        input_dim=int(X_test.shape[2]),
        d_model=int(config_dict.get("d_model", 128)),
        nhead=int(config_dict.get("nhead", 4)),
        num_layers=int(config_dict.get("num_layers", 2)),
        dim_feedforward=int(config_dict.get("dim_feedforward", 256)),
        dropout=float(config_dict.get("dropout", 0.1)),
        batch_size=args.batch_size,
        epochs=int(config_dict.get("epochs", len(train_history.get("history", {}).get("train_loss", [])) or 0)),
        lr=float(config_dict.get("lr", 1e-3)),
        test_size=float(metadata.get("test_size", 0.2)),
        seed=int(metadata.get("seed", 42)),
    )

    device = hw.get_device()
    model = hw.SkeletonTransformer(
        input_dim=config.input_dim,
        num_classes=len(hw.LABELS),
        target_frames=config.target_frames,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
    ).to(device)
    model.load_state_dict(torch.load(output_dir / "skeleton_transformer.pt", map_location=device))

    test_loader = DataLoader(hw.SkeletonDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False)
    test_acc, cm, report, _, _ = hw.evaluate(model, test_loader, device)
    hw.plot_confusion_matrix(cm, output_dir)

    sample_index = max(0, min(args.sample_index, len(X_test) - 1))
    pred, confidence, logits, probs = hw.predict_one(model, X_test[sample_index], device)
    test_paths = metadata.get("test_paths", [])
    video = test_paths[sample_index] if sample_index < len(test_paths) else previous_summary.get("inference", {}).get("video", f"X_test[{sample_index}]")

    inference = {
        "video": video,
        "true_class": hw.LABELS[int(y_test[sample_index])],
        "predicted_class": hw.LABELS[pred],
        "confidence": confidence,
        "logits": {hw.LABELS[i]: float(v) for i, v in enumerate(logits)},
        "probabilities": {hw.LABELS[i]: float(p) for i, p in enumerate(probs)},
    }

    summary = {
        "config": config.__dict__,
        "device": str(device),
        "feature_source": metadata.get("feature_source", previous_summary.get("feature_source", "unknown")),
        "num_samples": int(len(X_train) + len(X_test)),
        "class_counts": metadata.get("class_counts", previous_summary.get("class_counts", {})),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "history": train_history.get("history", previous_summary.get("history", {})),
        "test_accuracy": test_acc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "inference": inference,
    }
    (output_dir / "experiment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"测试准确率: {test_acc:.4f}")
    print("混淆矩阵:")
    print(cm)
    print("分类报告:")
    print(report)
    print(f"测试视频: {inference['video']}")
    print(f"真实类别: {inference['true_class']}")
    print(f"预测类别: {inference['predicted_class']}")
    print(f"置信度: {inference['confidence']:.4f}")
    print("Logits:")
    for name, value in inference["logits"].items():
        print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    main()
