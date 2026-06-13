"""
训练代码

功能：
- 读取 01_preprocess.py 生成的 X_train.npy / y_train.npy / X_test.npy / y_test.npy
- 构建 Dataset 与 DataLoader
- 构建 Skeleton Transformer
- 使用 CrossEntropyLoss 与 Adam 训练
- 保存 skeleton_transformer.pt、training_loss.png、training_accuracy.png、train_history.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import hw13_common as hw


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2: train Skeleton Transformer")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    hw.set_seed(args.seed)

    X_train = np.load(output_dir / "X_train.npy")
    y_train = np.load(output_dir / "y_train.npy")
    X_test = np.load(output_dir / "X_test.npy")
    y_test = np.load(output_dir / "y_test.npy")

    config = hw.ExperimentConfig(
        output_dir=str(output_dir),
        target_frames=int(X_train.shape[1]),
        input_dim=int(X_train.shape[2]),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )

    train_loader = DataLoader(hw.SkeletonDataset(X_train, y_train), batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(hw.SkeletonDataset(X_test, y_test), batch_size=config.batch_size, shuffle=False)

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
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = hw.run_epoch(model, train_loader, criterion, optimizer, device, True)
        with torch.no_grad():
            test_loss, test_acc = hw.run_epoch(model, test_loader, criterion, optimizer, device, False)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        print(
            f"[epoch {epoch:02d}/{config.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.3f}"
        )

    torch.save(model.state_dict(), output_dir / "skeleton_transformer.pt")
    hw.plot_history(history, output_dir)

    train_result = {
        "config": asdict(config),
        "device": str(device),
        "history": history,
        "model_path": str(output_dir / "skeleton_transformer.pt"),
    }
    (output_dir / "train_history.json").write_text(
        json.dumps(train_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("训练完成")
    print(f"模型文件: {(output_dir / 'skeleton_transformer.pt').resolve()}")
    print(f"最后一轮测试准确率: {history['test_acc'][-1]:.4f}")


if __name__ == "__main__":
    main()
