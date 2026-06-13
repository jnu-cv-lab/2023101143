"""
第13课作业：羽毛球击球动作识别

功能：
1. 遍历 archive/ 中 6 类羽毛球视频。
2. 优先使用 MediaPipe Pose 提取 33 个关键点；若本机未安装 MediaPipe，则使用
   OpenCV 运动/轮廓代理特征生成同样形状的 [T, 132] 序列，保证课堂实验可运行。
3. 训练轻量 Skeleton Transformer，输出测试集准确率、混淆矩阵、分类报告和单样本推理。
4. 保存模型、曲线图、npy 数据和实验摘要 JSON 到 outputs/。

"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


LABELS = [
    "forehand_drive",
    "forehand_lift",
    "forehand_net_shot",
    "forehand_clear",
    "backhand_drive",
    "backhand_net_shot",
]

LABEL_CN = {
    "forehand_drive": "正手平抽 / 正手驱动球",
    "forehand_lift": "正手挑球",
    "forehand_net_shot": "正手网前球",
    "forehand_clear": "正手高远球",
    "backhand_drive": "反手平抽 / 反手驱动球",
    "backhand_net_shot": "反手网前球",
}

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
DEFAULT_POSE_MODEL = Path("models") / "pose_landmarker_lite.task"


@dataclass
class ExperimentConfig:
    data_dir: str = "archive"
    output_dir: str = "outputs"
    target_frames: int = 30
    input_dim: int = 132
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1
    batch_size: int = 16
    epochs: int = 20
    lr: float = 1e-3
    test_size: float = 0.2
    seed: int = 42
    max_videos_per_class: Optional[int] = None
    pose_model: Optional[str] = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MediaPipePoseExtractor:
    def __init__(self, backend: str, detector, mp_module=None) -> None:
        self.backend = backend
        self.detector = detector
        self.mp = mp_module

    def extract_frame(self, rgb: np.ndarray) -> Optional[np.ndarray]:
        if self.backend == "solutions":
            result = self.detector.process(rgb)
            if result.pose_landmarks is None:
                return None
            pts = [
                [lm.x, lm.y, lm.z, lm.visibility]
                for lm in result.pose_landmarks.landmark
            ]
            return np.asarray(pts, dtype=np.float32)

        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(image)
        if not result.pose_landmarks:
            return None
        pts = []
        for lm in result.pose_landmarks[0]:
            visibility = getattr(lm, "visibility", getattr(lm, "presence", 1.0))
            pts.append([lm.x, lm.y, lm.z, visibility])
        return np.asarray(pts, dtype=np.float32)

    def close(self) -> None:
        close = getattr(self.detector, "close", None)
        if close is not None:
            close()


def try_create_mediapipe_pose(pose_model: Optional[str] = None) -> Optional[MediaPipePoseExtractor]:
    try:
        import mediapipe as mp  # type: ignore

        if hasattr(mp, "solutions"):
            detector = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            return MediaPipePoseExtractor("solutions", detector, mp)

        if not pose_model and DEFAULT_POSE_MODEL.exists():
            pose_model = str(DEFAULT_POSE_MODEL)
            print(f"[mediapipe] 自动使用默认模型：{pose_model}")

        if not pose_model:
            print(
                "[mediapipe] 已安装新版 mediapipe tasks，但未提供 --pose-model，"
                "将回退到 OpenCV 代理特征。"
            )
            return None

        from mediapipe.tasks import python as mp_python  # type: ignore
        from mediapipe.tasks.python import vision  # type: ignore

        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(pose_model)),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
        detector = vision.PoseLandmarker.create_from_options(options)
        return MediaPipePoseExtractor("tasks_pose_landmarker", detector, mp)
    except Exception as exc:
        print(f"[mediapipe] 初始化失败，将回退到 OpenCV 代理特征：{exc}")
        return None


def list_videos(data_dir: Path, max_videos_per_class: Optional[int]) -> Tuple[List[Path], List[int]]:
    paths: List[Path] = []
    labels: List[int] = []
    for label_idx, class_name in enumerate(LABELS):
        class_dir = data_dir / class_name
        files = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS)
        if max_videos_per_class is not None:
            files = files[:max_videos_per_class]
        paths.extend(files)
        labels.extend([label_idx] * len(files))
    return paths, labels


def read_video_frames(video_path: Path, target_frames: int) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频：{video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        indices = list(range(target_frames))
    else:
        indices = np.linspace(0, max(total - 1, 0), target_frames).astype(int).tolist()

    frames: List[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise ValueError(f"视频没有可读帧：{video_path}")
    while len(frames) < target_frames:
        frames.append(frames[-1].copy())
    return frames[:target_frames]


def normalize_pose_sequence(seq: np.ndarray) -> np.ndarray:
    """对 [T, 33, 4] 做中心和尺度归一化，再展平成 [T, 132]。"""
    coords = seq[:, :, :3].copy()
    vis = seq[:, :, 3:4].copy()

    # MediaPipe 的左右肩/髋索引。代理特征也使用同一索引，若尺度异常则回退到全局方差。
    shoulder_center = (coords[:, 11, :2] + coords[:, 12, :2]) / 2
    hip_center = (coords[:, 23, :2] + coords[:, 24, :2]) / 2
    origin = np.concatenate([hip_center, np.zeros((seq.shape[0], 1), dtype=np.float32)], axis=1)
    shoulder_width = np.linalg.norm(coords[:, 11, :2] - coords[:, 12, :2], axis=1)
    fallback_scale = np.std(coords[:, :, :2].reshape(seq.shape[0], -1), axis=1)
    scale = np.where(shoulder_width > 1e-4, shoulder_width, fallback_scale)
    scale = np.maximum(scale, 1e-4).reshape(-1, 1, 1)

    coords = (coords - origin[:, None, :]) / scale
    normalized = np.concatenate([coords, vis], axis=2)
    return normalized.reshape(seq.shape[0], -1).astype(np.float32)


def extract_with_mediapipe(frames: Sequence[np.ndarray], pose: MediaPipePoseExtractor) -> Optional[np.ndarray]:
    landmarks: List[np.ndarray] = []
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pts = pose.extract_frame(rgb)
        if pts is None:
            landmarks.append(np.zeros((33, 4), dtype=np.float32))
            continue
        landmarks.append(pts)

    seq = np.stack(landmarks)
    return normalize_pose_sequence(seq)


def _weighted_center(mask: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> Tuple[float, float, float]:
    roi = mask[y0:y1, x0:x1]
    if roi.size == 0:
        return (x0 / mask.shape[1], y0 / mask.shape[0], 0.0)
    ys, xs = np.nonzero(roi)
    if len(xs) == 0:
        return ((x0 + x1) / 2 / mask.shape[1], (y0 + y1) / 2 / mask.shape[0], 0.0)
    return ((x0 + float(xs.mean())) / mask.shape[1], (y0 + float(ys.mean())) / mask.shape[0], min(1.0, len(xs) / roi.size * 8))


def extract_with_opencv_proxy(frames: Sequence[np.ndarray]) -> np.ndarray:
    """MediaPipe 不可用时的可运行替代：生成 33 个运动/轮廓代理点。"""
    seq: List[np.ndarray] = []
    prev_gray: Optional[np.ndarray] = None
    for frame in frames:
        resized = cv2.resize(frame, (192, 192))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if prev_gray is None:
            motion = cv2.Canny(gray, 60, 140)
        else:
            diff = cv2.absdiff(gray, prev_gray)
            _, motion = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
            motion = cv2.morphologyEx(motion, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        prev_gray = gray

        # 11 x 3 网格 = 33 个代理关键点，每点 x/y/局部强度/visibility。
        pts = []
        h, w = motion.shape
        for gy in range(11):
            for gx in range(3):
                x0 = int(gx * w / 3)
                x1 = int((gx + 1) * w / 3)
                y0 = int(gy * h / 11)
                y1 = int((gy + 1) * h / 11)
                x, y, score = _weighted_center(motion, x0, y0, x1, y1)
                z = float(gray[y0:y1, x0:x1].mean() / 255.0) if y1 > y0 and x1 > x0 else 0.0
                pts.append([x, y, z, score])
        seq.append(np.asarray(pts, dtype=np.float32))
    return normalize_pose_sequence(np.stack(seq))


def extract_skeleton_sequence(video_path: Path, target_frames: int, pose=None) -> np.ndarray:
    frames = read_video_frames(video_path, target_frames)
    if pose is not None:
        seq = extract_with_mediapipe(frames, pose)
        if seq is not None:
            return seq
    return extract_with_opencv_proxy(frames)


def preprocess_dataset(config: ExperimentConfig) -> Tuple[np.ndarray, np.ndarray, List[str], str]:
    data_dir = Path(config.data_dir)
    video_paths, labels = list_videos(data_dir, config.max_videos_per_class)
    if not video_paths:
        raise FileNotFoundError(f"未在 {data_dir} 找到视频文件")

    pose = try_create_mediapipe_pose(config.pose_model)
    feature_source = f"mediapipe_{pose.backend}" if pose is not None else "opencv_proxy_no_mediapipe"

    features: List[np.ndarray] = []
    kept_labels: List[int] = []
    kept_paths: List[str] = []
    for i, (path, label) in enumerate(zip(video_paths, labels), start=1):
        try:
            features.append(extract_skeleton_sequence(path, config.target_frames, pose))
            kept_labels.append(label)
            kept_paths.append(str(path))
            print(f"[preprocess] {i:04d}/{len(video_paths)} OK {path}")
        except Exception as exc:
            print(f"[preprocess] 跳过 {path}: {exc}")

    if pose is not None:
        pose.close()
    if not features:
        raise RuntimeError("没有成功提取任何视频特征")

    X = np.stack(features).astype(np.float32)
    y = np.asarray(kept_labels, dtype=np.int64)
    return X, y, kept_paths, feature_source


class SkeletonDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class SkeletonTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        target_frames: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.position = nn.Parameter(torch.zeros(1, target_frames, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x) + self.position[:, : x.size(1), :]
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)


def run_epoch(model, loader, criterion, optimizer, device: torch.device, train: bool) -> Tuple[float, float]:
    model.train(train)
    losses: List[float] = []
    preds: List[int] = []
    labels: List[int] = []
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        if train:
            optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        if train:
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
        preds.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())
        labels.extend(y_batch.detach().cpu().numpy().tolist())
    return float(np.mean(losses)), float(accuracy_score(labels, preds))


def evaluate(model, loader, device: torch.device) -> Tuple[float, np.ndarray, str, List[int], List[int]]:
    model.eval()
    preds: List[int] = []
    labels: List[int] = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch.to(device))
            preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            labels.extend(y_batch.numpy().tolist())
    acc = float(accuracy_score(labels, preds))
    cm = confusion_matrix(labels, preds, labels=list(range(len(LABELS))))
    report = classification_report(
        labels,
        preds,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        zero_division=0,
    )
    return acc, cm, report, labels, preds


def plot_history(history: Dict[str, List[float]], output_dir: Path) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(history["train_loss"], label="train loss")
    plt.plot(history["test_loss"], label="test loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_loss.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(history["train_acc"], label="train acc")
    plt.plot(history["test_acc"], label="test acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_accuracy.png", dpi=160)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, output_dir: Path) -> None:
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(LABELS))
    plt.xticks(ticks, LABELS, rotation=35, ha="right", fontsize=8)
    plt.yticks(ticks, LABELS, fontsize=8)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close()


def predict_one(model, sample: np.ndarray, device: torch.device) -> Tuple[int, float, List[float], List[float]]:
    model.eval()
    with torch.no_grad():
        x = torch.tensor(sample[None, ...], dtype=torch.float32).to(device)
        logits = model(x).cpu().numpy()[0]
        prob = torch.softmax(torch.tensor(logits), dim=0).numpy()
    pred = int(prob.argmax())
    return pred, float(prob[pred]), logits.tolist(), prob.tolist()


def train_and_evaluate(
    config: ExperimentConfig,
    X: np.ndarray,
    y: np.ndarray,
    paths: Sequence[str],
    feature_source: str,
) -> Dict:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    label_map = {str(i): name for i, name in enumerate(LABELS)}
    (output_dir / "label_map.json").write_text(json.dumps(label_map, ensure_ascii=False, indent=2), encoding="utf-8")

    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=config.test_size,
        random_state=config.seed,
        stratify=y,
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "X_test.npy", X_test)
    np.save(output_dir / "y_test.npy", y_test)

    train_loader = DataLoader(SkeletonDataset(X_train, y_train), batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(SkeletonDataset(X_test, y_test), batch_size=config.batch_size, shuffle=False)

    device = get_device()
    model = SkeletonTransformer(
        input_dim=config.input_dim,
        num_classes=len(LABELS),
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
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, True)
        with torch.no_grad():
            test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer, device, False)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        print(
            f"[epoch {epoch:02d}/{config.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.3f}"
        )

    test_acc, cm, report, true_labels, pred_labels = evaluate(model, test_loader, device)
    torch.save(model.state_dict(), output_dir / "skeleton_transformer.pt")
    plot_history(history, output_dir)
    plot_confusion_matrix(cm, output_dir)

    sample_idx = int(test_idx[0])
    pred, confidence, logits, probs = predict_one(model, X[sample_idx], device)
    inference = {
        "video": paths[sample_idx],
        "true_class": LABELS[int(y[sample_idx])],
        "predicted_class": LABELS[pred],
        "confidence": confidence,
        "logits": {LABELS[i]: float(v) for i, v in enumerate(logits)},
        "probabilities": {LABELS[i]: float(p) for i, p in enumerate(probs)},
    }

    summary = {
        "config": asdict(config),
        "device": str(device),
        "feature_source": feature_source,
        "num_samples": int(len(y)),
        "class_counts": {LABELS[i]: int((y == i).sum()) for i in range(len(LABELS))},
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "history": history,
        "test_accuracy": test_acc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "inference": inference,
    }
    (output_dir / "experiment_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="Skeleton Transformer Badminton Action Recognition")
    parser.add_argument("--data-dir", default="archive")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--target-frames", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-videos-per-class", type=int, default=None)
    parser.add_argument(
        "--pose-model",
        default=None,
        help="新版 mediapipe tasks 需要的 pose_landmarker_lite.task 模型路径。",
    )
    args = parser.parse_args()
    return ExperimentConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        target_frames=args.target_frames,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        test_size=args.test_size,
        seed=args.seed,
        max_videos_per_class=args.max_videos_per_class,
        pose_model=args.pose_model,
    )


def main() -> None:
    config = parse_args()
    set_seed(config.seed)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    X, y, paths, feature_source = preprocess_dataset(config)
    summary = train_and_evaluate(config, X, y, paths, feature_source)
    print("\n========== 实验完成 ==========")
    print(f"特征来源: {summary['feature_source']}")
    print(f"测试准确率: {summary['test_accuracy']:.4f}")
    print(f"预测样本: {summary['inference']['video']}")
    print(
        f"Predicted class: {summary['inference']['predicted_class']} "
        f"Confidence: {summary['inference']['confidence']:.4f}"
    )
    print("Logits:")
    for name, value in summary["inference"]["logits"].items():
        print(f"  {name}: {value:.4f}")
    print(f"输出目录: {Path(config.output_dir).resolve()}")


if __name__ == "__main__":
    main()
