from __future__ import annotations

import csv
import json
import random
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from build_report import create_report


SEED = 42
EPOCHS = 5
BATCH_SIZE = 64
VALIDATION_SIZE = 10_000
OUTPUT_DIR = Path("outputs")
DATA_DIR = Path("data")
CLASS_NAMES = [str(i) for i in range(10)]


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_accuracy: float
    validation_loss: float
    validation_accuracy: float


@dataclass
class ExperimentResult:
    optimizer: str
    learning_rate: float
    test_loss: float
    test_accuracy: float
    history: list[EpochMetrics]


class SimpleCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def environment_check(log: Callable[[str], None]) -> torch.device:
    log("=== Environment Check ===")
    log(f"Python: {sys.version.split()[0]}")
    log(f"PyTorch: {torch.__version__}")
    log(f"CUDA available: {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")
    tensor_a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    tensor_b = torch.tensor([[2.0, 0.0], [1.0, 2.0]])
    log(f"Simple tensor operation result: {(tensor_a @ tensor_b).tolist()}")
    log("")
    return device


def get_dataloaders() -> tuple[DataLoader, DataLoader, DataLoader, datasets.MNIST]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    full_train = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)

    train_size = len(full_train) - VALIDATION_SIZE
    generator = torch.Generator().manual_seed(SEED)
    train_dataset, validation_dataset = random_split(
        full_train, [train_size, VALIDATION_SIZE], generator=generator
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, validation_loader, test_loader, full_train


def denormalize(image: torch.Tensor) -> np.ndarray:
    image = image.squeeze().detach().cpu().numpy()
    image = image * 0.3081 + 0.1307
    return np.clip(image, 0.0, 1.0)


def first_index_per_class(dataset: datasets.MNIST) -> list[int]:
    indices: dict[int, int] = {}
    for index in range(len(dataset)):
        _, label = dataset[index]
        label_value = int(label)
        if label_value not in indices:
            indices[label_value] = index
        if len(indices) == len(CLASS_NAMES):
            break
    return [indices[label] for label in range(len(CLASS_NAMES))]


def save_sample_images(dataset: datasets.MNIST, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for ax, index in zip(axes.ravel(), first_index_per_class(dataset)):
        image, label = dataset[index]
        ax.imshow(denormalize(image), cmap="gray")
        ax.set_title(f"True: {CLASS_NAMES[label]}")
        ax.axis("off")
    fig.suptitle("MNIST Training Samples", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            if optimizer is not None:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += batch_size

    return total_loss / total, correct / total


def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> tuple[float, float]:
    return run_one_epoch(model, loader, criterion, device, optimizer=None)


def train_experiment(
    optimizer_name: str,
    learning_rate: float,
    optimizer_factory: Callable[[nn.Module], optim.Optimizer],
    train_loader: DataLoader,
    validation_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    log: Callable[[str], None],
) -> ExperimentResult:
    set_seed(SEED)
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_factory(model)
    history: list[EpochMetrics] = []

    log(f"=== Training with {optimizer_name} (lr={learning_rate}) ===")
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = run_one_epoch(
            model, train_loader, criterion, device, optimizer=optimizer
        )
        validation_loss, validation_accuracy = evaluate(model, validation_loader, criterion, device)
        history.append(
            EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                validation_loss=validation_loss,
                validation_accuracy=validation_accuracy,
            )
        )
        log(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train loss {train_loss:.4f}, train acc {train_accuracy:.4f} | "
            f"val loss {validation_loss:.4f}, val acc {validation_accuracy:.4f}"
        )

    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    log(f"Test loss: {test_loss:.4f}, test accuracy: {test_accuracy:.4f}")
    log("")

    save_test_predictions(
        model,
        test_loader,
        device,
        OUTPUT_DIR / f"test_predictions_{optimizer_name.lower()}.png",
    )

    return ExperimentResult(
        optimizer=optimizer_name,
        learning_rate=learning_rate,
        test_loss=test_loss,
        test_accuracy=test_accuracy,
        history=history,
    )


def save_test_predictions(model: nn.Module, loader: DataLoader, device: torch.device, output_path: Path) -> None:
    model.eval()
    selected_images: list[torch.Tensor] = []
    selected_labels: list[int] = []
    seen_labels: set[int] = set()
    for batch_images, batch_labels in loader:
        for image, label in zip(batch_images, batch_labels):
            label_value = int(label.item())
            if label_value not in seen_labels:
                seen_labels.add(label_value)
                selected_images.append(image)
                selected_labels.append(label_value)
            if len(seen_labels) == len(CLASS_NAMES):
                break
        if len(seen_labels) == len(CLASS_NAMES):
            break

    ordered = sorted(zip(selected_labels, selected_images), key=lambda item: item[0])
    labels = torch.tensor([label for label, _ in ordered])
    images = torch.stack([image for _, image in ordered])
    images = images.to(device)
    with torch.no_grad():
        predictions = model(images).argmax(dim=1).cpu()

    fig, axes = plt.subplots(2, 5, figsize=(11, 5.8))
    for ax, image, label, prediction in zip(axes.ravel(), images.cpu(), labels, predictions):
        ax.imshow(denormalize(image), cmap="gray")
        ax.set_title(f"True: {label.item()}  Pred: {prediction.item()}", fontsize=11)
        ax.axis("off")
    fig.suptitle("MNIST Test Predictions", fontsize=14)
    fig.subplots_adjust(top=0.86, hspace=0.45, wspace=0.08)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_curves(result: ExperimentResult) -> None:
    epochs = [item.epoch for item in result.history]
    train_loss = [item.train_loss for item in result.history]
    validation_loss = [item.validation_loss for item in result.history]
    train_accuracy = [item.train_accuracy for item in result.history]
    validation_accuracy = [item.validation_accuracy for item in result.history]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, train_loss, marker="o", label="Training Loss")
    ax.plot(epochs, validation_loss, marker="s", label="Validation Loss")
    ax.set_title("Training and Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "loss_curve.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, train_accuracy, marker="o", label="Training Accuracy")
    ax.plot(epochs, validation_accuracy, marker="s", label="Validation Accuracy")
    ax.set_title("Training and Validation Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "accuracy_curve.png", dpi=180)
    plt.close(fig)


def save_optimizer_comparison(results: list[ExperimentResult]) -> None:
    with (OUTPUT_DIR / "optimizer_comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Optimizer", "Learning Rate", "Test Loss", "Test Accuracy"])
        for result in results:
            writer.writerow(
                [
                    result.optimizer,
                    result.learning_rate,
                    f"{result.test_loss:.4f}",
                    f"{result.test_accuracy:.4f}",
                ]
            )


def save_metrics_json(results: list[ExperimentResult], device: torch.device) -> None:
    payload = {
        "device": str(device),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "validation_size": VALIDATION_SIZE,
        "results": [
            {
                **{key: value for key, value in asdict(result).items() if key != "history"},
                "history": [asdict(item) for item in result.history],
            }
            for result in results
        ],
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    log_lines: list[str] = []

    def log(message: str) -> None:
        print(message)
        log_lines.append(message)

    set_seed(SEED)
    device = environment_check(log)
    train_loader, validation_loader, test_loader, full_train = get_dataloaders()
    save_sample_images(full_train, OUTPUT_DIR / "sample_images.png")

    experiments = [
        (
            "SGD",
            0.01,
            lambda model: optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
        ),
        (
            "Adam",
            0.001,
            lambda model: optim.Adam(model.parameters(), lr=0.001),
        ),
    ]
    results = [
        train_experiment(name, lr, factory, train_loader, validation_loader, test_loader, device, log)
        for name, lr, factory in experiments
    ]

    best_result = max(results, key=lambda item: item.test_accuracy)
    shutil.copyfile(
        OUTPUT_DIR / f"test_predictions_{best_result.optimizer.lower()}.png",
        OUTPUT_DIR / "test_predictions.png",
    )
    plot_curves(best_result)
    save_optimizer_comparison(results)
    save_metrics_json(results, device)
    (OUTPUT_DIR / "training_log.txt").write_text("\n".join(log_lines), encoding="utf-8")
    create_report(OUTPUT_DIR / "metrics.json", Path("2023101143_沈嘉克_ML_CV_Assignment_Report.docx"))
    log("Report generated: 2023101143_沈嘉克_ML_CV_Assignment_Report.docx")


if __name__ == "__main__":
    main()
