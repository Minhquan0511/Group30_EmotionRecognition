"""Huan luyen va so sanh Softmax, MLP va CNN nho tren FER-2013."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


CLASS_NAMES = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def apply_clahe(images: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Tang tuong phan cuc bo cho tung anh xam."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return np.stack([clahe.apply(image) for image in images])


def calculate_normalization(images: np.ndarray) -> tuple[float, float]:
    """Chi dung tap train de tinh mean/std, tranh ro ri du lieu."""
    normalized = images.astype(np.float32) / 255.0
    mean = float(normalized.mean())
    std = float(normalized.std())
    return mean, max(std, 1e-6)


class FerDataset(Dataset):
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        mean: float = 0.0,
        std: float = 1.0,
        augment: bool = False,
    ):
        self.images = images
        self.labels = labels
        self.mean = mean
        self.std = std
        self.augment = augment

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[index].copy()
        if self.augment:
            if random.random() < 0.5:
                image = cv2.flip(image, 1)

            angle = random.uniform(-10.0, 10.0)
            scale = random.uniform(0.90, 1.10)
            shift_x = random.uniform(-4.0, 4.0)
            shift_y = random.uniform(-4.0, 4.0)
            matrix = cv2.getRotationMatrix2D((23.5, 23.5), angle, scale)
            matrix[0, 2] += shift_x
            matrix[1, 2] += shift_y
            image = cv2.warpAffine(
                image,
                matrix,
                (48, 48),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )

            contrast = random.uniform(0.90, 1.10)
            brightness = random.uniform(-8.0, 8.0)
            image = np.clip(image.astype(np.float32) * contrast + brightness, 0, 255)

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0
        image = (image - self.mean) / self.std

        label = torch.tensor(self.labels[index], dtype=torch.long)
        return image, label


class SoftmaxModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Flatten(), nn.Linear(48 * 48, 7))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MlpModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48 * 48, 128),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(128, 7),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class SmallCnn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, 64),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(64, 7),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


MODEL_BUILDERS = {
    "softmax": SoftmaxModel,
    "mlp": MlpModel,
    "cnn": SmallCnn,
}


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def calculate_class_weights(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(labels, minlength=7).astype(np.float32)
    weights = len(labels) / (7.0 * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    model.eval()
    losses: list[float] = []
    predictions: list[int] = []
    targets: list[int] = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        losses.append(float(criterion(logits, labels).item()))
        predictions.extend(logits.argmax(dim=1).cpu().numpy().tolist())
        targets.extend(labels.cpu().numpy().tolist())

    target_array = np.asarray(targets)
    prediction_array = np.asarray(predictions)
    accuracy = accuracy_score(target_array, prediction_array)
    macro_f1 = f1_score(target_array, prediction_array, average="macro", zero_division=0)
    return (
        float(np.mean(losses)),
        float(accuracy),
        float(macro_f1),
        target_array,
        prediction_array,
    )


def plot_history(history: dict[str, list[float]], path: Path, title: str) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(epochs, history["val_accuracy"], label="Accuracy")
    axes[1].plot(epochs, history["val_macro_f1"], label="Macro-F1")
    axes[1].set_title("Validation metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_confusion_matrix(targets: np.ndarray, predictions: np.ndarray, path: Path) -> None:
    matrix = confusion_matrix(targets, predictions, labels=range(7))
    figure, axis = plt.subplots(figsize=(6.2, 5.2))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_xticks(range(7), CLASS_NAMES, rotation=35, ha="right")
    axis.set_yticks(range(7), CLASS_NAMES)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    for row in range(7):
        for column in range(7):
            axis.text(column, row, matrix[row, column], ha="center", va="center", fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def train_one_model(
    name: str,
    loaders: dict[str, DataLoader],
    class_weights: torch.Tensor,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    output_dir: Path,
    preprocessing: dict[str, float | bool],
) -> dict[str, float | int | str]:
    model = MODEL_BUILDERS[name]().to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = {key: [] for key in ["train_loss", "val_loss", "val_accuracy", "val_macro_f1"]}
    best_state = copy.deepcopy(model.state_dict())
    best_f1 = -1.0
    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []
        for images, labels in loaders["train"]:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        val_loss, val_accuracy, val_f1, _, _ = evaluate(
            model, loaders["val"], device, criterion
        )
        history["train_loss"].append(float(np.mean(train_losses)))
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["val_macro_f1"].append(val_f1)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"{name:7s} epoch {epoch:02d}/{epochs}: "
            f"val_acc={val_accuracy:.4f}, val_macro_f1={val_f1:.4f}"
        )

    elapsed = time.perf_counter() - start_time
    model.load_state_dict(best_state)
    torch.save(
        {
            "model_state": best_state,
            "model_name": name,
            "preprocessing": preprocessing,
        },
        output_dir / f"{name}_best.pt",
    )
    test_loss, test_accuracy, test_f1, targets, predictions = evaluate(
        model, loaders["test"], device, criterion
    )
    plot_history(history, output_dir / f"{name}_history.png", name.upper())
    plot_confusion_matrix(targets, predictions, output_dir / f"{name}_confusion_matrix.png")
    return {
        "model": name,
        "parameters": parameter_count,
        "best_val_macro_f1": round(best_f1, 6),
        "test_loss": round(test_loss, 6),
        "test_accuracy": round(test_accuracy, 6),
        "test_macro_f1": round(test_f1, 6),
        "training_seconds": round(elapsed, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=MODEL_BUILDERS, default=list(MODEL_BUILDERS))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--mlp-learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate rieng cho MLP",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--clahe", action="store_true")
    parser.add_argument("--clahe-clip-limit", type=float, default=2.0)
    parser.add_argument("--standardize", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results/run"))
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(args.data)
    required = {f"{kind}_{split}" for kind in ("x", "y") for split in ("train", "val", "test")}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"Tep NPZ thieu mang: {sorted(missing)}")

    images = {split: data[f"x_{split}"] for split in ("train", "val", "test")}
    if args.clahe:
        print("Dang ap dung CLAHE cho train/validation/test...")
        images = {
            split: apply_clahe(split_images, args.clahe_clip_limit)
            for split, split_images in images.items()
        }

    if args.standardize:
        mean, std = calculate_normalization(images["train"])
    else:
        mean, std = 0.0, 1.0
    preprocessing = {
        "clahe": args.clahe,
        "clahe_clip_limit": args.clahe_clip_limit,
        "standardize": args.standardize,
        "mean": mean,
        "std": std,
    }
    datasets = {
        split: FerDataset(
            images[split],
            data[f"y_{split}"],
            mean=mean,
            std=std,
            augment=(split == "train" and args.augment),
        )
        for split in ("train", "val", "test")
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=0,
        )
        for split, dataset in datasets.items()
    }
    device = choose_device()
    class_weights = calculate_class_weights(data["y_train"])
    print(f"Device: {device}")
    print(f"Class weights: {class_weights.tolist()}")
    print(f"Preprocessing: {preprocessing}; augmentation={args.augment}")

    config = vars(args).copy()
    config["data"] = str(config["data"])
    config["output_dir"] = str(config["output_dir"])
    config["device"] = str(device)
    config["split_sizes"] = {split: len(dataset) for split, dataset in datasets.items()}
    config["preprocessing"] = preprocessing
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    summaries = [
        train_one_model(
            name,
            loaders,
            class_weights,
            device,
            args.epochs,
            args.mlp_learning_rate if name == "mlp" else args.learning_rate,
            args.output_dir,
            preprocessing,
        )
        for name in args.models
    ]
    (args.output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
