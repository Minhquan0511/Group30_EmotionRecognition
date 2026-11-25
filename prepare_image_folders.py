"""Chuyen FER-2013 dang thu muc anh thanh NPZ theo dung split goc."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


CLASS_TO_LABEL = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "sad": 4,
    "surprise": 5,
    "neutral": 6,
}


def read_images(paths: list[Path], labels: list[int]) -> tuple[np.ndarray, np.ndarray]:
    images: list[np.ndarray] = []
    for index, path in enumerate(paths, start=1):
        with Image.open(path) as image:
            image_array = np.asarray(image.convert("L"), dtype=np.uint8)
        if image_array.shape != (48, 48):
            raise ValueError(f"Anh {path} co kich thuoc {image_array.shape}, can (48, 48)")
        images.append(image_array)
        if index % 5000 == 0:
            print(f"Da doc {index}/{len(paths)} anh")
    return np.stack(images), np.asarray(labels, dtype=np.int64)


def collect_train(root: Path) -> tuple[list[Path], list[int]]:
    paths: list[Path] = []
    labels: list[int] = []
    for class_name, label in CLASS_TO_LABEL.items():
        class_paths = sorted((root / "train" / class_name).glob("*.jpg"))
        if not class_paths:
            raise ValueError(f"Khong co anh trong train/{class_name}")
        paths.extend(class_paths)
        labels.extend([label] * len(class_paths))
    return paths, labels


def collect_test_split(root: Path, prefix: str) -> tuple[list[Path], list[int]]:
    paths: list[Path] = []
    labels: list[int] = []
    for class_name, label in CLASS_TO_LABEL.items():
        class_paths = sorted((root / "test" / class_name).glob(f"{prefix}_*.jpg"))
        if not class_paths:
            raise ValueError(f"Khong co anh {prefix} trong test/{class_name}")
        paths.extend(class_paths)
        labels.extend([label] * len(class_paths))
    return paths, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/fer2013_full.npz"))
    args = parser.parse_args()

    train_paths, train_labels = collect_train(args.root)
    val_paths, val_labels = collect_test_split(args.root, "PublicTest")
    test_paths, test_labels = collect_test_split(args.root, "PrivateTest")

    expected = {"train": 28709, "val": 3589, "test": 3589}
    actual = {"train": len(train_paths), "val": len(val_paths), "test": len(test_paths)}
    if actual != expected:
        raise ValueError(f"So luong anh khong dung FER-2013 goc: {actual}, can {expected}")

    print(f"So luong split: {actual}")
    x_train, y_train = read_images(train_paths, train_labels)
    x_val, y_val = read_images(val_paths, val_labels)
    x_test, y_test = read_images(test_paths, test_labels)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
    )
    print(f"Da luu: {args.output}")


if __name__ == "__main__":
    main()
