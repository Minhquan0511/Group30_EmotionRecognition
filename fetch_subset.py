"""Lay tap con FER-2013 tu Hugging Face datasets-server de smoke test."""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np


DATASET = "abhilash88/fer2013-enhanced"
CONFIG = "default"
MAX_PAGE_SIZE = 100


def fetch_rows(split: str, size: int) -> tuple[np.ndarray, np.ndarray]:
    images: list[np.ndarray] = []
    labels: list[int] = []

    for offset in range(0, size, MAX_PAGE_SIZE):
        length = min(MAX_PAGE_SIZE, size - offset)
        query = urllib.parse.urlencode(
            {
                "dataset": DATASET,
                "config": CONFIG,
                "split": split,
                "offset": offset,
                "length": length,
            }
        )
        url = f"https://datasets-server.huggingface.co/rows?{query}"

        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    payload = json.load(response)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)

        for item in payload["rows"]:
            row = item["row"]
            image = np.asarray(row["image"], dtype=np.uint8)
            if image.shape != (48, 48):
                raise ValueError(f"Kich thuoc anh khong hop le: {image.shape}")
            images.append(image)
            labels.append(int(row["emotion"]))
        print(f"{split}: {len(images)}/{size}")

    return np.stack(images), np.asarray(labels, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=int, default=2100)
    parser.add_argument("--val-size", type=int, default=700)
    parser.add_argument("--test-size", type=int, default=700)
    parser.add_argument(
        "--output", type=Path, default=Path("data/fer2013_subset.npz")
    )
    args = parser.parse_args()

    x_train, y_train = fetch_rows("train", args.train_size)
    x_val, y_val = fetch_rows("validation", args.val_size)
    x_test, y_test = fetch_rows("test", args.test_size)

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

