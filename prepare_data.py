"""Chuyen FER-2013 CSV goc thanh tep NPZ de huan luyen nhanh hon."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


USAGE_TO_SPLIT = {
    "Training": "train",
    "PublicTest": "val",
    "PrivateTest": "test",
}


def parse_pixels(pixel_text: str) -> np.ndarray:
    pixels = np.fromstring(pixel_text, dtype=np.uint8, sep=" ")
    if pixels.size != 48 * 48:
        raise ValueError(f"Anh co {pixels.size} pixel, can dung 2304 pixel")
    return pixels.reshape(48, 48)


def convert(csv_path: Path, output_path: Path) -> None:
    frame = pd.read_csv(csv_path)
    required = {"emotion", "pixels", "Usage"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"CSV thieu cot: {sorted(missing)}")

    arrays: dict[str, np.ndarray] = {}
    for usage, split in USAGE_TO_SPLIT.items():
        part = frame.loc[frame["Usage"] == usage]
        if part.empty:
            raise ValueError(f"Khong tim thay du lieu Usage={usage}")
        arrays[f"x_{split}"] = np.stack(
            [parse_pixels(text) for text in part["pixels"]]
        )
        arrays[f"y_{split}"] = part["emotion"].to_numpy(dtype=np.int64)
        print(f"{split}: {len(part)} anh")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    print(f"Da luu: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/fer2013.npz"))
    args = parser.parse_args()
    convert(args.csv, args.output)


if __name__ == "__main__":
    main()

