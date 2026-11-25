"""Nhan dien cam xuc tu webcam bang checkpoint CNN da huan luyen."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from train_compare import CLASS_NAMES, SmallCnn


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def preprocess(
    face_gray: np.ndarray,
    device: torch.device,
    preprocessing: dict[str, float | bool],
) -> torch.Tensor:
    resized = cv2.resize(face_gray, (48, 48), interpolation=cv2.INTER_AREA)
    if bool(preprocessing.get("clahe", False)):
        clahe = cv2.createCLAHE(
            clipLimit=float(preprocessing.get("clahe_clip_limit", 2.0)),
            tileGridSize=(8, 8),
        )
        resized = clahe.apply(resized)
    tensor = torch.tensor(resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    tensor = tensor / 255.0
    mean = float(preprocessing.get("mean", 0.0))
    std = float(preprocessing.get("std", 1.0))
    return ((tensor - mean) / std).to(device)


def load_model(
    checkpoint: Path, device: torch.device
) -> tuple[SmallCnn, dict[str, float | bool]]:
    model = SmallCnn().to(device)
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    if "model_state" in saved:
        state = saved["model_state"]
        preprocessing = saved.get("preprocessing", {})
    else:
        state = saved
        preprocessing = {}
    model.load_state_dict(state)
    model.eval()
    return model, preprocessing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--smooth-frames",
        type=int,
        default=5,
        help="So khung hinh dung de lam muot xac suat",
    )
    args = parser.parse_args()

    if args.smooth_frames < 1:
        raise ValueError("--smooth-frames phai >= 1")

    device = choose_device()
    model, preprocessing = load_model(args.checkpoint, device)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError("Khong tai duoc bo phat hien khuon mat Haar Cascade")

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(f"Khong mo duoc camera {args.camera}")

    probability_history: deque[np.ndarray] = deque(maxlen=args.smooth_frames)
    print(f"Device: {device}. Nhan Q de thoat.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Khong doc duoc khung hinh tu camera")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(48, 48),
            )

            # Demo mot nguoi: uu tien khuon mat co dien tich lon nhat.
            if len(faces) > 0:
                x, y, width, height = max(faces, key=lambda box: box[2] * box[3])
                face = gray[y : y + height, x : x + width]
                image = preprocess(face, device, preprocessing)
                with torch.no_grad():
                    probabilities = torch.softmax(model(image), dim=1)[0].cpu().numpy()
                probability_history.append(probabilities)
                smoothed = np.mean(probability_history, axis=0)
                label_index = int(np.argmax(smoothed))
                confidence = float(smoothed[label_index])

                cv2.rectangle(frame, (x, y), (x + width, y + height), (31, 170, 120), 2)
                text = f"{CLASS_NAMES[label_index]}: {confidence * 100:.1f}%"
                cv2.putText(
                    frame,
                    text,
                    (x, max(24, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (31, 170, 120),
                    2,
                    cv2.LINE_AA,
                )
            else:
                probability_history.clear()

            cv2.putText(
                frame,
                "Q: thoat",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Group 30 - FER-2013 realtime demo", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
