from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from ultralytics import YOLO


class PersonDetector:
    """Phát hiện người bằng YOLO11/YOLOv8."""

    def __init__(self, model_path: str | Path = "yolo11n.pt") -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Không tìm thấy model: {self.model_path}")
        self.model = YOLO(str(self.model_path))

    def detect(self, frame: np.ndarray, conf: float = 0.25) -> list[dict]:
        """Phát hiện người và trả về bbox, confidence."""
        results = self.model(frame, conf=conf, classes=[0], verbose=False)
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            xyxy = boxes.xyxy.cpu().numpy().astype(float)
            scores = boxes.conf.cpu().numpy().astype(float)
            for box, score in zip(xyxy, scores):
                detections.append(
                    {
                        "bbox": [float(v) for v in box],
                        "confidence": float(score),
                    }
                )
        detections.sort(key=lambda x: x["confidence"], reverse=True)
        return detections
