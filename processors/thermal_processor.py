from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


class ThermalVideoReader:
    """Đọc video nhiệt từ file và trả về từng frame."""

    def __init__(self, video_path: str | Path) -> None:
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video không tồn tại: {self.video_path}")

        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Không thể mở video: {self.video_path}")

        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    def read(self) -> np.ndarray | None:
        """Đọc một frame từ video."""
        success, frame = self.cap.read()
        return frame if success else None

    def release(self) -> None:
        """Giải phóng tài nguyên."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self) -> "ThermalVideoReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


class ThermalProcessor:
    """Xử lý frame nhiệt: colormap, cân bằng histogram, ghép hiển thị."""

    def __init__(self) -> None:
        self.colormaps = {
            "JET": cv2.COLORMAP_JET,
            "INFERNO": cv2.COLORMAP_INFERNO,
            "MAGMA": cv2.COLORMAP_MAGMA,
        }

    def apply_colormap(self, frame: np.ndarray, name: str = "INFERNO") -> np.ndarray:
        """Chuyển frame xám sang ảnh màu colormap."""
        if frame.ndim != 2:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        return cv2.applyColorMap(gray, self.colormaps[name])

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        """Cải thiện độ tương phản nhẹ cho video thường."""
        if frame.ndim != 2:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        enhanced = cv2.equalizeHist(gray)
        return cv2.applyColorMap(enhanced, self.colormaps["INFERNO"])

    def build_side_by_side(self, frame: np.ndarray, name: str = "INFERNO") -> np.ndarray:
        """Hiển thị song song ảnh gốc và ảnh giả màu."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        colored = self.apply_colormap(gray, name=name)
        original = frame if frame.ndim == 3 else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return np.hstack([original, colored])

    def get_available_colormaps(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        """Trả về tất cả colormap để kiểm tra thủ công."""
        return {name: self.apply_colormap(frame, name=name) for name in self.colormaps}


class PoseEstimator:
    """Ước lượng pose bằng MediaPipe."""

    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5) -> None:
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def estimate(self, frame: np.ndarray) -> dict | None:
        """Trả về landmarks và annotation nếu phát hiện người."""
        if frame is None:
            return None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self.pose.process(rgb)
        rgb.flags.writeable = True

        if not result.pose_landmarks:
            return None

        h, w = frame.shape[:2]
        landmarks = []
        for idx, lm in enumerate(result.pose_landmarks.landmark):
            landmarks.append(
                {
                    "index": idx,
                    "x": float(lm.x * w),
                    "y": float(lm.y * h),
                    "z": float(lm.z),
                    "visibility": float(lm.visibility),
                }
            )

        return {
            "landmarks": landmarks,
            "world_landmarks": [
                {
                    "index": idx,
                    "x": float(lm.x),
                    "y": float(lm.y),
                    "z": float(lm.z),
                    "visibility": float(lm.visibility),
                }
                for idx, lm in enumerate(result.pose_world_landmarks.landmark)
            ]
            if result.pose_world_landmarks
            else [],
        }

    def close(self) -> None:
        self.pose.close()
