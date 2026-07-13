from __future__ import annotations

from typing import Any, Sequence

import cv2
import numpy as np


LANDMARK_REGIONS = {
    0: "face",
    1: "left_eye_inner",
    2: "left_eye",
    3: "left_eye_outer",
    4: "right_eye_inner",
    5: "right_eye",
    6: "right_eye_outer",
    7: "left_ear",
    8: "right_ear",
    9: "mouth_left",
    10: "mouth_right",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    17: "left_pinky",
    18: "right_pinky",
    19: "left_index",
    20: "right_index",
    21: "left_thumb",
    22: "right_thumb",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    29: "left_heel",
    30: "right_heel",
    31: "left_foot_index",
    32: "right_foot_index",
}


class ThermalAnalyzer:
    """Phân tích nhiệt toàn thân và theo vùng bbox/landmark.

    Lưu ý: Nếu không có calibration thực, các giá trị nhiệt độ
    là giá trị tương đối.
    """

    def __init__(
        self,
        relative_mode: bool = True,
        min_temperature_celsius: float = 20.0,
        max_temperature_celsius: float = 45.0,
        colormap_size: int = 10,
    ) -> None:
        self.relative_mode = relative_mode
        self.min_temperature_celsius = min_temperature_celsius
        self.max_temperature_celsius = max_temperature_celsius
        self.colormap_size = colormap_size

    @staticmethod
    def _to_grayscale_float(frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        return gray.astype(np.float32)

    def _pixel_to_relative_temp(self, gray: np.ndarray, pixel_value: float) -> float:
        pixel_range = float(np.max(gray) - np.min(gray))
        if pixel_range < 1e-6:
            normalized = 0.5
        else:
            normalized = (float(pixel_value) - float(np.min(gray))) / pixel_range
        return self.min_temperature_celsius + normalized * (self.max_temperature_celsius - self.min_temperature_celsius)

    def analyze_roi(self, frame: np.ndarray, bbox: Sequence[float]) -> dict[str, Any]:
        """Phân tích vùng bounding box."""
        gray = self._to_grayscale_float(frame)
        x1, y1, x2, y2 = [int(coord) for coord in bbox]

        h, w = gray.shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))

        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            return {
                "temperature": None,
                "avg_pixel": None,
                "min_pixel": None,
                "max_pixel": None,
                "hot_region": False,
                "cold_region": False,
            }

        avg_pixel = float(np.mean(roi))
        min_pixel = float(np.min(roi))
        max_pixel = float(np.max(roi))

        temperature = self._pixel_to_relative_temp(gray, avg_pixel)
        hot_region = bool(max_pixel > avg_pixel * 1.05)
        cold_region = bool(min_pixel < avg_pixel * 0.95)

        return {
            "temperature": round(temperature, 1),
            "avg_pixel": avg_pixel,
            "min_pixel": min_pixel,
            "max_pixel": max_pixel,
            "hot_region": hot_region,
            "cold_region": cold_region,
        }

    def analyze_landmark_temperature(self, frame: np.ndarray, landmarks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Phân tích nhiệt tại các landmark cơ thể."""
        if not landmarks:
            return []

        gray = self._to_grayscale_float(frame)
        h, w = gray.shape[:2]
        landmark_temps = []
        for landmark in landmarks:
            lx = int(max(0, min(landmark["x"], w - 1)))
            ly = int(max(0, min(landmark["y"], h - 1)))
            pixel_value = float(gray[ly, lx])
            temperature = self._pixel_to_relative_temp(gray, pixel_value)
            landmark_temps.append(
                {
                    "index": landmark["index"],
                    "label": LANDMARK_REGIONS.get(landmark["index"], f"point_{landmark['index']}"),
                    "x": landmark["x"],
                    "y": landmark["y"],
                    "pixel": pixel_value,
                    "temperature": round(temperature, 1),
                    "visibility": landmark["visibility"],
                }
            )
        return landmark_temps

    def analyze_frame(self, frame: np.ndarray, detections: list[dict], pose_result: dict | None = None) -> list[dict[str, Any]]:
        """Phân tích toàn thân: bbox + landmark + summary."""
        results = []
        for detection in detections:
            roi_analysis = self.analyze_roi(frame, detection["bbox"])
            landmark_temps: list[dict[str, Any]] = []
            if pose_result and pose_result.get("landmarks"):
                landmark_temps = self.analyze_landmark_temperature(frame, pose_result["landmarks"])

            if landmark_temps:
                visible_temps = [item["temperature"] for item in landmark_temps if item["visibility"] > 0.4]
                if visible_temps:
                    avg_temperature = round(float(np.mean(visible_temps)), 1)
                    min_temperature = round(float(np.min(visible_temps)), 1)
                    max_temperature = round(float(np.max(visible_temps)), 1)
                    hot_regions = [item for item in landmark_temps if item["visibility"] > 0.4 and item["temperature"] >= avg_temperature]
                    cold_regions = [item for item in landmark_temps if item["visibility"] > 0.4 and item["temperature"] < avg_temperature - 0.5]
                else:
                    avg_temperature = roi_analysis["temperature"]
                    min_temperature = roi_analysis["temperature"]
                    max_temperature = roi_analysis["temperature"]
                    hot_regions = []
                    cold_regions = []
            else:
                avg_temperature = roi_analysis["temperature"]
                min_temperature = roi_analysis["temperature"]
                max_temperature = roi_analysis["temperature"]
                hot_regions = []
                cold_regions = []

            results.append(
                {
                    "bbox": detection["bbox"],
                    "confidence": detection["confidence"],
                    "roi": roi_analysis,
                    "summary": {
                        "avg_temperature": avg_temperature,
                        "min_temperature": min_temperature,
                        "max_temperature": max_temperature,
                        "hot_regions_count": len(hot_regions),
                        "cold_regions_count": len(cold_regions),
                        "thermal_status": "NORMAL",
                    },
                    "landmark_temps": landmark_temps,
                    "colormap_bar": self._build_colormap_bar(),
                }
            )

            if results[-1]["summary"]["hot_regions_count"] > 0:
                results[-1]["summary"]["thermal_status"] = "HOT"
            elif results[-1]["summary"]["cold_regions_count"] > 0:
                results[-1]["summary"]["thermal_status"] = "COLD"

        return results

    def _build_colormap_bar(self) -> np.ndarray:
        bar = np.zeros((60, 220, 3), dtype=np.uint8)
        for i in range(bar.shape[1]):
            value = int((i / (bar.shape[1] - 1)) * 255)
            gray = np.full((bar.shape[0], 1), value, dtype=np.uint8)
            colored = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
            bar[:, i : i + 1] = colored
        cv2.rectangle(bar, (0, 0), (bar.shape[1] - 1, bar.shape[0] - 1), (255, 255, 255), 1)
        cv2.putText(bar, f"{self.max_temperature_celsius}C", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(bar, f"{self.min_temperature_celsius}C", (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return bar
