from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np


PANEL_BG = (15, 15, 15)
PANEL_TEXT = (240, 240, 240)
ACCENT_NORMAL = (180, 255, 180)
ACCENT_HOT = (80, 80, 255)
ACCENT_COLD = (255, 180, 80)


def _put_text(image: np.ndarray, text: str, position: tuple[int, int], color: tuple[int, int, int] = PANEL_TEXT) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, position, font, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, position, font, 0.6, color, 2, cv2.LINE_AA)


def _put_multiline(image: np.ndarray, lines: Sequence[str], origin: tuple[int, int], line_gap: int = 22, color: tuple[int, int, int] = PANEL_TEXT) -> None:
    x, y = origin
    for line in lines:
        _put_text(image, line, (x, y), color=color)
        y += line_gap


class Dashboard:
    """Giao diện OpenCV hiển thị kết quả phân tích."""

    def __init__(self, window_name: str = "OmniCare Thermal AI") -> None:
        self.window_name = window_name
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def _draw_info_panel(self, frame: np.ndarray, result: dict) -> np.ndarray:
        output = frame.copy()
        x1, y1, x2, y2 = [int(v) for v in result["bbox"]]
        status = result["summary"]["thermal_status"]
        accent = ACCENT_NORMAL if status == "NORMAL" else ACCENT_HOT if status == "HOT" else ACCENT_COLD

        overlay = output.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), accent, -1)
        cv2.addWeighted(overlay, 0.08, output, 0.92, 0, output)
        cv2.rectangle(output, (x1, y1), (x2, y2), accent, 2)

        cv2.rectangle(output, (x1, y1 - 120), (x1 + 270, y1), PANEL_BG, -1)
        cv2.rectangle(output, (x1, y1 - 120), (x1 + 270, y1), accent, 1)

        lines = [
            f"Confidence: {result['confidence'] * 100:.0f}%",
            f"Avg temp: {result['summary']['avg_temperature']} C",
            f"Min temp: {result['summary']['min_temperature']} C",
            f"Max temp: {result['summary']['max_temperature']} C",
            f"Hot regions: {result['summary']['hot_regions_count']}",
            f"Cold regions: {result['summary']['cold_regions_count']}",
            f"Status: {status}",
        ]
        _put_multiline(output, lines, origin=(x1 + 10, y1 - 108), line_gap=16, color=accent)
        return output

    def render(self, frame: np.ndarray, results: list[dict]) -> np.ndarray:
        image = frame.copy()

        if not results:
            panel_h = min(170, image.shape[0])
            roi = image[:panel_h, :]
            top_panel = np.full((panel_h, image.shape[1], 3), PANEL_BG, dtype=np.uint8)
            cv2.addWeighted(top_panel, 0.75, roi, 0.25, 0, roi)
            lines = [
                "OmniCare Thermal AI",
                "Person: 0",
                "Temperature: N/A C",
                "Thermal Status: NO PERSON",
                "Confidence: N/A",
                "Body keypoints: 0",
            ]
            _put_multiline(image, lines, origin=(20, 30), line_gap=24)
            return image

        person = results[0]
        panel_h = min(170, image.shape[0])
        roi = image[:panel_h, :]
        top_panel = np.full((panel_h, image.shape[1], 3), PANEL_BG, dtype=np.uint8)
        cv2.addWeighted(top_panel, 0.75, roi, 0.25, 0, roi)

        status = person["summary"]["thermal_status"]
        top_lines = [
            "OmniCare Thermal AI",
            f"Person: {len(results)}",
            f"Temperature: {person['summary']['avg_temperature']} C",
            f"Thermal Status: {status}",
            f"Detection Confidence: {person['confidence'] * 100:.0f}%",
            f"Body keypoints: {len(person.get('landmark_temps', []))}",
        ]
        _put_multiline(image, top_lines, origin=(20, 22), line_gap=24)

        for index, landmark in enumerate(person.get("landmark_temps", []) or []):
            if landmark.get("visibility", 0) <= 0.4:
                continue
            lx = int(landmark.get("x", 0))
            ly = int(landmark.get("y", 0))
            label = f"{landmark['label']}: {landmark['temperature']}C"
            _put_text(image, label, (lx + 6, ly - 6), color=(255, 255, 0))
            cv2.circle(image, (lx, ly), 4, (0, 255, 255), -1, lineType=cv2.LINE_AA)

        image = self._draw_info_panel(image, person)

        if len(results) > 1:
            cv2.putText(
                image,
                f"+{len(results) - 1} more",
                (20, image.shape[0] - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                2,
                cv2.LINE_AA,
            )
        return image

    def render_compact(self, frame: np.ndarray, results: list[dict]) -> np.ndarray:
        image = frame.copy()
        for index, person in enumerate(results[:2]):
            x1, y1, x2, y2 = [int(v) for v in person["bbox"]]
            status = person["summary"]["thermal_status"]
            accent = ACCENT_NORMAL if status == "NORMAL" else ACCENT_HOT if status == "HOT" else ACCENT_COLD
            cv2.rectangle(image, (x1, y1), (x2, y2), accent, 2)

            if person.get("landmark_temps"):
                for landmark in person["landmark_temps"]:
                    if landmark["visibility"] > 0.4:
                        cv2.circle(image, (int(landmark["x"]), int(landmark["y"])), 3, (0, 255, 255), -1, lineType=cv2.LINE_AA)

            panel = np.full((110, 220, 3), PANEL_BG, dtype=np.uint8)
            cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, panel.shape[0] - 1), accent, 1)
            text = [
                f"Person {index + 1}",
                f"Temp: {person['summary']['avg_temperature']} C",
                f"Status: {status}",
                f"Conf: {person['confidence'] * 100:.0f}%",
            ]
            _put_multiline(panel, text, origin=(8, 20), line_gap=22)
            image[y1 : y1 + panel.shape[0], x1 : x1 + panel.shape[1]] = panel

        return image

    def show(self, frame: np.ndarray) -> None:
        cv2.imshow(self.window_name, frame)

    def wait_key(self, delay: int = 1) -> int:
        return cv2.waitKey(delay) & 0xFF

    def close(self) -> None:
        cv2.destroyAllWindows()
