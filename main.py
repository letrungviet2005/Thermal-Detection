from __future__ import annotations

from pathlib import Path
import time

import cv2
import numpy as np

from detectors import PersonDetector, ThermalAnalyzer
from processors import ThermalProcessor, ThermalVideoReader, PoseEstimator
from ui.dashboard import Dashboard

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEO = PROJECT_ROOT / "videos" / "thermal.mp4"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "yolo11n.pt"
SIMULATION_VIDEO = PROJECT_ROOT / "videos" / "simulation.mp4"


class SimulationThermalVideoReader:
    """Tạo video giả lập từ video thường nếu không có video nhiệt."""

    def __init__(self, source: str | Path, fake_colormap: str = "INFERNO") -> None:
        self.cap = cv2.VideoCapture(str(source))
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Video giả lập không tồn tại: {source}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.fake_colormap = fake_colormap
        self.processor = ThermalProcessor()

    def read(self) -> np.ndarray | None:
        success, frame = self.cap.read()
        if not success:
            return None
        return self.processor.enhance(frame)

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self) -> "SimulationThermalVideoReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


def run_pipeline(video_path: str | Path, model_path: str | Path = DEFAULT_MODEL, simulation: bool = False, show_mode: str = "default") -> None:
    """Pipeline chính: đọc video -> phát hiện người -> phân tích nhiệt -> hiển thị."""
    video_path = Path(video_path)
    model_path = Path(model_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video không tồn tại: {video_path}")

    reader = SimulationThermalVideoReader(video_path) if simulation else ThermalVideoReader(video_path)
    processor = ThermalProcessor()
    pose_estimator = PoseEstimator()
    detector = PersonDetector(model_path)
    analyzer = ThermalAnalyzer()
    dashboard = Dashboard()

    print(f"Đang xử lý video: {video_path}")
    print("Nhấn 'q' để thoát, 'p' để tạm dừng, 's' để lưu snapshot, 'm' để đổi chế độ hiển thị.")

    paused = False
    frame_index = 0
    frame_duration_ms = 1000.0 / reader.fps if reader.fps > 0 else 33.0
    last_frame_time = time.time()

    with reader:
        while True:
            if not paused:
                frame = reader.read()
                if frame is None:
                    print("Đã hết video.")
                    break
                frame_index += 1

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
                colored = processor.apply_colormap(gray, name="INFERNO")
                pose_result = pose_estimator.estimate(colored)
                detections = detector.detect(colored, conf=0.25)
                results = analyzer.analyze_frame(colored, detections, pose_result=pose_result)
                output = {"persons": results}

                display = processor.build_side_by_side(colored)
                if show_mode == "compact":
                    display = dashboard.render_compact(display, results)
                else:
                    display = dashboard.render(display, results)
                dashboard.show(display)

            if not paused:
                elapsed_ms = (time.time() - last_frame_time) * 1000.0
                wait_ms = max(1, int(frame_duration_ms - elapsed_ms))
                last_frame_time = time.time()
                if wait_ms > 1:
                    time.sleep(wait_ms / 1000.0)
            key = dashboard.wait_key(delay=1)

            if key == ord("q"):
                break
            elif key == ord("p"):
                paused = not paused
            elif key == ord(" ") and paused:
                paused = False
            elif key == ord("s"):
                save_path = PROJECT_ROOT / "videos" / f"snapshot_{frame_index}.jpg"
                cv2.imwrite(str(save_path), display)
                print(f"Đã lưu snapshot: {save_path}")
            elif key == ord("m"):
                show_mode = "compact" if show_mode == "default" else "default"
                print(f"Chuyển chế độ hiển thị: {show_mode}")

    pose_estimator.close()
    dashboard.close()


def main() -> None:
    print("OmniCare Thermal AI")
    print("Tùy chọn:")
    print("1. Chạy với video nhiệt")
    print("2. Chạy chế độ giả lập với video thường")

    choice = input("Nhập lựa chọn (1/2): ").strip() or "2"
    if choice == "1":
        video_path = input(f"Đường dẫn video nhiệt (mặc định {DEFAULT_VIDEO}): ").strip() or str(DEFAULT_VIDEO)
        run_pipeline(video_path, model_path=DEFAULT_MODEL, simulation=False)
    else:
        video_path = input(f"Đường dẫn video thường để giả lập (mặc định {SIMULATION_VIDEO}): ").strip() or str(
            SIMULATION_VIDEO
        )
        if not Path(video_path).exists():
            print(f"Không tìm thấy video giả lập: {video_path}")
            return
        run_pipeline(video_path, model_path=DEFAULT_MODEL, simulation=True)


if __name__ == "__main__":
    main()
