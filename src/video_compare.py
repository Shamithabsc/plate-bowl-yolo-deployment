from ultralytics import YOLO
import cv2
import time
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

VIDEO_PATH = r"C:\Users\shamm\Documents\PIKKY\plate-bowl-yolo-deployment\test_videos\4.mp4"

MODEL_DIR = Path(
    r"C:\Users\shamm\Documents\PIKKY\plate-bowl-yolo-deployment\models"
)

MODELS = {
    "YOLO11n": MODEL_DIR / "yolo11n_best.pt",
    "YOLO11s": MODEL_DIR / "yolo11s_best.pt",
    "YOLO11m": MODEL_DIR / "yolo11m_best.pt"
}

IMAGE_SIZE = 512
CONFIDENCE = 0.4

# ============================================================
# TEST EACH MODEL
# ============================================================

for model_name, model_path in MODELS.items():

    print("\n" + "=" * 60)
    print(f"Testing {model_name}")
    print("=" * 60)

    model = YOLO(str(model_path))

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError("Could not open video.")

    frame_count = 0

    start = time.perf_counter()

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        results = model.predict(
            source=frame,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            device="cpu",
            verbose=False
        )

        annotated = results[0].plot()

        cv2.imshow(
            f"{model_name} - Press Q to stop",
            annotated
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    end = time.perf_counter()

    cap.release()
    cv2.destroyAllWindows()

    total_time = end - start

    latency = (
        total_time / frame_count
    ) * 1000

    fps = frame_count / total_time

    print(f"Frames   : {frame_count}")
    print(f"Time     : {total_time:.2f} s")
    print(f"Latency  : {latency:.2f} ms/frame")
    print(f"FPS      : {fps:.2f}")