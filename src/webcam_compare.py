from ultralytics import YOLO
import cv2
import time
import csv
from pathlib import Path

# ============================================================
# SETTINGS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\shamm\Documents\PIKKY\plate-bowl-yolo-deployment"
)

MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "webcam_compare"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "YOLO11n": MODEL_DIR / "yolo11n_best.pt",
    "YOLO11s": MODEL_DIR / "yolo11s_best.pt",
    "YOLO11m": MODEL_DIR / "yolo11m_best.pt",
}

IMAGE_SIZE = 512
CONFIDENCE = 0.4

# ============================================================
# RUN ONE MODEL
# ============================================================

def test_model(model_name, model_path):

    print("\n" + "=" * 60)
    print(f"Testing {model_name}")
    print("=" * 60)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("Loading model...")

    model = YOLO(str(model_path))

    print("Model loaded.")

    # --------------------------------------------------------
    # Webcam
    # --------------------------------------------------------

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam.")
        return

    # Set webcam resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Webcam Resolution: {width} x {height}")
    print("Press Q to stop.")

    # --------------------------------------------------------
    # Output files
    # --------------------------------------------------------

    video_path = OUTPUT_DIR / f"{model_name}_webcam.mp4"
    csv_path = OUTPUT_DIR / f"{model_name}_detections.csv"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(
        str(video_path),
        fourcc,
        20,
        (width, height)
    )

    csv_file = open(
        csv_path,
        "w",
        newline=""
    )

    writer = csv.writer(csv_file)

    writer.writerow([
        "Frame",
        "Class",
        "Confidence",
        "x1",
        "y1",
        "x2",
        "y2"
    ])

    # --------------------------------------------------------
    # Performance variables
    # --------------------------------------------------------

    frame_count = 0
    detection_count = 0

    total_inference_time = 0

    start_time = time.perf_counter()

    # --------------------------------------------------------
    # Webcam loop
    # --------------------------------------------------------

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # ----------------------------------------------------
        # YOLO prediction
        # ----------------------------------------------------

        inference_start = time.perf_counter()

        results = model.predict(
            source=frame,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            device="cpu",
            verbose=False
        )

        inference_end = time.perf_counter()

        inference_time = (
            inference_end - inference_start
        )

        total_inference_time += inference_time

        result = results[0]

        # ----------------------------------------------------
        # Extract detections
        # ----------------------------------------------------

        if result.boxes is not None:

            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for box, conf, cls in zip(
                boxes,
                confidences,
                classes
            ):

                x1, y1, x2, y2 = box

                class_id = int(cls)

                class_name = model.names[class_id]

                detection_count += 1

                writer.writerow([
                    frame_count,
                    class_name,
                    round(float(conf), 4),
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2)
                ])

        # ----------------------------------------------------
        # Draw bounding boxes
        # ----------------------------------------------------

        annotated_frame = result.plot()

        # ----------------------------------------------------
        # Calculate FPS
        # ----------------------------------------------------

        current_fps = 1 / inference_time

        cv2.putText(
            annotated_frame,
            f"{model_name}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            annotated_frame,
            f"FPS: {current_fps:.2f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            annotated_frame,
            f"Frame: {frame_count}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # ----------------------------------------------------
        # Show webcam
        # ----------------------------------------------------

        cv2.imshow(
            f"{model_name} Webcam Detection",
            annotated_frame
        )

        # ----------------------------------------------------
        # Save video
        # ----------------------------------------------------

        out.write(annotated_frame)

        # ----------------------------------------------------
        # Quit
        # ----------------------------------------------------

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # --------------------------------------------------------
    # Finish
    # --------------------------------------------------------

    end_time = time.perf_counter()

    total_time = end_time - start_time

    average_latency = (
        total_inference_time / frame_count
    ) * 1000

    average_fps = (
        frame_count / total_time
    )

    cap.release()
    out.release()
    csv_file.close()

    cv2.destroyAllWindows()

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print(f"{model_name} RESULTS")
    print("=" * 60)

    print(f"Frames processed : {frame_count}")
    print(f"Detections       : {detection_count}")
    print(f"Average latency  : {average_latency:.2f} ms")
    print(f"Average FPS      : {average_fps:.2f}")

    print(f"CSV file         : {csv_path}")
    print(f"Output video     : {video_path}")

    print("=" * 60)

    return {
        "model": model_name,
        "frames": frame_count,
        "detections": detection_count,
        "latency": average_latency,
        "fps": average_fps
    }


# ============================================================
# MAIN
# ============================================================

results = []

for model_name, model_path in MODELS.items():

    if not model_path.exists():

        print(
            f"\nWARNING: Model not found: {model_path}"
        )

        continue

    result = test_model(
        model_name,
        model_path
    )

    if result:
        results.append(result)


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n\n")
print("=" * 70)
print("FINAL WEBCAM MODEL COMPARISON")
print("=" * 70)

print(
    f"{'Model':<12}"
    f"{'Frames':<12}"
    f"{'Detections':<15}"
    f"{'Latency(ms)':<15}"
    f"{'FPS':<10}"
)

print("-" * 70)

for r in results:

    print(
        f"{r['model']:<12}"
        f"{r['frames']:<12}"
        f"{r['detections']:<15}"
        f"{r['latency']:<15.2f}"
        f"{r['fps']:<10.2f}"
    )

print("=" * 70)