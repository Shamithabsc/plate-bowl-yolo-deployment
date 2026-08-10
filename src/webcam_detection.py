from ultralytics import YOLO
import cv2
import time
import csv
from pathlib import Path

# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = r"C:\Users\shamm\Documents\PIKKY\plate-bowl-yolo-deployment\models\yolo11s_best.pt"

IMAGE_SIZE = 512
CONFIDENCE = 0.4

# Output folder
OUTPUT_DIR = Path(
    r"C:\Users\shamm\Documents\PIKKY\plate-bowl-yolo-deployment\outputs"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_DIR / "webcam_detections.csv"
VIDEO_PATH = OUTPUT_DIR / "webcam_output.mp4"

# ============================================================
# LOAD MODEL
# ============================================================

print("Loading YOLO11s...")

model = YOLO(MODEL_PATH)

print("Model Loaded Successfully!")

# ============================================================
# OPEN WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

# Request 512 x 512 camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 512)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 512)

# Get actual camera resolution
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Webcam Resolution: {width} x {height}")

# ============================================================
# VIDEO WRITER
# ============================================================

camera_fps = cap.get(cv2.CAP_PROP_FPS)

# Some webcams return 0 or an invalid FPS
if camera_fps <= 0:
    camera_fps = 20.0

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

video_writer = cv2.VideoWriter(
    str(VIDEO_PATH),
    fourcc,
    camera_fps,
    (width, height)
)

# ============================================================
# CSV FILE
# ============================================================

csv_file = open(
    CSV_PATH,
    "w",
    newline="",
    encoding="utf-8"
)

writer = csv.writer(csv_file)

writer.writerow([
    "Frame",
    "Timestamp",
    "Class",
    "Confidence",
    "x1",
    "y1",
    "x2",
    "y2"
])

# ============================================================
# FPS VARIABLES
# ============================================================

frame_count = 0

prev_time = time.perf_counter()

fps = 0

# ============================================================
# LIVE DETECTION
# ============================================================

print("\nWebcam Started")
print("Press Q to stop.\n")

try:

    while True:

        # ----------------------------------------------------
        # READ FRAME
        # ----------------------------------------------------

        ret, frame = cap.read()

        if not ret:
            print("Could not read frame.")
            break

        frame_count += 1

        # ----------------------------------------------------
        # YOLO DETECTION
        # ----------------------------------------------------

        results = model.predict(
            source=frame,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            device="cpu",
            verbose=False
        )

        result = results[0]

        # ----------------------------------------------------
        # COUNT OBJECTS
        # ----------------------------------------------------

        plate_count = 0
        bowl_count = 0

        # ----------------------------------------------------
        # SAVE DETECTIONS TO CSV
        # ----------------------------------------------------

        if result.boxes is not None:

            for box in result.boxes:

                # Class ID
                class_id = int(box.cls[0])

                # Class name
                class_name = model.names[class_id]

                # Confidence
                confidence = float(box.conf[0])

                # Bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Count classes
                if class_name.lower() == "plate":
                    plate_count += 1

                elif class_name.lower() == "bowl":
                    bowl_count += 1

                # Timestamp
                timestamp = time.strftime("%H:%M:%S")

                # Write detection to CSV
                writer.writerow([
                    frame_count,
                    timestamp,
                    class_name,
                    round(confidence, 4),
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2)
                ])

        # ----------------------------------------------------
        # DRAW BOUNDING BOXES
        # ----------------------------------------------------

        annotated_frame = result.plot()

        # ----------------------------------------------------
        # CALCULATE FPS
        # ----------------------------------------------------

        current_time = time.perf_counter()

        elapsed = current_time - prev_time

        if elapsed > 0:
            instant_fps = 1 / elapsed

            # Smooth FPS
            fps = (0.9 * fps) + (0.1 * instant_fps)

        prev_time = current_time

        # ----------------------------------------------------
        # DISPLAY INFORMATION
        # ----------------------------------------------------

        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            annotated_frame,
            f"Plates: {plate_count}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

        cv2.putText(
            annotated_frame,
            f"Bowls: {bowl_count}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            annotated_frame,
            f"Frame: {frame_count}",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # ----------------------------------------------------
        # SHOW WEBCAM
        # ----------------------------------------------------

        cv2.imshow(
            "YOLO11s Live Plate & Bowl Detection",
            annotated_frame
        )

        # ----------------------------------------------------
        # SAVE ANNOTATED VIDEO
        # ----------------------------------------------------

        video_writer.write(annotated_frame)

        # ----------------------------------------------------
        # QUIT
        # ----------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

finally:

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()

    video_writer.release()

    csv_file.close()

    cv2.destroyAllWindows()

# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 60)
print("WEBCAM TEST COMPLETED")
print("=" * 60)

print(f"Frames processed : {frame_count}")
print(f"CSV file         : {CSV_PATH}")
print(f"Output video     : {VIDEO_PATH}")

print("=" * 60)