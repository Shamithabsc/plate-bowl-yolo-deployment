from ultralytics import YOLO
import cv2
import time
from pathlib import Path

# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = r"C:\Users\shamm\Documents\PIKKY\plate-bowl-yolo-deployment\models\yolo11s_best.pt"

VIDEO_PATH = r"C:\Users\shamm\Documents\PIKKY\plate-bowl-yolo-deployment\test_videos\11.mp4"

OUTPUT_PATH = r"C:\Users\shamm\Documents\PIKKY\plate-bowl-yolo-deployment\test_videos\out3_yolo11s.mp4"

IMAGE_SIZE = 512
CONFIDENCE = 0.4

# ============================================================
# LOAD MODEL
# ============================================================

print("Loading YOLO11s...")

model = YOLO(MODEL_PATH)

print("Model loaded.")

# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)
 
if not cap.isOpened():
    raise RuntimeError("Could not open video.")

# Original video properties
fps_original = cap.get(cv2.CAP_PROP_FPS)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Original video: {width} x {height}")
print(f"Original FPS: {fps_original}")

# ============================================================
# OUTPUT VIDEO
# ============================================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps_original,
    (width, height)
)

# ============================================================
# PROCESS VIDEO
# ============================================================

frame_count = 0

start_time = time.perf_counter()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # YOLO detection
    results = model.predict(
        source=frame,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        device="cpu",
        verbose=False
    )

    # Draw bounding boxes
    annotated_frame = results[0].plot()

    # Write output
    out.write(annotated_frame)

    # Display
    cv2.imshow(
        "YOLO11s Plate-Bowl Detection",
        annotated_frame
    )

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

end_time = time.perf_counter()

# ============================================================
# CLEANUP
# ============================================================

cap.release()
out.release()
cv2.destroyAllWindows()

# ============================================================
# PERFORMANCE
# ============================================================

total_time = end_time - start_time

processing_fps = frame_count / total_time

latency_ms = (
    total_time / frame_count
) * 1000

print("\n" + "=" * 60)
print("VIDEO RESULTS")
print("=" * 60)

print(f"Frames processed : {frame_count}")
print(f"Total time       : {total_time:.2f} seconds")
print(f"Latency          : {latency_ms:.2f} ms/frame")
print(f"Processing FPS   : {processing_fps:.2f}")
print(f"Output video     : {OUTPUT_PATH}")