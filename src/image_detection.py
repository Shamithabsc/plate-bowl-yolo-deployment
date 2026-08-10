from ultralytics import YOLO
import os

# -----------------------------
# PATHS
# -----------------------------

MODEL_PATH = r"C:\Users\shamm\Documents\PIKKY\plate-bowl-yolo-deployment\models\yolo11s_best.pt"

IMAGE_PATH = r"C:\Users\shamm\Documents\PIKKY\plate-bowl-yolo-deployment\test_images\test_img5.jpg"

# -----------------------------
# CHECK FILES
# -----------------------------

print("Model exists:", os.path.exists(MODEL_PATH))
print("Image exists:", os.path.exists(IMAGE_PATH))

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

# -----------------------------
# LOAD MODEL
# -----------------------------

print("\nLoading YOLO11s model...")

model = YOLO(MODEL_PATH)

print("Model loaded successfully.")

# -----------------------------
# RUN DETECTION
# -----------------------------

print("\nRunning detection...")

results = model.predict(
    source=IMAGE_PATH,
    imgsz=512,
    conf=0.5,
    save=True
)

print("\nDetection completed.")
print("Output saved in runs/detect/predict")