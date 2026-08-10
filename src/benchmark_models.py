from ultralytics import YOLO
from pathlib import Path
import time
import csv

# ============================================================
# PATHS
# ============================================================

TEST_DIR = Path(
    r"C:\Users\shamm\Documents\PIKKY\plate-bowl-yolo-deployment\test_images"
)

MODELS = {
    "YOLO11n": r"C:\Users\shamm\Documents\PIKKY\plate-bowl-yolo-deployment\models\yolo11n_best.pt",
    "YOLO11s": r"C:\Users\shamm\Documents\PIKKY\plate-bowl-yolo-deployment\models\yolo11s_best.pt",
    "YOLO11m": r"C:\Users\shamm\Documents\PIKKY\plate-bowl-yolo-deployment\models\yolo11m_best.pt",
}

# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 512
CONFIDENCE = 0.5

# ============================================================
# GET IMAGES
# ============================================================

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

images = sorted([
    p for p in TEST_DIR.iterdir()
    if p.suffix.lower() in extensions
])

if not images:
    raise RuntimeError("No test images found.")

print("=" * 70)
print("YOLO11 DEPLOYMENT SPEED BENCHMARK")
print("=" * 70)

print(f"Images       : {len(images)}")
print(f"Input size   : {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"Confidence   : {CONFIDENCE}")
print("Device       : CPU")

# ============================================================
# RESULTS
# ============================================================

results_table = []

# ============================================================
# TEST EACH MODEL
# ============================================================

for model_name, model_path in MODELS.items():

    print("\n" + "=" * 70)
    print(f"TESTING {model_name}")
    print("=" * 70)

    # Load model
    model = YOLO(model_path)

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    print("Warming up...")

    model.predict(
        source=str(images[0]),
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        device="cpu",
        verbose=False
    )

    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------

    print("Running benchmark...")

    start = time.perf_counter()

    for image_path in images:

        model.predict(
            source=str(image_path),
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            device="cpu",
            verbose=False
        )

    end = time.perf_counter()

    total_time = end - start

    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    number_images = len(images)

    average_latency_ms = (
        total_time / number_images
    ) * 1000

    fps = number_images / total_time

    result = {
        "Model": model_name,
        "Images": number_images,
        "Total Time (s)": round(total_time, 3),
        "Average Latency (ms)": round(average_latency_ms, 2),
        "FPS": round(fps, 2)
    }

    results_table.append(result)

    print(f"\n{model_name}")
    print(f"Total time      : {total_time:.3f} s")
    print(f"Average latency : {average_latency_ms:.2f} ms/image")
    print(f"FPS             : {fps:.2f}")

# ============================================================
# FINAL TABLE
# ============================================================

print("\n")
print("=" * 80)
print("FINAL SPEED COMPARISON")
print("=" * 80)

print(
    f"{'Model':<12}"
    f"{'Images':<10}"
    f"{'Total Time':<15}"
    f"{'Latency':<20}"
    f"{'FPS':<10}"
)

print("-" * 80)

for result in results_table:

    print(
        f"{result['Model']:<12}"
        f"{result['Images']:<10}"
        f"{result['Total Time (s)']:<15}"
        f"{result['Average Latency (ms)']:<20}"
        f"{result['FPS']:<10}"
    )

# ============================================================
# SAVE CSV
# ============================================================

output_file = Path("speed_comparison.csv")

with open(output_file, "w", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=results_table[0].keys()
    )

    writer.writeheader()
    writer.writerows(results_table)

print("\nSaved:")
print(output_file.resolve())