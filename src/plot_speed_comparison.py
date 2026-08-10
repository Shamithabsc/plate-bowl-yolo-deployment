import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv("speed_comparison.csv")

print(df)

# ------------------------------------------------------------
# Latency
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    df["Model"],
    df["Average Latency (ms)"]
)

plt.xlabel("Model")
plt.ylabel("Average Latency (ms)")
plt.title("YOLO11 Model Latency Comparison")

plt.tight_layout()
plt.savefig(
    "latency_comparison.png",
    dpi=300
)

plt.show()

# ------------------------------------------------------------
# FPS
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    df["Model"],
    df["FPS"]
)

plt.xlabel("Model")
plt.ylabel("Frames Per Second")
plt.title("YOLO11 Model FPS Comparison")

plt.tight_layout()
plt.savefig(
    "fps_comparison.png",
    dpi=300
)

plt.show()