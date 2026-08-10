import pandas as pd

CSV_PATH = r"C:\Users\shamm\Documents\PIKKY\plate-bowl-yolo-deployment\outputs\webcam_detections.csv"

df = pd.read_csv(CSV_PATH)

print("\n" + "=" * 60)
print("WEBCAM DETECTION ANALYSIS")
print("=" * 60)

print(f"Total detections : {len(df)}")

print("\nDetections by class:")
print(df["Class"].value_counts())

print("\nAverage confidence:")
print(
    df.groupby("Class")["Confidence"]
    .mean()
    .round(3)
)

print("\nMinimum confidence:")
print(
    df.groupby("Class")["Confidence"]
    .min()
    .round(3)
)

print("\nMaximum confidence:")
print(
    df.groupby("Class")["Confidence"]
    .max()
    .round(3)
)

print("\nDetections per frame:")
print(
    df.groupby("Frame")["Class"]
    .count()
    .describe()
)

print("\n" + "=" * 60)