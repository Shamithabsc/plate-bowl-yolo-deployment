# Plate & Bowl Detection using YOLO11

A computer vision project for detecting **plates and bowls** in images, videos, and live webcam streams using custom-trained **YOLO11n, YOLO11s, and YOLO11m** object detection models.

The project evaluates the three YOLO11 model variants based on detection accuracy, model size, inference speed, and real-time webcam performance, with the final goal of selecting a suitable model for deployment.

---

## 1. Project Overview

The objective of this project is to develop and evaluate an object detection system capable of identifying:

* **Plate**
* **Bowl**

The trained YOLO11 models are tested under multiple real-world input conditions:

1. Single image detection
2. Batch image detection
3. Video detection
4. Video speed/latency comparison
5. Live webcam detection
6. Webcam detection logging and analysis
7. Comparison of YOLO11n, YOLO11s and YOLO11m
8. Preparation for real-time application/deployment

The project focuses not only on model accuracy but also on **inference speed, computational requirements, and practical deployment performance**.

Ultralytics YOLO provides Python APIs for training, validation, prediction and deployment-oriented model export. YOLO prediction supports images, directories, videos and live camera sources.

---

# 2. Models Used

Three custom-trained YOLO11 detection models are evaluated:

| Model   | Parameters | Model Size | GFLOPs | Purpose                                       |
| ------- | ---------: | ---------: | -----: | --------------------------------------------- |
| YOLO11n |     ~2.58M |    5.20 MB |   ~6.3 | Lightweight / fastest model                   |
| YOLO11s |     ~9.41M |   18.27 MB |  ~21.3 | Balanced model                                |
| YOLO11m |    ~20.03M |   38.62 MB |  ~67.7 | Larger / more computationally expensive model |

The models were trained specifically for the two project classes:

```text
0 - plate
1 - bowl
```

Each model has its own trained `best.pt` weight file.

---

# 3. Dataset

The object detection dataset contains images of plates and bowls.

### Dataset Distribution

| Split      | Images | Plates | Bowls | Total Objects |
| ---------- | -----: | -----: | ----: | ------------: |
| Train      |   3356 |   5045 |  1900 |          6945 |
| Validation |    420 |    647 |   236 |           883 |
| Test       |    420 |    577 |   304 |           881 |

The dataset contains different appearances and object conditions. However, during real-world webcam testing, additional domain differences were observed, particularly for **steel/stainless-steel plates and bowls** compared with the mostly ceramic/glass objects present in the training data.

This is being investigated as a potential cause of lower detection performance in some real-world scenarios.

---

# 4. Project Structure

```text
plate-bowl-yolo-deployment/
│
├── models/
│   ├── yolo11n_best.pt
│   ├── yolo11s_best.pt
│   └── yolo11m_best.pt
│
├── src/
│   ├── image_detection.py
│   ├── batch_image_detection.py
│   ├── video_detection.py
│   ├── video_compare.py
│   ├── webcam_detection.py
│   ├── analyze_webcam.py
│   └── camera_test.py
│
├── test_images/
│   ├── test_img1.jpg
│   ├── test_img2.jpg
│   └── ...
│
├── test_videos/
│   ├── 10.mp4
│   └── ...
│
├── outputs/
│   ├── webcam_detections.csv
│   ├── webcam_output.mp4
│   └── ...
│
├── requirements.txt
├── README.md
└── .venv/
```

---

# 5. Environment Setup

The project is developed in Python using a virtual environment.

Main libraries used include:

* Python
* PyTorch
* Ultralytics
* OpenCV
* Streamlit
* Streamlit-WebRTC
* NumPy
* Pandas

The environment uses:

```text
Python 3.12
PyTorch 2.13.0+cpu
Ultralytics 8.4.112
Streamlit 1.60.0
Streamlit-WebRTC 0.76.2
```

The project uses a virtual environment:

```text
.venv/
```

To activate it in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell execution policy prevents activation, the Python executable inside the environment can also be used directly.

---

# 6. Trained Model Weights

The trained models are stored inside the `models` folder:

```text
models/
├── yolo11n_best.pt
├── yolo11s_best.pt
└── yolo11m_best.pt
```

The `best.pt` file contains the trained model weights produced during training.

Each model can be independently loaded using:

```python
from ultralytics import YOLO

model = YOLO("models/yolo11s_best.pt")
```

---

# 7. Image Detection

The first stage of deployment testing is image detection.

The image detection script:

1. Loads a trained YOLO model.
2. Loads a test image.
3. Resizes/preprocesses the image for inference.
4. Runs YOLO object detection.
5. Identifies plates and bowls.
6. Applies the confidence threshold.
7. Generates bounding boxes.
8. Displays/saves the detection result.

Example:

```python
model.predict(
    source=image,
    imgsz=512,
    conf=0.4
)
```

The resulting image contains bounding boxes around detected objects.

Example output concept:

```text
┌─────────────────────────────┐
│                             │
│       ┌──────────────┐      │
│       │    PLATE     │      │
│       │   0.82       │      │
│       └──────────────┘      │
│                             │
│   ┌───────────┐             │
│   │   BOWL    │             │
│   │   0.76    │             │
│   └───────────┘             │
│                             │
└─────────────────────────────┘
```

Ultralytics prediction returns `Results` objects containing detection information such as bounding boxes, and annotated results can be displayed or saved.

---

# 8. Batch Image Testing

After testing individual images, multiple test images are processed as a batch.

The purpose is to determine how the model behaves across a larger set of unseen images.

Batch testing evaluates:

* Number of detected plates
* Number of detected bowls
* Confidence scores
* Bounding-box locations
* Missed detections
* False detections
* Overall inference behavior

This provides a more reliable assessment than testing only one image.

---

# 9. Image Detection Performance

The trained models are evaluated using standard object detection metrics:

### Precision

Measures how many predicted objects are actually correct.

### Recall

Measures how many actual objects were successfully detected.

### mAP@0.5

Mean Average Precision using an IoU threshold of 0.5.

### mAP@0.5:0.95

Mean Average Precision averaged over IoU thresholds from 0.5 to 0.95.

### F1 Score

The harmonic mean of precision and recall.

### Confidence Threshold

The minimum confidence required for a detection to be displayed.

A lower confidence threshold can increase detections but may also increase false positives.

---

# 10. Video Detection

After image testing, the models are tested using video input.

The video pipeline works as follows:

```text
Input Video
     ↓
Read Video Frame
     ↓
Resize / Preprocess
     ↓
YOLO Detection
     ↓
Detect Plate / Bowl
     ↓
Draw Bounding Boxes
     ↓
Display Frame
     ↓
Save Annotated Frame
     ↓
Read Next Frame
```

The model processes the video **frame by frame**.

For example, a video containing 116 frames results in approximately 116 individual inference operations.

Each frame is passed through the YOLO model:

```python
results = model.predict(
    source=frame,
    imgsz=512,
    conf=0.3
)
```

The predictions are then drawn onto the frame.

The annotated frames are combined into an output video.

---

# 11. Video Speed Testing

The `video_compare.py` script compares the inference speed of all three models.

For each model, the following values are measured:

* Number of frames
* Total processing time
* Average latency per frame
* Processing FPS

Example output:

```text
YOLO11n
Frames   : 116
Time     : 11.95 s
Latency  : 102.98 ms/frame
FPS      : 9.71

YOLO11s
Frames   : 116
Time     : 16.02 s
Latency  : 138.09 ms/frame
FPS      : 7.24

YOLO11m
Frames   : 116
Time     : 38.96 s
Latency  : 335.85 ms/frame
FPS      : 2.98
```

This demonstrates the computational trade-off between the models.

The smaller YOLO11n model provides the highest processing speed, while YOLO11m requires substantially more computation.

---

# 12. Video Output

The video detection script generates an annotated output video.

The output contains:

* Original video frames
* Detected plates
* Detected bowls
* Bounding boxes
* Confidence scores
* Class labels

This allows qualitative inspection of how well the model performs over time.

It also helps identify:

* Missed objects
* Incorrect classifications
* Unstable detections
* Detections appearing/disappearing between frames
* Problems with distant objects
* Problems caused by occlusion

---

# 13. Webcam Detection

The next stage is real-time webcam detection.

The webcam pipeline is:

```text
Webcam
  ↓
Capture Frame
  ↓
Resize / Preprocess
  ↓
YOLO11s
  ↓
Object Detection
  ↓
Confidence Filtering
  ↓
Bounding Boxes
  ↓
Display Live Result
  ↓
Save Detection Data
```

The webcam is accessed using OpenCV:

```python
cap = cv2.VideoCapture(0)
```

Frames are continuously captured:

```python
ret, frame = cap.read()
```

Each frame is passed to YOLO:

```python
results = model.predict(
    source=frame,
    imgsz=512,
    conf=0.4
)
```

The detections are then displayed in a live OpenCV window.

The webcam test also calculates real-time FPS.

---

# 14. Webcam Detection Logging

Unlike simple visual testing, the webcam implementation also stores the detection results.

The detection information is saved into:

```text
outputs/webcam_detections.csv
```

The CSV contains information such as:

```text
Frame
Class
Confidence
x1
y1
x2
y2
```

This makes it possible to perform quantitative analysis of webcam predictions after the live test.

For example:

```text
Frame  Class   Confidence
1      bowl    0.72
2      bowl    0.68
3      plate   0.51
...
```

The webcam output video is also stored for later inspection.

---

# 15. Webcam Detection Analysis

The `analyze_webcam.py` script analyzes the saved CSV file.

The analysis includes:

* Total number of detections
* Number of plate detections
* Number of bowl detections
* Average confidence
* Minimum confidence
* Maximum confidence
* Number of detections per frame

An example result from YOLO11s was:

```text
Total detections : 637

Detections by class:
bowl     370
plate    267

Average confidence:
bowl     0.634
plate    0.526
```

This indicates that the model was detecting bowls more confidently than plates during the particular webcam test.

---

# 16. Model Comparison

The three models are compared based on:

| Requirement        | YOLO11n   | YOLO11s   | YOLO11m          |
| ------------------ | --------- | --------- | ---------------- |
| Detection Accuracy | Evaluated | Evaluated | Evaluated        |
| Inference Speed    | Fastest   | Medium    | Slowest          |
| Model Size         | Smallest  | Medium    | Largest          |
| Computational Cost | Lowest    | Medium    | Highest          |
| Edge Deployment    | Suitable  | Suitable  | More demanding   |
| Overall Balance    | Good      | Strong    | Accuracy-focused |

The model selection should not be based only on mAP.

For deployment, the following factors need to be considered together:

```text
Accuracy
   +
Inference Speed
   +
Model Size
   +
Memory Usage
   +
Real-time Performance
```

---

# 17. Current Model Performance

The validation results obtained during the experiments were approximately:

### YOLO11n

```text
Precision    : 0.7666
Recall       : 0.5875
mAP50        : 0.7020
mAP50-95     : 0.5717
F1 Score     : 0.6652
```

### YOLO11s

```text
Precision    : 0.7952
Recall       : 0.5862
mAP50        : 0.7014
mAP50-95     : 0.5774
F1 Score     : 0.6749
```

### YOLO11m

```text
Precision    : 0.7587
Recall       : 0.5949
mAP50        : 0.6998
mAP50-95     : 0.5691
F1 Score     : 0.6669
```

The validation results show that increasing model size did **not automatically produce better detection performance** for this particular dataset and training setup.

YOLO11s provides a useful balance between detection performance and computational cost.

---

# 18. Current Deployment Findings

During real-world testing, several observations were made.

### Bowls

The model generally detects bowls relatively well.

### Plates

Plate detection is less consistent, especially when the plate is:

* Far from the camera
* Small in the frame
* Partially hidden
* Under different lighting
* Made from reflective steel

### Distance

Detection performance decreases when objects are moved farther away from the camera.

This is expected because the object occupies fewer pixels in the input frame.

### Material

The training dataset contained more ceramic/glass-like objects, while real-world testing included more stainless-steel utensils.

This creates a potential **domain shift** between the training data and deployment environment.

---

# 19. Image Resolution Investigation

The current inference configuration uses:

```text
IMAGE_SIZE = 512
```

The project can investigate higher inference resolutions such as:

```text
512 × 512
640 × 640
768 × 768
```

Higher resolution may help with small/distant objects because more visual information is retained.

However, increasing resolution also increases computational cost and may reduce FPS.

Therefore, resolution should be evaluated experimentally rather than changed without measurement.

---

# 20. Confidence Threshold Investigation

The current confidence threshold is generally around:

```text
0.3 - 0.4
```

A lower threshold can be used during investigation.

For example:

```python
CONFIDENCE = 0.2
```

This can determine whether a missed plate is actually being predicted with low confidence.

The final confidence threshold should be selected based on the desired balance between:

* False positives
* False negatives
* Detection stability
* Real-time performance

---

# 21. Current Improvement Strategy

The next improvement stage is to collect additional deployment-specific images.

Particular attention should be given to:

### Plates

* Stainless steel plates
* Ceramic plates
* Different colors
* Different sizes
* Different distances
* Different lighting
* Partially visible plates
* Plates containing food

### Bowls

* Stainless steel bowls
* Ceramic bowls
* Glass bowls
* Different sizes
* Different shapes
* Different distances
* Different lighting
* Partially visible bowls

Additional negative examples can also be included, such as:

* Cups
* Glasses
* Trays
* Containers
* Lids
* Pans
* Other kitchen utensils

The objective is to make the training distribution more representative of the actual deployment environment.

---

# 22. Real-Time Deployment Plan

The planned deployment architecture is:

```text
YOLO11 best.pt
       ↓
Python inference
       ↓
OpenCV / Webcam
       ↓
Real-time detection
       ↓
Bounding boxes
       ↓
Confidence scores
       ↓
Performance logging
       ↓
Streamlit application
```

A Streamlit-based interface can later be used to provide a more user-friendly application.

For browser-based real-time video, `streamlit-webrtc` can be used to connect webcam/video streams to a Streamlit application.

---

# 23. Future Deployment

Possible future deployment stages include:

### Stage 1

Complete image testing.

### Stage 2

Complete video testing.

### Stage 3

Complete webcam testing.

### Stage 4

Improve detection performance using deployment-specific training data.

### Stage 5

Compare the improved YOLO11 models again.

### Stage 6

Select the final model.

### Stage 7

Build a Streamlit interface.

### Stage 8

Integrate live webcam detection into the Streamlit application.

### Stage 9

Investigate model export for optimized deployment.

Ultralytics supports exporting trained models to deployment formats such as ONNX and TensorRT, which can be investigated after the PyTorch-based implementation is stable.

---

# 24. How the Complete System Works

The complete project can be summarized as:

```text
                TRAINED DATASET
                       │
                       ▼
              YOLO11 MODEL TRAINING
                       │
             ┌─────────┼─────────┐
             ▼         ▼         ▼
          YOLO11n   YOLO11s   YOLO11m
             │         │         │
             └─────────┼─────────┘
                       ▼
                MODEL VALIDATION
                       │
                       ▼
              ACCURACY COMPARISON
                       │
                       ▼
                IMAGE TESTING
                       │
                       ▼
                BATCH TESTING
                       │
                       ▼
                 VIDEO TESTING
                       │
                       ▼
               SPEED COMPARISON
                       │
                       ▼
                WEBCAM TESTING
                       │
                       ▼
             DETECTION DATA LOGGING
                       │
                       ▼
              WEBCAM DATA ANALYSIS
                       │
                       ▼
             IDENTIFY MODEL WEAKNESS
                       │
              ┌────────┴─────────┐
              ▼                  ▼
       Distance / Size      Material / Lighting
              │                  │
              └────────┬─────────┘
                       ▼
             COLLECT MORE DATA
                       │
                       ▼
                 FINE-TUNING
                       │
                       ▼
               RE-EVALUATION
                       │
                       ▼
              FINAL MODEL SELECTION
                       │
                       ▼
               STREAMLIT / WEB APP
                       │
                       ▼
              REAL-TIME DEPLOYMENT
```

---

# 25. Important Project Goal

The goal of this project is **not simply to train a YOLO model**.

The project covers the complete process of taking a trained computer vision model and evaluating its suitability for a real-world application:

```text
Training
   ↓
Validation
   ↓
Inference
   ↓
Performance Testing
   ↓
Real-world Testing
   ↓
Failure Analysis
   ↓
Dataset Improvement
   ↓
Fine-tuning
   ↓
Deployment
```

This allows the model to be evaluated both quantitatively through metrics such as precision, recall, mAP and FPS, and qualitatively through actual image, video and webcam predictions.

---

# 26. Technologies Used

* **Python**
* **YOLO11**
* **Ultralytics**
* **PyTorch**
* **OpenCV**
* **Pandas**
* **NumPy**
* **Streamlit**
* **Streamlit-WebRTC**
* **CSV-based detection logging**

---

# 27. Running the Project

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Run image detection:

```powershell
python src\image_detection.py
```

Run video detection:

```powershell
python src\video_detection.py
```

Compare video inference speed:

```powershell
python src\video_compare.py
```

Run webcam detection:

```powershell
python src\webcam_detection.py
```

Analyze webcam detections:

```powershell
python src\analyze_webcam.py
```

The webcam application can be closed using:

```text
Q
```

---

# 28. Current Status

### Completed

* [x] Dataset preparation
* [x] YOLO11n training
* [x] YOLO11s training
* [x] YOLO11m training
* [x] Model validation
* [x] Model architecture comparison
* [x] Single image testing
* [x] Batch image testing
* [x] Image bounding-box visualization
* [x] Video detection
* [x] Video speed comparison
* [x] Webcam detection
* [x] Webcam FPS measurement
* [x] Webcam detection logging
* [x] Webcam CSV analysis
* [x] Real-world performance analysis

### In Progress / Next

* [ ] Improve plate detection
* [ ] Test steel plates and bowls
* [ ] Test different object distances
* [ ] Test different inference resolutions
* [ ] Optimize confidence threshold
* [ ] Collect deployment-specific images
* [ ] Fine-tune model with additional real-world data
* [ ] Re-test improved model
* [ ] Final model selection
* [ ] Streamlit deployment
* [ ] Real-time browser/webcam deployment

---

# 29. Conclusion

This project implements and evaluates a YOLO11-based object detection system for plates and bowls across progressively more realistic deployment scenarios.

The models are first evaluated using validation metrics and then tested on images, batches of images, videos and live webcam input. In addition to detection accuracy, inference latency, FPS, model size and real-time behavior are measured.

The current experiments indicate that **YOLO11s provides a strong balance between detection performance and computational requirements**, while YOLO11n provides faster inference and YOLO11m requires substantially more computation.

Real-world webcam testing has also identified areas for improvement, particularly **plate detection, small/distant objects and differences between training-object materials and deployment objects**. These findings will guide the next stage of dataset improvement and fine-tuning.

The final objective is to obtain a robust plate-and-bowl detector that can operate reliably on real-time webcam input and can subsequently be integrated into a Streamlit-based application.
