import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import tempfile
import os
import time
from pathlib import Path

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Plate & Bowl Detection",
    page_icon="🍽️",
    layout="wide"
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATHS = {
    "YOLO11n": MODEL_DIR / "yolo11n_best.pt",
    "YOLO11s": MODEL_DIR / "yolo11s_best.pt",
    "YOLO11m": MODEL_DIR / "yolo11m_best.pt"
}

# ============================================================
# TITLE
# ============================================================

st.title("🍽️ Plate & Bowl Detection")
st.write(
    "YOLO11-based object detection for plates and bowls."
)

st.divider()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Detection Settings")

model_name = st.sidebar.selectbox(
    "Select Model",
    ["YOLO11s", "YOLO11n", "YOLO11m"],
    index=0
)

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.40,
    step=0.05
)

image_size = st.sidebar.selectbox(
    "Image Size",
    [320, 416, 512, 640],
    index=2
)

st.sidebar.info(
    f"""
Model: {model_name}

Confidence: {confidence}

Image Size: {image_size}
"""
)

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model(model_path):

    if not model_path.exists():
        st.error(f"Model not found: {model_path}")
        st.stop()

    return YOLO(str(model_path))


model = load_model(MODEL_PATHS[model_name])

# ============================================================
# MAIN TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(
    [
        "🖼️ Image Detection",
        "🎥 Video Detection",
        "📷 Webcam Detection"
    ]
)

# ============================================================
# IMAGE DETECTION
# ============================================================

with tab1:

    st.header("Image Detection")

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.subheader("Input Image")

        st.image(
            image,
            use_container_width=True
        )

        if st.button(
            "🔍 Detect Objects",
            key="image_detect"
        ):

            start_time = time.perf_counter()

            image_array = np.array(image)

            results = model.predict(
                source=image_array,
                imgsz=image_size,
                conf=confidence,
                verbose=False
            )

            end_time = time.perf_counter()

            processing_time = end_time - start_time

            result = results[0]

            annotated_image = result.plot()

            # Convert BGR → RGB
            annotated_image = cv2.cvtColor(
                annotated_image,
                cv2.COLOR_BGR2RGB
            )

            # ------------------------------------------------
            # DETECTION COUNTS
            # ------------------------------------------------

            plate_count = 0
            bowl_count = 0

            detections = []

            if result.boxes is not None:

                for box in result.boxes:

                    class_id = int(
                        box.cls[0].item()
                    )

                    conf = float(
                        box.conf[0].item()
                    )

                    class_name = result.names[class_id]

                    if class_name.lower() == "plate":
                        plate_count += 1

                    elif class_name.lower() == "bowl":
                        bowl_count += 1

                    detections.append(
                        {
                            "Class": class_name,
                            "Confidence": round(conf, 3)
                        }
                    )

            # ------------------------------------------------
            # DISPLAY
            # ------------------------------------------------

            st.subheader("Detection Result")

            st.image(
                annotated_image,
                use_container_width=True
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Plates",
                plate_count
            )

            col2.metric(
                "Bowls",
                bowl_count
            )

            col3.metric(
                "Inference Time",
                f"{processing_time * 1000:.1f} ms"
            )

            if detections:

                st.subheader("Detection Details")

                st.dataframe(
                    detections,
                    use_container_width=True
                )

            else:

                st.warning(
                    "No plates or bowls detected."
                )


# ============================================================
# VIDEO DETECTION
# ============================================================

with tab2:

    st.header("Video Detection")

    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov"],
        key="video_upload"
    )

    if uploaded_video is not None:

        # Save uploaded video temporarily
        input_video = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        input_video.write(
            uploaded_video.read()
        )

        input_video.close()

        st.video(
            input_video.name
        )

        if st.button(
            "▶️ Run Video Detection",
            key="video_detect"
        ):

            cap = cv2.VideoCapture(
                input_video.name
            )

            if not cap.isOpened():

                st.error(
                    "Could not open video."
                )

            else:

                fps = cap.get(
                    cv2.CAP_PROP_FPS
                )

                width = int(
                    cap.get(
                        cv2.CAP_PROP_FRAME_WIDTH
                    )
                )

                height = int(
                    cap.get(
                        cv2.CAP_PROP_FRAME_HEIGHT
                    )
                )

                total_frames = int(
                    cap.get(
                        cv2.CAP_PROP_FRAME_COUNT
                    )
                )

                output_video = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                )

                output_path = output_video.name

                output_video.close()

                fourcc = cv2.VideoWriter_fourcc(
                    *"mp4v"
                )

                writer = cv2.VideoWriter(
                    output_path,
                    fourcc,
                    fps if fps > 0 else 25,
                    (width, height)
                )

                progress_bar = st.progress(0)

                status_text = st.empty()

                preview = st.empty()

                frame_count = 0

                total_processing_time = 0

                while True:

                    ret, frame = cap.read()

                    if not ret:
                        break

                    frame_start = time.perf_counter()

                    results = model.predict(
                        source=frame,
                        imgsz=image_size,
                        conf=confidence,
                        verbose=False
                    )

                    frame_end = time.perf_counter()

                    total_processing_time += (
                        frame_end - frame_start
                    )

                    annotated_frame = results[0].plot()

                    writer.write(
                        annotated_frame
                    )

                    frame_count += 1

                    # Show occasional preview
                    if frame_count % 5 == 0:

                        preview_frame = cv2.cvtColor(
                            annotated_frame,
                            cv2.COLOR_BGR2RGB
                        )

                        preview.image(
                            preview_frame,
                            channels="RGB"
                        )

                    if total_frames > 0:

                        progress = (
                            frame_count /
                            total_frames
                        )

                        progress_bar.progress(
                            min(progress, 1.0)
                        )

                    status_text.write(
                        f"Processing frame "
                        f"{frame_count}/{total_frames}"
                    )

                cap.release()
                writer.release()

                # ------------------------------------------------
                # PERFORMANCE
                # ------------------------------------------------

                if frame_count > 0:

                    avg_latency = (
                        total_processing_time /
                        frame_count
                    )

                    processing_fps = (
                        frame_count /
                        total_processing_time
                    )

                else:

                    avg_latency = 0
                    processing_fps = 0

                st.success(
                    "Video detection completed."
                )

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Frames Processed",
                    frame_count
                )

                col2.metric(
                    "Latency",
                    f"{avg_latency * 1000:.2f} ms/frame"
                )

                col3.metric(
                    "Processing FPS",
                    f"{processing_fps:.2f}"
                )

                st.subheader(
                    "Processed Video"
                )

                st.video(
                    output_path
                )

                # Download button
                with open(
                    output_path,
                    "rb"
                ) as file:

                    st.download_button(
                        label="⬇️ Download Processed Video",
                        data=file,
                        file_name="plate_bowl_detection.mp4",
                        mime="video/mp4"
                    )


# ============================================================
# WEBCAM DETECTION
# ============================================================

with tab3:

    st.header("Webcam Detection")

    st.info(
        "Click Start below to use your browser webcam."
    )

    st.warning(
        "Webcam detection requires streamlit-webrtc."
    )

    try:

        from streamlit_webrtc import (
            webrtc_streamer,
            VideoProcessorBase,
            RTCConfiguration
        )

        import av

        class YOLOVideoProcessor(
            VideoProcessorBase
        ):

            def __init__(self):

                self.model = model

                self.confidence = confidence

                self.image_size = image_size

                self.frame_count = 0

                self.total_time = 0

                self.plate_count = 0

                self.bowl_count = 0

            def recv(self, frame):

                img = frame.to_ndarray(
                    format="bgr24"
                )

                start = time.perf_counter()

                results = self.model.predict(
                    source=img,
                    imgsz=self.image_size,
                    conf=self.confidence,
                    verbose=False
                )

                end = time.perf_counter()

                self.total_time += (
                    end - start
                )

                self.frame_count += 1

                result = results[0]

                # Count detections
                self.plate_count = 0
                self.bowl_count = 0

                if result.boxes is not None:

                    for box in result.boxes:

                        class_id = int(
                            box.cls[0].item()
                        )

                        class_name = (
                            result.names[class_id]
                        )

                        if class_name.lower() == "plate":

                            self.plate_count += 1

                        elif class_name.lower() == "bowl":

                            self.bowl_count += 1

                annotated = result.plot()

                # FPS
                if self.total_time > 0:

                    fps = (
                        self.frame_count /
                        self.total_time
                    )

                else:

                    fps = 0

                cv2.putText(
                    annotated,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    annotated,
                    f"Plates: {self.plate_count}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )

                cv2.putText(
                    annotated,
                    f"Bowls: {self.bowl_count}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )

                return av.VideoFrame.from_ndarray(
                    annotated,
                    format="bgr24"
                )

        webrtc_streamer(
            key="plate-bowl-webcam",
            video_processor_factory=YOLOVideoProcessor,
            media_stream_constraints={
                "video": True,
                "audio": False
            },
            async_processing=True
        )

    except ImportError:

        st.error(
            "streamlit-webrtc is not installed."
        )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    f"Model: {model_name} | "
    f"YOLO image size: {image_size} | "
    f"Confidence threshold: {confidence}"
)