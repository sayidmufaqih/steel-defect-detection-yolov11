import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Industrial Defect Detection",
    page_icon="🔍",
    layout="wide"
)


# =========================
# LOAD MODEL
# =========================

@st.cache_resource
def load_model():
    model = YOLO("models/best.pt")
    return model


model = load_model()


# =========================
# TITLE & DESCRIPTION
# =========================

st.title("🔍 Industrial Surface Defect Detection")

st.write(
    "Upload an industrial steel surface image and "
    "the YOLOv11 model will detect and classify surface defects."
)


# =========================
# SIDEBAR
# =========================

st.sidebar.header("Detection Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.10,
    step=0.05
)

st.sidebar.write(
    f"Current threshold: **{confidence:.2f}**"
)


# =========================
# IMAGE UPLOAD
# =========================

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


# =========================
# REAL-TIME DETECTION
# =========================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    # Run YOLO direct from PIL Image (No need for tempfile)
    with st.spinner("Detecting defects..."):
        results = model.predict(
            source=image,
            conf=confidence,
            verbose=False
        )

    result = results[0]

    # Plot detection result (RGB format for Streamlit display)
    annotated_image = result.plot()  # returns BGR numpy array
    annotated_image_rgb = annotated_image[..., ::-1]  # Convert BGR to RGB

    with col2:
        st.subheader("Detection Result")
        st.image(
            annotated_image_rgb,
            use_container_width=True
        )


    # =========================
    # DETECTION SUMMARY
    # =========================

    st.subheader("Detection Summary")

    boxes = result.boxes

    if boxes is not None and len(boxes) > 0:

        detected_classes = []

        for box in boxes:
            class_id = int(box.cls[0])
            confidence_score = float(box.conf[0])
            class_name = model.names[class_id]

            detected_classes.append({
                "Defect": class_name,
                "Confidence": confidence_score
            })

        # Display detection details
        for detection in detected_classes:
            st.write(
                f"• **{detection['Defect']}** — Confidence: `{detection['Confidence']:.2%}`"
            )

        st.success(
            f"Total detected: **{len(detected_classes)} defect(s)**."
        )

    else:

        st.warning(
            "No defects detected above the selected confidence threshold."
        )