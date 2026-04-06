import os

import gdown
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model


MODEL_NAME = "best_model_phase1.keras"
GDRIVE_URL = "https://drive.google.com/file/d/1Pdo2SjtoEpoIVgRBkti3iFXJrF_5vyCA/view?usp=sharing"
CLASS_NAMES = [
    "Fresh Apples",
    "Fresh Banana",
    "Fresh Oranges",
    "Rotten Apples",
    "Rotten Banana",
    "Rotten Oranges",
]


st.set_page_config(
    page_title="FreshScan - Fruit Quality AI",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Manrope:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Manrope', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 20%, rgba(247, 148, 29, 0.20), transparent 35%),
        radial-gradient(circle at 90% 10%, rgba(46, 196, 182, 0.18), transparent 35%),
        linear-gradient(180deg, #f8fafc 0%, #eef2ff 50%, #f8fafc 100%);
}

.hero {
    background: rgba(255, 255, 255, 0.75);
    border: 1px solid rgba(15, 23, 42, 0.08);
    backdrop-filter: blur(6px);
    border-radius: 18px;
    padding: 24px;
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
    margin-bottom: 14px;
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    line-height: 1.2;
    color: #0f172a;
}

.hero-sub {
    margin-top: 8px;
    color: #334155;
    font-size: 1rem;
}

.metric-card {
    border: 1px solid rgba(15, 23, 42, 0.08);
    border-radius: 14px;
    padding: 14px 16px;
    background: rgba(255, 255, 255, 0.85);
    margin-bottom: 12px;
}

.metric-title {
    color: #475569;
    font-size: 0.9rem;
}

.metric-value {
    color: #0f172a;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
}
</style>
""",
    unsafe_allow_html=True,
)


def download_from_gdrive(url: str, output: str) -> bool:
    """Download model from Google Drive when not present in deployment."""
    try:
        if "drive.google.com/file/d/" in url:
            file_id = url.split("/d/")[1].split("/")[0]
            direct_url = f"https://drive.google.com/uc?id={file_id}"
        else:
            direct_url = url
        gdown.download(direct_url, output, quiet=True)
        return True
    except Exception:
        return False


@st.cache_resource
def load_fruit_model():
    if not os.path.exists(MODEL_NAME):
        return None
    try:
        return load_model(MODEL_NAME)
    except Exception:
        return None


def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    preview = image.copy()
    model_image = image.resize((224, 224))
    image_array = np.asarray(model_image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return preview, image_array


def get_top_predictions(probabilities: np.ndarray):
    probs = probabilities.flatten()
    sorted_indices = np.argsort(probs)[::-1]
    return [(idx, float(probs[idx])) for idx in sorted_indices]


st.markdown(
    """
<div class="hero">
  <div class="hero-title">FreshScan | Fruit Quality Inspector</div>
  <div class="hero-sub">Upload one fruit image and get an instant AI decision: fresh or rotten, with confidence scores.</div>
</div>
""",
    unsafe_allow_html=True,
)


if not os.path.exists(MODEL_NAME):
    with st.spinner("Model file missing. Downloading model..."):
        if download_from_gdrive(GDRIVE_URL, MODEL_NAME):
            st.success("Model downloaded successfully.")
        else:
            st.error("Model file not found and download failed.")

model = load_fruit_model()

with st.sidebar:
    st.header("System")
    st.caption("FreshScan deployment status")
    model_ready = model is not None
    st.metric("Model status", "Ready" if model_ready else "Missing")
    if os.path.exists(MODEL_NAME):
        size_mb = os.path.getsize(MODEL_NAME) / (1024 * 1024)
        st.metric("Model size", f"{size_mb:.2f} MB")
    st.info("Supported classes: 6")

if model is None:
    st.stop()


left_col, right_col = st.columns([1.05, 1], gap="large")

with left_col:
    st.subheader("Image Upload")
    uploaded = st.file_uploader(
        "Drop an image (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        help="Use a clear single-fruit image for best accuracy.",
    )

    if uploaded is not None:
        preview_image, model_input = preprocess_image(uploaded)
        st.image(preview_image, caption="Input image", use_container_width=True)
    else:
        st.markdown(
            """
<div class="metric-card">
  <div class="metric-title">Tip</div>
  <div>Center the fruit, keep background simple, and avoid blur for a cleaner prediction.</div>
</div>
""",
            unsafe_allow_html=True,
        )

with right_col:
    st.subheader("Prediction")

    if uploaded is not None:
        with st.spinner("Analyzing image..."):
            prediction = model.predict(model_input, verbose=0)[0]

        top_predictions = get_top_predictions(prediction)
        best_idx, best_score = top_predictions[0]
        best_label = CLASS_NAMES[best_idx] if best_idx < len(CLASS_NAMES) else str(best_idx)

        st.markdown(
            f"""
<div class="metric-card">
  <div class="metric-title">Predicted class</div>
  <div class="metric-value">{best_label}</div>
</div>
<div class="metric-card">
  <div class="metric-title">Confidence</div>
  <div class="metric-value">{best_score * 100:.2f}%</div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.write("Top predictions")
        for class_idx, score in top_predictions[:3]:
            class_name = CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else str(class_idx)
            st.write(f"{class_name}: {score * 100:.2f}%")
            st.progress(min(max(score, 0.0), 1.0))

        with st.expander("Show raw probabilities"):
            st.write(prediction)
    else:
        st.markdown(
            """
<div class="metric-card">
  <div class="metric-title">Awaiting image</div>
  <div>Upload a fruit image to get a real-time quality classification.</div>
</div>
""",
            unsafe_allow_html=True,
        )