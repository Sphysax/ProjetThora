import base64
from io import BytesIO
from pathlib import Path

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

from train_v1_common import (
    BEST_MODEL_PATH,
    DEVICE,
    model,
    train_dataset,
    valid_transforms,
)

st.set_page_config(
    page_title="Pneumonia X-ray Classifier",
    page_icon="🩻",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_model():
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {BEST_MODEL_PATH}")
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def preprocess_image(uploaded_file) -> Image.Image:
    image = Image.open(uploaded_file).convert("RGB")
    return image


def predict_image(image: Image.Image) -> dict:
    model = load_model()
    tensor = valid_transforms(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1).squeeze(0)
        pred_idx = torch.argmax(probs).item()

    classes = train_dataset.classes
    predicted_class = classes[pred_idx]
    probabilities = {label: float(probs[i].item()) for i, label in enumerate(classes)}

    return {
        "predicted_class": predicted_class,
        "confidence": probabilities[predicted_class],
        "probabilities": probabilities,
    }


def image_to_base64(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


st.title("Chest X-ray Pneumonia Classifier")
st.write("Upload an X-ray image and the model will predict whether it shows pneumonia or not.")

uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])

if uploaded_file is not None:
    image = preprocess_image(uploaded_file)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Prediction")
        try:
            result = predict_image(image)
            st.success(f"Predicted class: {result['predicted_class']}")
            st.write(f"Confidence: {result['confidence']:.4f}")

            st.progress(float(result["confidence"]))

            st.subheader("Class probabilities")
            for label, prob in result["probabilities"].items():
                st.write(f"**{label}**: {prob:.6f}")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

    st.subheader("Raw image preview")
    st.code(uploaded_file.name)
else:
    st.info("Please upload an X-ray image to begin.")