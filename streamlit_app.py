import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path

from train_v1_common import BEST_MODEL_PATH, DEVICE, model, train_dataset, valid_transforms

st.set_page_config(
    page_title="Pneumonia X-ray Classifier",
    page_icon="🩻",
    layout="wide",
)

st.title("Chest X-ray Pneumonia Classifier")
st.write("Upload an X-ray image and the model will predict whether it shows pneumonia or not.")


@st.cache_resource
def load_model():
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {BEST_MODEL_PATH}")
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    return image


def predict_image(image: Image.Image):
    loaded_model = load_model()
    tensor = valid_transforms(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = loaded_model(tensor)
        probs = F.softmax(outputs, dim=1).squeeze(0)
        pred_idx = torch.argmax(probs).item()

    classes = train_dataset.classes
    predicted_class = classes[pred_idx]
    probabilities = {label: float(probs[i].item()) for i, label in enumerate(classes)}

    return predicted_class, probabilities


uploaded_file = st.file_uploader(
    "Choose an X-ray image",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
)

if uploaded_file is not None:
    image = preprocess_image(uploaded_file)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Prediction")
        try:
            predicted_class, probabilities = predict_image(image)
            confidence = probabilities[predicted_class]

            st.success(f"Predicted class: {predicted_class}")
            st.write(f"Confidence: {confidence:.4f}")
            st.progress(float(confidence))

            st.subheader("Class probabilities")
            for label, prob in probabilities.items():
                st.write(f"**{label}**: {prob:.6f}")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
else:
    st.info("Please upload an X-ray image to begin.")