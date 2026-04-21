import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import streamlit as st

from config import *
from models.resnet18 import get_resnet18


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Road Quality Assessment",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# -----------------------------
# Load model (cached)
# -----------------------------
@st.cache_resource
def load_model():
    model = get_resnet18(NUM_CLASSES)

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE_DIR, "model.pth")

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )
    model.to(DEVICE)
    model.eval()
    return model


model = load_model()


# -----------------------------
# Image preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])


def predict_batch(images):
    results = []

    for img in images:
        image = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(image)
            probs = F.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)

        results.append((CLASS_NAMES[pred.item()], conf.item()))

    return results

st.markdown("""
### 🧾 Classification Legend
- ✅ **Good** – Road is in good condition  
- ℹ️ **Satisfactory** – Minor wear, monitoring recommended  
- ⚠️ **Poor** – Repair recommended  
- 🚨 **Very Poor** – Immediate repair required  
""")


# -----------------------------
# UI
# -----------------------------
st.title("🚧 Road Quality Assessment System")
st.write("Upload one or more road images to assess their condition.")

uploaded_files = st.file_uploader(
    "📤 Upload road images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    st.subheader("🖼️ Uploaded Images")

    images = []
    for file in uploaded_files:
        img = Image.open(file).convert("RGB")
        images.append(img)
        st.image(img, caption=file.name, width=600)

    # -----------------------------
    # Predict button + loading
    # -----------------------------
    if st.button("🔍 Predict All Images"):
        with st.spinner("Analyzing road conditions... Please wait"):
            results = predict_batch(images)

        st.subheader("☑️ Prediction Results")

        # -----------------------------
        # Display results
        # -----------------------------
        for idx, (label, confidence) in enumerate(results):
            st.markdown(f"### 📌 Image {idx + 1}")
            st.write(f"**Condition:** {label}")
            st.write(f"**Confidence:** {confidence * 100:.2f}%")

            if label == "very_poor":
                st.error("🚨 Very poor road condition detected! Immediate repair required.")

            elif label == "poor":
                st.warning("⚠️ Poor road condition detected. Maintenance is recommended.")

            elif label == "satisfactory":
                st.info("ℹ️ Satisfactory road condition. Monitoring is advised.")

            elif label == "good":
                st.success("✅ Good road condition. No action required.")
