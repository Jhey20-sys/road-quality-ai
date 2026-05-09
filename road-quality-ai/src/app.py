import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import streamlit as st

from config import *
from models.mobilenetv2 import get_mobilenetv2

# Page config

st.set_page_config(
    page_title="Road Quality Assessment",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Load model (cached)

@st.cache_resource
def load_model():
    MODELS_DIR = "trained_models"
    WEIGHTS_FILENAME = "mobilenetv2_best.pth"
    REL_PATH = os.path.join(MODELS_DIR, WEIGHTS_FILENAME)

    # Walk up from app.py's directory until we find trained_models/mobilenetv2_best.pth
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [here]
    current = here
    for _ in range(4):
        current = os.path.dirname(current)
        candidates.append(current)

    MODEL_PATH = None
    for d in candidates:
        # Primary: look in <candidate>/trained_models/<weights>
        candidate_path = os.path.join(d, REL_PATH)
        if os.path.exists(candidate_path):
            MODEL_PATH = candidate_path
            break
        # Fallback: also check <candidate>/<weights> directly (legacy layout)
        legacy_path = os.path.join(d, WEIGHTS_FILENAME)
        if os.path.exists(legacy_path):
            MODEL_PATH = legacy_path
            break

    if MODEL_PATH is None:
        st.error(
            f"❌ Could not find `{REL_PATH}` (or legacy `{WEIGHTS_FILENAME}`) "
            "in any of these directories:\n\n"
            + "\n".join(f"- `{d}`" for d in candidates)
        )
        st.stop()

    model = get_mobilenetv2(NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model
model = load_model()

# Image preprocessing

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
### 📋 Classification Legend
- 🥳 **Good** – Road is in good condition  
- ℹ️ **Satisfactory** – Minor wear, monitoring recommended  
- ⚠️ **Poor** – Repair recommended  
- 🚨 **Very Poor** – Immediate repair required  
""")

# UI

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

    if images and st.button("🔍 Predict All Images"):
        with st.spinner("Analyzing road conditions... Please wait"):
            results = predict_batch(images)

        st.subheader("☑️ Prediction Results")

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
                st.success("🥳 Good road condition. No action required.")