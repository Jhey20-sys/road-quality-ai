import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from config import *
#from models.custom_cnn import CustomCNN
from models.resnet18 import get_resnet18


# -------------------------------------------------
# Load model once
# -------------------------------------------------
def load_model():
    #model = CustomCNN(NUM_CLASSES)
    model = get_resnet18(NUM_CLASSES)

    model.load_state_dict(
        torch.load("model.pth", map_location=DEVICE)
    )
    model.to(DEVICE)
    model.eval()
    return model


# -------------------------------------------------
# Image preprocessing
# -------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])


# -------------------------------------------------
# Predict single image
# -------------------------------------------------
def predict_image(model, image_path):

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)
        conf, predicted = torch.max(probs, 1)

    return CLASS_NAMES[predicted.item()], conf.item()


# -------------------------------------------------
# Predict all images in a folder (real-life simulation)
# -------------------------------------------------
def predict_folder(folder_path):
    model = load_model()

    print("\n🚦 ROAD CONDITION ASSESSMENT RESULTS\n")

    for img in os.listdir(folder_path):
        if img.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(folder_path, img)
            label, confidence = predict_image(model, img_path)

            print(f"🖼 {img}")
            print(f"   ➜ Condition : {label}")
            print(f"   ➜ Confidence: {confidence * 100:.2f}%\n")


# -------------------------------------------------
# Main execution
# -------------------------------------------------
if __name__ == "__main__":

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    SAMPLE_FOLDER = os.path.join(BASE_DIR, "samples_for_prediction")

    predict_folder(SAMPLE_FOLDER)
