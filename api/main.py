from fastapi import FastAPI, UploadFile, File
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64
import cv2

from src.gradcam import get_gradcam, overlay_heatmap

app = FastAPI()

# Load model once (IMPORTANT)
model = tf.keras.models.load_model("models/model.h5")

IMG_SIZE = (224, 224)
CLASS_NAMES = ["no_tumor", "tumor"]
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
def preprocess(image):
    image = image.resize(IMG_SIZE)
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

@app.get("/")
def home():
    return {"message": "Brain Tumor Detection API is running 🚀"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    
    # Read image
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    img_array = preprocess(image)

    # Prediction
    preds = model.predict(img_array)
    class_idx = int(np.argmax(preds))
    confidence = float(np.max(preds))

    # 🔥 Grad-CAM
    last_conv_layer = "top_conv"
    heatmap = get_gradcam(model, img_array, last_conv_layer)

    original = np.array(image)
    overlay = overlay_heatmap(heatmap, original)

    # Convert image to base64 (for API response)
    _, buffer = cv2.imencode('.jpg', overlay)
    heatmap_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "prediction": CLASS_NAMES[class_idx],
        "confidence": confidence,
        "heatmap": heatmap_base64
    }