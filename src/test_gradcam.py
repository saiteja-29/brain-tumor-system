import numpy as np
import cv2
from keras.models import load_model
from PIL import Image
from .gradcam import get_gradcam, overlay_heatmap

MODEL_PATH = "models/model.h5"
IMAGE_PATH = "data/processed/test/tumor/G_92.jpg"  # change this

IMG_SIZE = (224, 224)

def preprocess(image):
    image = image.resize(IMG_SIZE)
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

def main():

    model = load_model(MODEL_PATH)

    image = Image.open(IMAGE_PATH).convert("RGB")
    img_array = preprocess(image)

    # 🔥 IMPORTANT: last conv layer name for EfficientNet
    last_conv_layer = "top_conv"

    heatmap = get_gradcam(model, img_array, last_conv_layer)

    original = cv2.imread(IMAGE_PATH)
    overlay = overlay_heatmap(heatmap, original)

    cv2.imwrite("gradcam_output.jpg", overlay)

    print("Grad-CAM saved as gradcam_output.jpg")

if __name__ == "__main__":
    main()