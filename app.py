from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import logging

app = Flask(__name__)

# Configuration for production and development
DEBUG = os.getenv("FLASK_DEBUG", "False").lower() == "true"
PORT = int(os.getenv("PORT", 5000))
HOST = os.getenv("HOST", "0.0.0.0")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model path with fallback
MODEL_PATH = os.getenv("MODEL_PATH", "brain_tumor_model.h5")
if not os.path.exists(MODEL_PATH):
    logger.error(f"Model file not found at {MODEL_PATH}")
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model = load_model(MODEL_PATH, compile=False)
logger.info(f"Model loaded from {MODEL_PATH}")


logger.info(f"Model input shape: {model.input_shape}")

IMG_HEIGHT = 128
IMG_WIDTH = 128

labels = ["glioma", "meningioma", "notumor", "pituitary"]

# Upload folder configuration
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def prepare_image(path):
    img = image.load_img(path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img)
    
    # Normalize to [0, 1] range
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return "Error: No file provided", 400
        
        file = request.files["file"]
        if file.filename == "":
            return "Error: Empty filename", 400

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        logger.info(f"File saved to {filepath}")

        img = prepare_image(filepath)

        preds = model.predict(img)

        predicted_class = labels[np.argmax(preds)]
        confidence = float(np.max(preds))

        return render_template(
            "index.html",
            prediction=predicted_class,
            confidence=f"{confidence*100:.2f}",
            img_path = filepath.replace("\\", "/")
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return f"Error: {str(e)}", 500


if __name__ == "__main__":
    logger.info(f"Starting Flask app on {HOST}:{PORT} (debug={DEBUG})")
    app.run(host=HOST, port=PORT, debug=DEBUG)