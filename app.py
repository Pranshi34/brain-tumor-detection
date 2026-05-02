from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load model once at startup
model = tf.keras.models.load_model("brain_tumor_model.h5")

# Class labels
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]


def preprocess_image(image_bytes):
    """Resize to 256x256 RGB and normalize"""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((256, 256))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# ✅ HOME ROUTE (shows HTML UI)
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


# ✅ OPTIONAL: health check API
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": "Brain Tumor Classifier",
        "classes": CLASS_NAMES
    })


# ✅ PREDICTION ROUTE
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        img_tensor = preprocess_image(file.read())
        preds = model.predict(img_tensor)[0]

        predicted_index = int(np.argmax(preds))
        confidence = float(preds[predicted_index])

        return jsonify({
            "prediction": CLASS_NAMES[predicted_index],
            "confidence": round(confidence * 100, 2),
            "all_probabilities": {
                CLASS_NAMES[i]: round(float(preds[i]) * 100, 2)
                for i in range(len(CLASS_NAMES))
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# RUN APP
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)