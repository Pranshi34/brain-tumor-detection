from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io
# Import the specific layers causing the error to pass them into custom_objects
from tensorflow.keras.layers import Dense, InputLayer

app = Flask(__name__)

# ==========================================
# ✅ VERSION COMPATIBILITY FIX
# ==========================================
# This tells Keras to use the standard layers even if it sees 
# new arguments like 'batch_shape' or 'quantization_config'
custom_objects = {
    'Dense': Dense,
    'InputLayer': InputLayer
}

try:
    # We add custom_objects here to ignore the metadata it doesn't understand
    model = tf.keras.models.load_model(
        "brain_tumor_model.h5",
        custom_objects=custom_objects,
        compile=False
    )
    print("✅ SUCCESS: Model loaded successfully!")
except Exception as e:
    print("❌ ERROR: Model loading failed still. Error details:", e)
    model = None

# Class labels
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

def preprocess_image(image_bytes):
    """Resize to 256x256 RGB and normalize"""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((256, 256))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# HOME ROUTE
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# HEALTH CHECK
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "classes": CLASS_NAMES
    })

# PREDICTION ROUTE
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded on server"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        img_tensor = preprocess_image(file.read())
        preds = model.predict(img_tensor, verbose=0)[0]

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

# RUN SERVER
if __name__ == "__main__":
    # Ensure host is 0.0.0.0 for AWS visibility
    app.run(host="0.0.0.0", port=5000, debug=True)