import os
# Force legacy Keras behavior to handle older .h5 files
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# ==========================================
# ✅ COMPATIBILITY LAYER FOR AWS DEPLOYMENT
# ==========================================
# These classes tell TensorFlow to ignore the new arguments (batch_shape, quantization_config)
# that were likely created in your local training environment.

class CompatibleInput(tf.keras.layers.InputLayer):
    def __init__(self, *args, **kwargs):
        kwargs.pop('batch_shape', None)
        kwargs.pop('optional', None)
        super().__init__(*args, **kwargs)

class CompatibleDense(tf.keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)

app = Flask(__name__)

# ==========================================
# ✅ MODEL LOADING
# ==========================================
try:
    print("Loading model... please wait.")
    model = tf.keras.models.load_model(
        "brain_tumor_model.h5",
        custom_objects={
            'InputLayer': CompatibleInput,
            'Dense': CompatibleDense
        },
        compile=False
    )
    print("✅ SUCCESS: Model loaded successfully!")
except Exception as e:
    print(f"❌ ERROR: Model failed to load. Details: {e}")
    model = None

# Class labels for your brain tumor model
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

def preprocess_image(image_bytes):
    """Resize to 256x256 RGB and normalize to [0, 1]"""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((256, 256))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# ==========================================
# ✅ ROUTES
# ==========================================

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "model_loaded": model is not None
    })

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
        # Preprocess and Predict
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

# ==========================================
# ✅ START SERVER
# ==========================================
if __name__ == "__main__":
    # host='0.0.0.0' is required for AWS Public IP access
    app.run(host="0.0.0.0", port=5000, debug=True)