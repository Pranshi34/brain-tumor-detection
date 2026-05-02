from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# ==========================================
# ✅ CLEAN MODEL LOADING (Keras 3)
# ==========================================
try:
    print("Loading Keras 3 model... please wait.")
    # In Keras 3, we don't need custom_objects for standard layers
    model = tf.keras.models.load_model(
        "brain_tumor_model.h5",
        compile=False
    )
    print("✅ SUCCESS: Model loaded successfully!")
except Exception as e:
    print(f"❌ ERROR: Still failing. Details: {e}")
    model = None

CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((256, 256))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        img_tensor = preprocess_image(file.read())
        preds = model.predict(img_tensor, verbose=0)[0]
        predicted_index = int(np.argmax(preds))
        
        return jsonify({
            "prediction": CLASS_NAMES[predicted_index],
            "confidence": round(float(preds[predicted_index]) * 100, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Ensure port 5000 is open in AWS Security Groups!
    app.run(host="0.0.0.0", port=5000, debug=True)