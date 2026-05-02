import os
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# ==========================================
# ✅ THE SLEDGEHAMMER COMPATIBILITY FIX
# ==========================================
def fixed_load_model(model_path):
    """
    Manually strips problematic Keras 3 keywords from the model config 
    to allow older TensorFlow versions to load it.
    """
    try:
        # 1. First attempt a standard load
        return tf.keras.models.load_model(model_path, compile=False)
    except TypeError as e:
        print(f"Standard load failed, attempting metadata surgery... Error: {e}")
        
        # 2. If it fails, we use this bypass:
        import h5py
        from tensorflow.keras.layers import deserialize_keras_object
        
        # This prevents the specific 'quantization_config' and 'batch_shape' errors
        def custom_objects_dict():
            from tensorflow.keras.layers import Dense, InputLayer
            class CleanDense(Dense):
                def __init__(self, *args, **kwargs):
                    kwargs.pop('quantization_config', None)
                    super().__init__(*args, **kwargs)
            class CleanInput(InputLayer):
                def __init__(self, *args, **kwargs):
                    kwargs.pop('batch_shape', None)
                    super().__init__(*args, **kwargs)
            return {'Dense': CleanDense, 'InputLayer': CleanInput}

        return tf.keras.models.load_model(
            model_path, 
            custom_objects=custom_objects_dict(), 
            compile=False
        )

# LOAD MODEL
try:
    model = fixed_load_model("brain_tumor_model.h5")
    print("✅ SUCCESS: Model is finally loaded!")
except Exception as e:
    print(f"❌ CRITICAL ERROR: Could not load model even with surgery: {e}")
    model = None

# ==========================================
# ✅ APP ROUTES
# ==========================================

CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

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
        img = Image.open(io.BytesIO(file.read())).convert("RGB").resize((256, 256))
        img_tensor = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)
        
        preds = model.predict(img_tensor, verbose=0)[0]
        idx = int(np.argmax(preds))
        
        return jsonify({
            "prediction": CLASS_NAMES[idx],
            "confidence": round(float(preds[idx]) * 100, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)