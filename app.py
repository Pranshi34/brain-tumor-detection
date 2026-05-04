from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

MODEL_PATH = "brain_tumor_model.h5"
model = load_model(MODEL_PATH, compile=False, safe_mode=False)

# Print model input shape for debugging
print("Model input shape:", model.input_shape)

IMG_HEIGHT = 128
IMG_WIDTH = 128

labels = ["glioma", "meningioma", "notumor", "pituitary"]

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
        file = request.files["file"]

        upload_folder = os.path.join("static", "uploads")
        os.makedirs(upload_folder, exist_ok=True)

        filepath = os.path.join(upload_folder, file.filename)
        file.save(filepath)
        

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
        return f"Error: {str(e)}"


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)