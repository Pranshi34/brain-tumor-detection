from flask import Flask, render_template, request
from tensorflow.modelskeras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)

# Load model (fixed)
model = load_model("brain_tumor_model.h5", compile=False)

class_name = ['glioma', 'meningioma', 'notumor', 'pituitary']

UPLOAD_FOLDER = "static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Prediction function
def predict_image(path):
    try:
        img = load_img(path, target_size=(256, 256))  # adjust if needed
        img = img_to_array(img)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img)
        index = np.argmax(pred)

        return class_name[index]
    
    except Exception as e:
        return str(e)

# Route
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")

        if file and file.filename != "":
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            result = predict_image(filepath)

            return render_template("index.html", result=result, image=filepath)

    return render_template("index.html", result=None)

# Run app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)