from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import tf_keras

app = Flask(__name__)
model = tf_keras.models.load_model("brain_tumor_model.h5")

class_name = ['glioma', 'meningioma', 'notumor', 'pituitary']

UPLOAD_FOLDER = "static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def predict_image(path):
    img = load_img(path, target_size=(128,128))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    index = np.argmax(pred)
    return class_name[index]


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]

        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            result = predict_image(filepath)

            return render_template("index.html", result=result, image=filepath)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)