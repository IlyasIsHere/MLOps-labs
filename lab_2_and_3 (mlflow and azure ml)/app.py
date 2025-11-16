from flask import Flask, render_template_string, request
import io
import json
import numpy as np
import requests
from PIL import Image

# Paste your ACI scoring URI here
ACI_SCORING_URI = "http://1d9f7c07-d8ae-40c0-ba42-2ccb8f6e2e8b.spaincentral.azurecontainer.io/score"

app = Flask(__name__)

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

TEMPLATE = """
<!doctype html>
<title>CIFAR-10 Predictor</title>
<h1>CIFAR-10 Image Upload</h1>
<form method="post" enctype="multipart/form-data">
  <p><input type="file" name="file" accept="image/*" required>
  <button type="submit">Predict</button></p>
</form>
{% if error %}
  <p style="color:red;">{{ error }}</p>
{% endif %}
{% if prediction is not none %}
  <h2>Prediction</h2>
  <p>Predicted class: {{ prediction }}</p>
{% endif %}
"""

def preprocess_image(file_storage):
    img = Image.open(file_storage.stream).convert("RGB")
    img = img.resize((32, 32))  # CIFAR-10 size
    arr = np.array(img).astype("float32") / 255.0
    arr = arr.reshape(1, 32, 32, 3)  # add batch dimension
    return arr

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    if request.method == "POST":
        if "file" not in request.files or request.files["file"].filename == "":
            error = "Please choose an image file."
            return render_template_string(TEMPLATE, prediction=None, raw=None, error=error)

        file = request.files["file"]
        try:
            x_sample = preprocess_image(file)
        except Exception as e:
            error = f"Could not read image: {e}"
            return render_template_string(TEMPLATE, prediction=None, raw=None, error=error)

        payload = {"input": x_sample.tolist()}

        try:
            resp = requests.post(ACI_SCORING_URI, json=payload)
            resp.raise_for_status()
            data = resp.json()

            # Endpoint returns a JSON-encoded string with "predicted_classes" list
            parsed = json.loads(data)
            class_idx = int(parsed.get("predicted_classes")[0])
            prediction = f"{class_idx} ({CLASS_NAMES[class_idx]})"
        except Exception as e:
            error = f"Error calling endpoint: {e}"

    return render_template_string(TEMPLATE, prediction=prediction, error=error)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)