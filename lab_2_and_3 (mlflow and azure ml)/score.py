import json
import numpy as np
import mlflow
import mlflow.tensorflow
import os

# Global model object
model = None


def init():
    global model
    model_dir = os.environ.get("AZUREML_MODEL_DIR")  # AzureML sets this automatically
    model_uri = os.path.join(model_dir, "model")
    model = mlflow.tensorflow.load_model(model_uri)
    print("Model loaded from:", model_uri)


def run(raw_data):
    """
    Perform inference on input data.

    Expected input (JSON string):

        {
          "input": [
            [ [ [r,g,b], [r,g,b], ... ], ... ],  # image 1, shape (32,32,3)
            [ [ [r,g,b], ... ], ... ]           # image 2, ...
          ]
        }

    That is, "input" is a list of images, each a nested list of shape (32,32,3),
    with pixel values either in [0, 255] or already normalized to [0, 1].

    Returns JSON with predicted class indices and (optionally) probabilities:

        {
          "predicted_classes": [int, int, ...],
          "probabilities": [[float, ...], ...]
        }
    """
    try:
        # Parse input JSON
        if isinstance(raw_data, str):
            data = json.loads(raw_data)
        else:
            # Some hosting environments may pass a dict already
            data = raw_data

        images = data.get("input", None)
        if images is None:
            raise ValueError("JSON payload must contain an 'input' field.")

        # Convert to NumPy array
        x = np.array(images, dtype="float32")

        # If images seem to be in 0–255 range, normalize to 0–1
        if x.max() > 1.0:
            x = x / 255.0

        # Run prediction
        preds_prob = model.predict(x)
        preds_class = np.argmax(preds_prob, axis=1)

        # Build response
        result = {
            "predicted_classes": preds_class.tolist(),
            "probabilities": preds_prob.tolist(),
        }

        return json.dumps(result)

    except Exception as e:
        error = str(e)
        print("Error during inference:", error)
        return json.dumps({"error": error})