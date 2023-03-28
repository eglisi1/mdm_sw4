import os
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import onnxruntime as rt
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)


def load_onnx_model(model_path: str) -> rt.InferenceSession:
    sess = rt.InferenceSession(model_path)
    return sess


def prepare_input_data(image_data: str) -> np.ndarray:
    img = Image.open(BytesIO(base64.b64decode(image_data))).convert("RGBA")
    img_resized = img.resize((28, 28))

    # Create a new white background image
    background = Image.new("RGBA", img_resized.size, (255, 255, 255))

    # Paste the original image onto the white background
    combined_image = Image.alpha_composite(background, img_resized)

    # Convert the combined image to grayscale
    img_gray = combined_image.convert("L")

    img_array = 1 - (np.array(img_gray) / 255.0)
    img_expanded = img_array[np.newaxis, np.newaxis, :, :].astype(np.float32)
    return img_expanded


def run_onnx_model(session: rt.InferenceSession, input_data: np.ndarray) -> list:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: input_data})[0]
    return result.tolist()


@app.route("/", methods=["GET"])
def index() -> str:
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict() -> str:
    if request.method == "POST":
        input_data = request.form["image"]
        input_data = input_data.split(",")[1]
        prepared_data = prepare_input_data(input_data)
        prediction = run_onnx_model(model_session, prepared_data)
        print(prediction)
        predicted_digit = np.argmax(prediction)
        response = {"prediction": int(predicted_digit)}
        return jsonify(response)


MODEL_PATH = "model/mnist-12-int8.onnx"
model_session = load_onnx_model(MODEL_PATH)

if __name__ == "__main__":
    app.run()
