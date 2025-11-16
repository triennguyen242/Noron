import os
import base64
import numpy as np
from flask import Flask, request, jsonify
import tensorflow.lite as tflite
from PIL import Image
import io

app = Flask(__name__)

# Lấy port theo yêu cầu của Render/Fly.io
port = int(os.environ.get("PORT", 10000))

# Đường dẫn model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "mobilenetv5.tflite")

# Load model TFLite
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Hàm tiền xử lý ảnh
def preprocess_image(image_base64):
    img_data = base64.b64decode(image_base64)
    img = Image.open(io.BytesIO(img_data)).convert("RGB")

    # Resize theo yêu cầu model
    target_w = input_details[0]['shape'][1]
    target_h = input_details[0]['shape'][2]

    img = img.resize((target_w, target_h))
    img = np.array(img).astype(np.float32) / 255.0

    img = np.expand_dims(img, axis=0)
    return img

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        img_base64 = data.get("image")

        input_tensor = preprocess_image(img_base64)

        interpreter.set_tensor(input_details[0]["index"], input_tensor)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]["index"])
        prob = float(output[0][0])

        result = "anemia" if prob > 0.5 else "normal"

        return jsonify({"result": result, "confidence": prob})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port)
