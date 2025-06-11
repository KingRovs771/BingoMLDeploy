from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app, origins=["*"], supports_credentials=True)

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response

# Load model
model = load_model('best_model.h5')

class_names = [
    'Baterai', 'Daun', 'Elektronik', 'Kaca', 'Kardus',
    'Kertas', 'Lampu', 'Logam', 'Pakaian',
    'Plastik', 'Sampah Makanan', 'Sterofom'
]

group_map = {
    'Logam': 'Anorganik',
    'Plastik': 'Anorganik',
    'Pakaian': 'Anorganik',
    'Kaca': 'Anorganik',
    'Sterofom': 'Anorganik',
    'Daun': 'Organik',
    'Kardus': 'Organik',
    'Sampah Makanan': 'Organik',
    'Kertas': 'Organik',
    'Baterai': 'B3',
    'Lampu': 'B3',
    'Elektronik': 'B3'
}

def prepare_image_from_bytes(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def index():
    return 'API is running!'

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    img_bytes = file.read()
    img_ready = prepare_image_from_bytes(img_bytes)

    preds = model.predict(img_ready)
    pred_idx = np.argmax(preds, axis=1)[0]
    pred_class = class_names[pred_idx]
    pred_group = group_map.get(pred_class, 'Unknown')

    return jsonify({
        'Sampah': pred_class,
        'Kategori': pred_group
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)