from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import numpy as np
from PIL import Image
import io
import os
import psycopg2
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app, origins=["*"], supports_credentials=True)

# Configuration from .env
# Ganti dengan nilai database Anda yang sebenarnya
DATABASE_URL = "postgresql://bingouser:bingoAPI123@103.250.10.132:5432/bingo"
# Ganti dengan folder upload yang Anda inginkan
UPLOAD_FOLDER = "./uploads"
# Batas upload untuk pengguna anonim
MAX_ANONYMOUS_UPLOADS = 3

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, User-Uid" # Added User-Uid
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response

# Load model (make sure 'best_model.h5' is in the same directory or provide a full path)
try:
    model = load_model('best_model.h5')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    # Exit or handle the error appropriately if the model is crucial

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


kategori_info = {
    'Organik': {
        'description': 'Sampah organik berasal dari bahan-bahan alami yang dapat terurai secara biologis seperti sisa makanan, daun, dan ranting.',
        'disposalSteps': [
            'Pisahkan dari sampah anorganik',
            'Buang di tempat sampah organik',
            'Bisa digunakan untuk kompos jika memungkinkan'
        ]
    },
    'Anorganik': {
        'description': 'Sampah anorganik adalah sampah yang tidak dapat terurai secara alami seperti plastik, kaca, dan logam.',
        'disposalSteps': [
            'Pisahkan berdasarkan jenis material (plastik, kaca, logam)',
            'Cuci bersih jika terkontaminasi makanan',
            'Buang di tempat sampah daur ulang atau tempat sampah anorganik'
        ]
    },
    'B3': {
        'description': 'Sampah B3 mengandung bahan berbahaya seperti baterai, elektronik, dan bahan kimia yang memerlukan penanganan khusus.',
        'disposalSteps': [
            'Jangan dibuang bersama sampah biasa',
            'Bawa ke tempat pengumpulan sampah B3',
            'Hubungi layanan pengelolaan limbah berbahaya di daerah Anda'
        ]
    }
}


def prepare_image_from_bytes(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def get_db_connection():
    try:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

def get_ip_address():
    # Attempt to get the real IP address, falling back to request.remote_addr
    if request.headers.getlist("X-Forwarded-For"):
        return request.headers.getlist("X-Forwarded-For")[0].split(',')[0].strip()
    return request.remote_addr

@app.route('/')
def index():
    return 'API is running!'

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    user_uid = request.headers.get('User-Uid') # Get user_uid from header
    ip_user = get_ip_address()

    conn = get_db_connection()
    if not conn:
        return jsonify({'error': 'Failed to connect to database'}), 500

    cursor = conn.cursor()

    try:
        # Check upload limits for anonymous users
        if not user_uid:
            cursor.execute(
                "SELECT COUNT(*) FROM bingo_analyze WHERE ip_user = %s AND created_at >= NOW() - INTERVAL '24 hours'",
                (ip_user,)
            )
            upload_count = cursor.fetchone()[0]

            if upload_count >= MAX_ANONYMOUS_UPLOADS:
                return jsonify({
                    'error': f'Anonymous uploads limited to {MAX_ANONYMOUS_UPLOADS} per 24 hours.',
                    'current_uploads': upload_count
                }), 429 # Too Many Requests

        # --- Image processing and prediction ---
        img_bytes = file.read()
        img_ready = prepare_image_from_bytes(img_bytes)

        preds = model.predict(img_ready)
        pred_idx = np.argmax(preds, axis=1)[0]
        pred_class = class_names[pred_idx]
        pred_group = group_map.get(pred_class, 'Unknown')
        info = kategori_info.get(pred_group, {
            'description': 'Kategori tidak diketahui.',
            'disposalSteps': ['Silakan periksa ulang jenis sampah ini.']
        })
        # --- Save image to disk and get path ---
        # Generate a unique filename
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        original_filename, file_extension = os.path.splitext(file.filename)
        # Sanitize filename for security and path validity
        safe_filename = f"image_{timestamp}_{original_filename[:50].replace(' ', '_').replace('/', '_')}{file_extension}"
        image_path_on_disk = os.path.join(UPLOAD_FOLDER, safe_filename)

        # Save the image bytes to the determined path
        with open(image_path_on_disk, 'wb') as f:
            f.write(img_bytes)

        # Store a relative path or a URL (if served from a CDN/static server)
        # For simplicity, we'll store the relative path within the UPLOAD_FOLDER
        image_db_path = os.path.join(os.path.basename(UPLOAD_FOLDER), safe_filename)

        # --- Save prediction result to database ---
        description =info['description']
        kategori = pred_group
        disposalSteps = info['disposalSteps']
        created_at = datetime.now()
        update_at = datetime.now() # For initial creation, update_at is same as created_at

        # If user_uid is None, psycopg2 will insert NULL
        cursor.execute(
            """
            INSERT INTO bingo_analyze (analyze_uid, user_uid, ip_user, description, image, created_at, update_at, kategori, disposalSteps)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                f"ANALYZE-{timestamp}-{ip_user.replace('.', '-')}", # Example analyze_uid, adjust as needed
                user_uid,
                ip_user,
                description,
                image_db_path,
                created_at,
                update_at,
                kategori,
                disposalSteps
            )
        )
        conn.commit()

        return jsonify({
            'Sampah': pred_class,
            'Kategori': pred_group,
            'Deskripsi': info['description'],
            'LangkahPembuangan': info['disposalSteps'],
            'message': 'Prediction saved successfully!'
        })

    except psycopg2.Error as db_err:
        conn.rollback() # Rollback in case of error
        print(f"Database error: {db_err}")
        return jsonify({'error': 'Database operation failed', 'details': str(db_err)}), 500
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return jsonify({'error': 'An internal server error occurred', 'details': str(e)}), 500
    finally:
        if conn:
            cursor.close()
            conn.close()


@app.route('/history', methods=['GET'])
def get_history():
    user_uid = request.headers.get('User-Uid')
    if not user_uid:
        return jsonify({'error': 'User-Uid header is required to view history'}), 401

    conn = get_db_connection()
    if not conn:
        return jsonify({'error': 'Failed to connect to database'}), 500

    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT analyze_id, analyze_uid, user_uid, ip_user, description, image, created_at, update_at, kategori, disposalSteps"
            "FROM bingo_analyze WHERE user_uid = %s ORDER BY created_at DESC",
            (user_uid,)
        )
        history_records = cursor.fetchall()

        # Format the results
        columns = [desc[0] for desc in cursor.description]
        formatted_history = []
        for record in history_records:
            record_dict = dict(zip(columns, record))
            # Convert datetime objects to string for JSON serialization
            if 'created_at' in record_dict and record_dict['created_at']:
                record_dict['created_at'] = record_dict['created_at'].isoformat()
            if 'update_at' in record_dict and record_dict['update_at']:
                record_dict['update_at'] = record_dict['update_at'].isoformat()
            formatted_history.append(record_dict)

        return jsonify({'history': formatted_history})

    except psycopg2.Error as db_err:
        print(f"Database error fetching history: {db_err}")
        return jsonify({'error': 'Database operation failed', 'details': str(db_err)}), 500
    except Exception as e:
        print(f"An unexpected error occurred fetching history: {e}")
        return jsonify({'error': 'An internal server error occurred', 'details': str(e)}), 500
    finally:
        if conn:
            cursor.close()
            conn.close()

if __name__ == '__main__':
    # It's recommended to run Flask in production with a production-ready WSGI server
    # like Gunicorn, not directly with app.run().
    # For development:
    app.run(host='0.0.0.0', port=5000, debug=True)