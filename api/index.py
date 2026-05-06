from flask import Flask, request, jsonify
import joblib
import os
import numpy as np

app = Flask(__name__)

def load_file(file_name):
    # قائمة بالأماكن المحتملة للملف
    possible_paths = [
        os.path.join(os.path.dirname(__file__), file_name),           # جنبه في api/
        os.path.join(os.getcwd(), file_name),                        # في الفولدر الحالي
        os.path.join(os.getcwd(), 'api', file_name),                 # جوه api من الفولدر الرئيسي
        os.path.abspath(file_name)                                   # المسار المطلق
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found {file_name} at: {path}")
            return joblib.load(path)
    return None

# محاولة تحميل الملفات
model = load_file('pulsekey_model.pkl')
scaler = load_file('pulsekey_scaler.pkl')

@app.route('/')
def home():
    return "PulseKey AI Model is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({
            "status": "error", 
            "message": "Files not found. Make sure .pkl files are in the same folder as index.py"
        }), 500

    try:
        data = request.get_json()
        features = np.array(data['data']).reshape(1, -1)
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)
        
        return jsonify({
            "status": "success",
            "risk_level": int(prediction[0])
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

def handler(environ, start_response):
    return app(environ, start_response)
