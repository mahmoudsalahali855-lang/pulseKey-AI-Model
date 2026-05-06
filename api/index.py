from flask import Flask, request, jsonify
import joblib
import os
import numpy as np

app = Flask(__name__)

# تحديد المسار الحالي للفولدر (api) لضمان الوصول للملفات
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# تحميل الموديل والسكيلر بأساميهم اللي في الصورة
try:
    model_path = os.path.join(BASE_DIR, 'pulsekey_model.pkl')
    scaler_path = os.path.join(BASE_DIR, 'pulsekey_scaler.pkl')
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except Exception as e:
    print(f"Error loading files: {str(e)}")
    model = None
    scaler = None

@app.route('/')
def home():
    return "PulseKey AI Model is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({
            "status": "error",
            "message": "Model files not found on server. Check paths and filenames."
        }), 500

    try:
        # استقبال البيانات من Postman
        data = request.get_json()
        features = np.array(data['data']).reshape(1, -1)
        
        # عمل Scaling للبيانات
        scaled_features = scaler.transform(features)
        
        # التوقع
        prediction = model.predict(scaled_features)
        
        return jsonify({
            "status": "success",
            "risk_level": int(prediction[0])
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

# مهم لـ Vercel
def handler(environ, start_response):
    return app(environ, start_response)
