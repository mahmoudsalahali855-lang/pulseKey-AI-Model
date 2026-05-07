from flask import Flask, request, jsonify
import joblib
import os
import numpy as np

app = Flask(__name__)

# المسار المطلق لضمان الوصول للملفات pkl
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_resources():
    try:
        model_path = os.path.join(BASE_DIR, 'pulsekey_model.pkl')
        scaler_path = os.path.join(BASE_DIR, 'pulsekey_scaler.pkl')
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            m = joblib.load(model_path)
            s = joblib.load(scaler_path)
            return m, s
        return None, None
    except Exception as e:
        print(f"Loading error: {e}")
        return None, None

model, scaler = load_resources()

@app.route('/')
def home():
    return "PulseKey API is Online and Ready"

@app.route('/predict', methods=['POST'])
def predict():
    global model, scaler
    if model is None or scaler is None:
        model, scaler = load_resources()
        if model is None:
            return jsonify({
                "status": "error", 
                "message": "Model files missing. Ensure pulsekey_model.pkl and pulsekey_scaler.pkl are in the api folder."
            }), 500

    try:
        data = request.get_json()
        if not data or 'data' not in data:
            return jsonify({"status": "error", "message": "Invalid input. Expected 'data' key."}), 400
            
        features = np.array(data['data']).reshape(1, -1)
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)
        
        return jsonify({
            "status": "success",
            "risk_level": int(prediction[0])
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

# ده السطر اللي Vercel بيحبه عشان يشغل الـ Function
handler = app
