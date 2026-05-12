from flask import Flask, request, jsonify
import joblib
import os
import numpy as np
import pandas as pd

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# الترتيب الصحيح للأعمدة
FEATURES_ORDER = [
    'age', 'gender', 'diabetes', 'hypertension', 'heart_disease',
    'glucose_mg_dl', 'systolic_bp', 'diastolic_bp', 'heart_rate',
    'temperature_c', 'spo2'
]

def load_resources():
    try:
        model_path  = os.path.join(BASE_DIR, 'pulsekey_model.pkl')
        scaler_path = os.path.join(BASE_DIR, 'pulsekey_scaler.pkl')

        if not os.path.exists(model_path):
            model_path  = 'api/pulsekey_model.pkl'
            scaler_path = 'api/pulsekey_scaler.pkl'

        if os.path.exists(model_path) and os.path.exists(scaler_path):
            m = joblib.load(model_path)
            s = joblib.load(scaler_path)
            print("Successfully loaded model and scaler.")
            return m, s
        else:
            print(f"Files not found at: {model_path} or {scaler_path}")
            return None, None
    except Exception as e:
        print(f"Error loading resources: {str(e)}")
        return None, None

model, scaler = load_resources()

@app.route('/')
def home():
    return "PulseKey API is Online and Ready to Predict"

@app.route('/predict', methods=['POST'])
def predict():
    global model, scaler

    if model is None or scaler is None:
        model, scaler = load_resources()
        if model is None:
            return jsonify({
                "status": "error",
                "message": "Model files missing on server. Check api folder."
            }), 500

    try:
        json_data = request.get_json()

        if not json_data:
            return jsonify({"status": "error", "message": "JSON body is empty"}), 400

        if 'data' not in json_data:
            return jsonify({"status": "error", "message": "No 'data' key provided"}), 400

        raw = json_data['data']

        # ✅ بيقبل الاتنين: array أو dict
        if isinstance(raw, list):
            # ["data": [45, 1, 0, 1, 0, 150.0, 135.0, 85.0, 90.0, 37.0, 97.0]]
            features = np.array(raw, dtype=float).reshape(1, -1)

        elif isinstance(raw, dict):
            # {"data": {"age": 45, "gender": 1, ...}}
            df = pd.DataFrame([raw])
            df = df[FEATURES_ORDER]
            df = df.apply(pd.to_numeric, errors='coerce')
            features = df.values

        else:
            return jsonify({"status": "error", "message": "'data' must be a list or dict"}), 400

        scaled_features = scaler.transform(features)
        prediction      = model.predict(scaled_features)

        mapping = {0: "High Risk", 1: "Low Risk", 2: "Medium Risk"}
        res     = int(prediction[0])

        return jsonify({
            "status":     "success",
            "risk_level": res,
            "label":      mapping.get(res, "Unknown")
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

handler = app
