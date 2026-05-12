from flask import Flask, request, jsonify
import joblib
import os
import numpy as np
import pandas as pd

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# الترتيب الكامل للأعمدة
FEATURES_ORDER = [
    'age', 'gender', 'diabetes', 'hypertension', 'heart_disease',
    'glucose_mg_dl', 'systolic_bp', 'diastolic_bp', 'heart_rate',
    'temperature_c', 'spo2'
]

# الأعمدة الرقمية فقط اللي اتعمل عليها الـ Scaler
NUMERIC_COLS = ['glucose_mg_dl', 'systolic_bp', 'diastolic_bp', 'heart_rate', 'temperature_c', 'spo2']

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

        # تحويل الداتا لـ DataFrame
        if isinstance(raw, list):
            df = pd.DataFrame([raw], columns=FEATURES_ORDER)
        elif isinstance(raw, dict):
            df = pd.DataFrame([raw])
            df = df[FEATURES_ORDER]
        else:
            return jsonify({"status": "error", "message": "'data' must be a list or dict"}), 400

        df = df.apply(pd.to_numeric, errors='coerce')

        # ✅ Scale الـ 6 أعمدة الرقمية بس
        df[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])

        prediction = model.predict(df)

        mapping = {0: "High Risk", 1: "Low Risk", 2: "Medium Risk"}
        res = int(prediction[0])

        return jsonify({
            "status":     "success",
            "risk_level": res,
            "label":      mapping.get(res, "Unknown")
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

# ✅ مهم لـ Railway
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
