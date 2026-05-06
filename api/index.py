from flask import Flask, request, jsonify
import joblib
import os
import numpy as np

app = Flask(__name__)

# تحديد المسار بمنتهى الدقة
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_model():
    try:
        m_path = os.path.join(BASE_DIR, 'pulsekey_model.pkl')
        s_path = os.path.join(BASE_DIR, 'pulsekey_scaler.pkl')
        
        # التأكد من وجود الملفات قبل التحميل
        if not os.path.exists(m_path) or not os.path.exists(s_path):
            return None, None
            
        m = joblib.load(m_path)
        s = joblib.load(s_path)
        return m, s
    except:
        return None, None

model, scaler = get_model()

@app.route('/')
def home():
    return "PulseKey AI is Online"

@app.route('/predict', methods=['POST'])
def predict():
    global model, scaler
    if model is None or scaler is None:
        # محاولة تحميل ثانية في حالة الفشل الأول
        model, scaler = get_model()
        if model is None:
            return jsonify({"error": "Model files missing in api folder"}), 500

    try:
        json_data = request.get_json()
        if not json_data or 'data' not in json_data:
            return jsonify({"error": "No data provided"}), 400
            
        input_data = np.array(json_data['data']).reshape(1, -1)
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)
        
        return jsonify({
            "status": "success",
            "risk_level": int(prediction[0])
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# السطر ده هو اللي Vercel بيستخدمه فعلياً
app_handler = app
