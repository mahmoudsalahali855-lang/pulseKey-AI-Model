from flask import Flask, request, jsonify
import joblib
import os
import numpy as np

app = Flask(__name__)

# تحديد المسار المطلق للفولدر الحالي (api)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_resources():
    try:
        # المحاولة الأولى: المسار المطلق المباشر
        model_path = os.path.join(BASE_DIR, 'pulsekey_model.pkl')
        scaler_path = os.path.join(BASE_DIR, 'pulsekey_scaler.pkl')
        
        # المحاولة الثانية: لو السيرفر مش شايف المسار المطلق (احتياطي لـ Vercel)
        if not os.path.exists(model_path):
            model_path = 'api/pulsekey_model.pkl'
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

# تحميل الموديل والسكيلر عند تشغيل السيرفر
model, scaler = load_resources()

@app.route('/')
def home():
    return "PulseKey API is Online and Ready to Predict"

@app.route('/predict', methods=['POST'])
def predict():
    global model, scaler
    
    # التأكد إن الملفات متحملة، ولو مش موجودة يحاول يحملها تاني
    if model is None or scaler is None:
        model, scaler = load_resources()
        if model is None:
            return jsonify({
                "status": "error", 
                "message": "Model files missing on server. Check api folder."
            }), 500

    try:
        # استقبال البيانات من طلب البوست
        data = request.get_json()
        if not data or 'data' not in data:
            return jsonify({"status": "error", "message": "No 'data' key provided"}), 400
            
        # تحويل البيانات لـ numpy array وعمل الـ scaling
        features = np.array(data['data']).reshape(1, -1)
        scaled_features = scaler.transform(features)
        
        # التوقع (Prediction)
        prediction = model.predict(scaled_features)
        
        return jsonify({
            "status": "success",
            "risk_level": int(prediction[0])
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

# سطر مهم جداً لـ Vercel ليعرف الـ entry point
handler = app
