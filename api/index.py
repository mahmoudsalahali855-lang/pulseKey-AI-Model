from flask import Flask, request, jsonify
import joblib
import os
import numpy as np

app = Flask(__name__)

# تحديد المسار المطلق للفولدر الحالي (api)
# ده بيخلي Vercel يشوف الملفات اللي جنبه بالظبط
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# تحميل الموديل والسكيلر بأساميهم اللي موجودة في المستودع
try:
    model_path = os.path.join(BASE_DIR, 'pulsekey_model.pkl')
    scaler_path = os.path.join(BASE_DIR, 'pulsekey_scaler.pkl')
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("Models loaded successfully!")
except Exception as e:
    # لو حصل مشكلة في التحميل هتظهر في الـ Logs بتاعة Vercel
    print(f"Error loading files: {str(e)}")
    model = None
    scaler = None

@app.route('/')
def home():
    return "PulseKey AI Model is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    # التأكد من أن الموديلات تم تحميلها قبل التوقع
    if model is None or scaler is None:
        return jsonify({
            "status": "error",
            "message": "Model files not found on server. Please check paths and filenames in 'api' folder."
        }), 500

    try:
        # استقبال البيانات من Postman (يجب أن تكون قائمة أرقام)
        data = request.get_json()
        
        if not data or 'data' not in data:
            return jsonify({"status": "error", "message": "No data provided"}), 400
            
        features = np.array(data['data']).reshape(1, -1)
        
        # 1. عمل Scaling للبيانات بنفس السكيلر بتاع التدريب
        scaled_features = scaler.transform(features)
        
        # 2. التوقع باستخدام الموديل
        prediction = model.predict(scaled_features)
        
        # إرجاع النتيجة
        return jsonify({
            "status": "success",
            "risk_level": int(prediction[0])
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Prediction error: {str(e)}"
        }), 400

# السطر ده مهم جداً عشان Vercel يعرف يشغل Flask
def handler(environ, start_response):
    return app(environ, start_response)
