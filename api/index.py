import os
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app) # عشان يسمح للفرونت إند يكلم الباك إند بدون مشاكل (CORS)

# 1. تحديد المسار الصحيح للمجلد الرئيسي
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. تحميل الموديل والسكيلر (تأكد أن الأسماء تطابق الملفات في GitHub)
try:
    model_path = os.path.join(BASE_DIR, '..', 'pulsekey_model.pkl')
    scaler_path = os.path.join(BASE_DIR, '..', 'pulsekey_scaler.pkl')
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except Exception as e:
    print(f"Error loading models: {e}")

@app.route('/')
def home():
    return "PulseKey AI API is Running! Ready for predictions."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # استلام البيانات من الـ Request
        json_data = request.json
        if 'data' not in json_data:
            return jsonify({'status': 'error', 'error': 'Missing "data" key in request'})
        
        input_data = json_data['data']
        
        # تحويل البيانات باستخدام الـ Scaler ثم التنبؤ بالموديل
        scaled_data = scaler.transform([input_data])
        prediction = model.predict(scaled_data)
        
        # إرجاع النتيجة
        return jsonify({
            'status': 'success',
            'risk_level': int(prediction[0])
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error', 
            'error': str(e)
        })

# ده ضروري عشان Vercel يتعامل مع Flask كـ Serverless Function
if __name__ == '__main__':
    app.run(debug=True)
