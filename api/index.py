import os
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# تحديد مكان الملفات في الـ Root بتاع المشروع مباشرة
# Vercel بيحط الملفات اللي في الـ root في مكان يقدر الكود يشوفه بسهولة
MODEL_PATH = os.path.join(os.getcwd(), "pulsekey_model.pkl")
SCALER_PATH = os.path.join(os.getcwd(), "pulsekey_scaler.pkl")

# تحميل الموديل والـ scaler مباشرة
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Files Loaded Successfully")
except Exception as e:
    model = None
    scaler = None
    print(f"Loading Error: {str(e)}")

@app.route('/')
def home():
    return "PulseKey AI API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    # التأكد من تحميل الملفات قبل البدء
    if model is None or scaler is None:
        return jsonify({
            "status": "error", 
            "message": "Model files not loaded. Check if files exist in root directory."
        })
    
    try:
        json_data = request.get_json()
        input_list = json_data.get('data')
        
        if not input_list:
            return jsonify({"status": "error", "message": "No data provided"})

        # المعالجة والتنبؤ
        # استخدمنا scaler اللي تم تحميله فوق
        scaled_data = scaler.transform([input_list])
        prediction = model.predict(scaled_data)
        
        return jsonify({
            "status": "success",
            "risk_level": int(prediction[0])
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# مهم لـ Vercel
app.debug = True
