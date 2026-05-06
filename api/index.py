import os
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# التعديل هنا: يخرج من فولدر api ويروح للـ Root عشان يلاقي الملفات
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "pulsekey_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "pulsekey_scaler.pkl")

# تحميل الموديل والـ scaler
try:
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("Files Loaded Successfully from Root")
    else:
        model = None
        scaler = None
        print(f"Error: Files not found at {MODEL_PATH}")
except Exception as e:
    model = None
    scaler = None
    print(f"Loading Error: {str(e)}")

@app.route('/')
def home():
    return "PulseKey AI API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({
            "status": "error", 
            "message": f"Files not found. System looked at: {MODEL_PATH}"
        })
    
    try:
        json_data = request.get_json()
        input_list = json_data.get('data')
        
        if not input_list:
            return jsonify({"status": "error", "message": "No data provided"})

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
