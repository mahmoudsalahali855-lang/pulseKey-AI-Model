import os
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# تحديد مكان الملفات بدقة
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# بنطلع خطوة لبره فولدر api عشان نلاقي الملفات في الـ Root
MODEL_PATH = os.path.join(BASE_DIR, "..", "pulsekey_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "..", "pulsekey_scaler.pkl")

model = None
scaler = None

# محاولة تحميل الملفات عند التشغيل
try:
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    else:
        print(f"Files not found at: {MODEL_PATH} or {SCALER_PATH}")
except Exception as e:
    print(f"Loading Error: {str(e)}")

@app.route('/')
def home():
    return "PulseKey AI API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({"status": "error", "message": "Model files not loaded on server"})
    
    try:
        json_data = request.get_json()
        input_list = json_data.get('data')
        
        if not input_list:
            return jsonify({"status": "error", "message": "No data provided"})

        # المعالجة والتنبؤ
        scaled_data = scaler.transform([input_list])
        prediction = model.predict(scaled_data)
        
        return jsonify({
            "status": "success",
            "risk_level": int(prediction[0])
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
