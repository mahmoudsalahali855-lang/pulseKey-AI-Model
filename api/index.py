import os
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# بما إن الملفات والموديل كلهم جوه فولدر api
# الكود ده هيخليه يقرأ من نفس الفولدر اللي فيه الملف ده
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "pulsekey_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "pulsekey_scaler.pkl")

# تحميل الموديل والـ scaler عند بدء التشغيل
try:
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("Files Loaded Successfully from api folder")
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
    # التأكد من تحميل الملفات قبل البدء
    if model is None or scaler is None:
        return jsonify({
            "status": "error", 
            "message": "Model files not found. Check if .pkl files are inside 'api' folder."
        })
    
    try:
        json_data = request.get_json()
        if not json_data or 'data' not in json_data:
            return jsonify({"status": "error", "message": "No data provided in 'data' field"})

        input_list = json_data.get('data')
        
        # المعالجة والتنبؤ
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
