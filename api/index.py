import os
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# تحديد المسار الفعلي للفولدر اللي فيه الكود
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "pulsekey_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "pulsekey_scaler.pkl")

# تحميل الموديل والـ scaler عند بدء التشغيل
try:
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("Files Loaded Successfully")
    else:
        model = None
        scaler = None
        print("Error: Model or Scaler files not found in root!")
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
            "message": "Model files not found on server. Check file names and locations."
        })
    
    try:
        json_data = request.get_json()
        if not json_data or 'data' not in json_data:
            return jsonify({"status": "error", "message": "No data provided in 'data' field"})

        input_list = json_data.get('data')
        
        # التأكد أن البيانات مبعوتة كـ List
        if not isinstance(input_list, list):
            return jsonify({"status": "error", "message": "Data must be a list of features"})

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
if __name__ == "__main__":
    app.run(debug=True)
