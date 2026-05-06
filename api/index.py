import os
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Vercel بيقرأ الملفات من نفس الفولدر اللي فيه index.py مباشرة
# قراءة الموديل والـ scaler
try:
    model = joblib.load('pulsekey_model.pkl')
    scaler = joblib.load('pulsekey_scaler.pkl')
    print("Files Loaded Successfully")
except Exception as e:
    # لو فشل التحميل، هنحاول نقرأهم بمسار كامل للتأكيد
    try:
        base_path = os.path.dirname(__file__)
        model = joblib.load(os.path.join(base_path, 'pulsekey_model.pkl'))
        scaler = joblib.load(os.path.join(base_path, 'pulsekey_scaler.pkl'))
    except:
        model = None
        scaler = None

@app.route('/')
def home():
    return "PulseKey AI API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    # التأكد من أن المتغيرات موجودة وليست None
    if model is None or scaler is None:
        return jsonify({
            "status": "error", 
            "message": "Files not found. Ensure pulsekey_model.pkl and pulsekey_scaler.pkl are in the root directory."
        })
    
    try:
        json_data = request.get_json()
        input_list = json_data.get('data')
        
        if not input_list:
            return jsonify({"status": "error", "message": "No data provided"})

        # تنفيذ عملية الـ Scaling والتنبؤ
        scaled_data = scaler.transform([input_list])
        prediction = model.predict(scaled_data)
        
        return jsonify({
            "status": "success",
            "risk_level": int(prediction[0])
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# مهم جداً لـ Vercel
if __name__ == "__main__":
    app.run(debug=True)
