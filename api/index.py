from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# --- الجزء الخاص بتحميل الموديل والسكيلر بمسارات ديناميكية ---
try:
    # تحديد مسار الفولدر الحالي اللي فيه الكود
    base_path = os.path.dirname(__file__)
    
    # تحميل الموديل والسكيلر (تأكد أن الملفات بنفس هذه الأسماء في الفولدر)
    model_path = os.path.join(base_path, 'model.pkl')
    scaler_path = os.path.join(base_path, 'scaler.pkl')
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    model_loaded = True
except Exception as e:
    print(f"Error loading model: {e}")
    model_loaded = False

@app.route('/')
def home():
    return "PulseKey AI API is Running! Send POST request to /predict"

@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded:
        return jsonify({
            "status": "error",
            "message": "Model files not found on server. Check file paths."
        }), 500

    try:
        # استلام البيانات من Postman
        input_data = request.get_json()
        
        if not input_data or 'data' not in input_data:
            return jsonify({"status": "error", "message": "No data provided"}), 400
        
        # تحويل البيانات لـ Numpy Array
        features = np.array(input_data['data']).reshape(1, -1)
        
        # عمل Scaling للبيانات
        scaled_features = scaler.transform(features)
        
        # التوقع باستخدام الموديل
        prediction = model.predict(scaled_features)
        
        # إرسال النتيجة لـ Postman
        return jsonify({
            "status": "success",
            "risk_level": int(prediction[0])
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# مهم جداً لـ Vercel
if __name__ == '__main__':
    app.run(debug=True)
