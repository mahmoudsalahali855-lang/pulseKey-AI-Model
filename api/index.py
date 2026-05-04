from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# تحديد المسار الصحيح للمجلد اللي فيه الكود والموديل
base_path = os.path.dirname(__file__)

# تحميل الموديل والسكيلر باستخدام المسارات المطلقة
model_path = os.path.join(base_path, 'pulsekey_model.pkl')
scaler_path = os.path.join(base_path, 'pulsekey_scaler.pkl')

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except Exception as e:
    print(f"Error loading models: {e}")

@app.route('/')
def home():
    return "PulseKey AI API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # استقبال البيانات من الطلب
        data = request.json['data']
        
        # تحويل البيانات لمصفوفة NumPy وعمل Reshape
        input_data = np.array(data).reshape(1, -1)
        
        # معالجة البيانات بالسكيلر
        scaled_data = scaler.transform(input_data)
        
        # التنبؤ باستخدام الموديل
        prediction = model.predict(scaled_data)
        
        return jsonify({
            'status': 'success',
            'risk_level': int(prediction[0])
        })
    except Exception as e:
        return jsonify({'error': str(e)})

# مهم جداً لـ Vercel
if __name__ == '__main__':
    app.run(debug=True)
