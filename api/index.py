from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# تحميل الموديل والسكيلر
model = joblib.load('pulsekey_model.pkl')
scaler = joblib.load('pulsekey_scaler.pkl')

@app.route('/')
def home():
    return "PulseKey AI API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['data']
        # تحويل البيانات لمصفوفة
        input_data = np.array(data).reshape(1, -1)
        
        # معالجة البيانات بالسكيلر
        scaled_data = scaler.transform(input_data)
        
        # التنبؤ
        prediction = model.predict(scaled_data)
        
        return jsonify({
            'status': 'success',
            'risk_level': int(prediction[0])
        })
    except Exception as e:
        return jsonify({'error': str(e)})
