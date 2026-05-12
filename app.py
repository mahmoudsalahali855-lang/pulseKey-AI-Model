from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)

# تحميل الموديل والسكيلر
try:
    model = joblib.load('pulsekey_model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")

# الترتيب الصحيح لـ 11 متغير
features_order = [
    'age', 'gender', 'diabetes', 'hypertension', 'heart_disease', 
    'glucose_mg_dl', 'systolic_bp', 'diastolic_bp', 'heart_rate', 
    'temperature_c', 'spo2'
]

def get_medical_advice(vitals, risk_level):
    advices = []
    if vitals.get('glucose_mg_dl', 0) > 140:
        advices.append("🟡 السكر مرتفع: قلل السكريات.")
    if vitals.get('systolic_bp', 0) > 130:
        advices.append("🟡 الضغط مرتفع: قلل الأملاح.")
    
    # 0=High, 1=Low, 2=Medium
    if risk_level == 0:
        advices.append("🚨 حالة حرجة: استشر طبيبك فوراً.")
    elif risk_level == 2:
        advices.append("⚠️ حالة متوسطة: ارتاح وأعد القياس.")
    
    return advices if advices else ["✅ حالتك مستقرة حالياً."]

@app.route('/')
def home():
    return "PulseKey API is Online and Ready!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # استلام البيانات الخام
        json_data = request.get_json()
        
        # أهم سطر: لو الداتا فاضية خالص ابعت إيرور واضح
        if not json_data:
            return jsonify({'status': 'error', 'message': 'JSON body is empty'}), 400

        # مرونة في القراءة: بيدور على مفتاح 'data' ولو مش موجود بياخد الـ JSON كله
        if isinstance(json_data, dict) and 'data' in json_data:
            input_data = json_data['data']
        else:
            input_data = json_data

        # تحويل لـ DataFrame
        df = pd.DataFrame([input_data])
        
        # التأكد من وجود كل الأعمدة
        df = df[features_order]
        
        # تحويل كل القيم لأرقام (عشان نتفادى أي إيرور في الـ float)
        df = df.apply(pd.to_numeric, errors='coerce')

        # تطبيق السكيلر والتوقع
        nums = ['glucose_mg_dl', 'systolic_bp', 'diastolic_bp', 'heart_rate', 'temperature_c', 'spo2']
        df[nums] = scaler.transform(df[nums])
        
        prediction = model.predict(df)
        res = int(prediction[0])
        
        mapping = {0: "High Risk", 1: "Low Risk", 2: "Medium Risk"}
        
        return jsonify({
            'status': 'success',
            'risk_level': res,
            'label': mapping.get(res),
            'advice': get_medical_advice(input_data, res)
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': f"Prediction Error: {str(e)}"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
