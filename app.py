from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)

# 1. تحميل الموديلات (تأكد من الأسماء في الفولدر عندك)
try:
    model = joblib.load('pulsekey_model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("AI Assets Loaded!")
except Exception as e:
    print(f"Error: {e}")

# ترتيب الـ 11 متغير
features_order = [
    'age', 'gender', 'diabetes', 'hypertension', 'heart_disease', 
    'glucose_mg_dl', 'systolic_bp', 'diastolic_bp', 'heart_rate', 
    'temperature_c', 'spo2'
]

# ----------------- دالة النصائح الطبية (منطق المشروع) -----------------
def get_medical_advice(vitals, risk_level):
    advices = []
    # السكر
    if vitals.get('glucose_mg_dl', 0) > 140:
        advices.append("🟡 السكر مرتفع: قلل النشويات.")
    elif 0 < vitals.get('glucose_mg_dl', 0) < 70:
        advices.append("🔵 السكر منخفض: تناول مصدر سكر سريع.")
    
    # الضغط
    if vitals.get('systolic_bp', 0) > 130:
        advices.append("🟡 الضغط الانقباضي عالي: قلل الملح.")
    
    # بناءً على الـ Risk Level (التعديل الجديد: 0=High)
    if risk_level == 0:
        advices.append("🚨 حالة حرجة: تواصل مع طبيبك فوراً.")
    elif risk_level == 2:
        advices.append("⚠️ حالة متوسطة: يرجى الراحة وإعادة القياس.")
    
    return advices if advices else ["✅ حالتك مستقرة، استمر على نمطك الصحي."]

# ----------------- الخدمة الأولى: التحاليل (Predict) -----------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])[features_order]
        
        # Scaling للـ 6 الرقميين
        nums = ['glucose_mg_dl', 'systolic_bp', 'diastolic_bp', 'heart_rate', 'temperature_c', 'spo2']
        df[nums] = scaler.transform(df[nums])
        
        # التوقع
        prediction = model.predict(df)
        res = int(prediction[0])
        
        # المابينج الجديد
        mapping = {0: "High Risk", 1: "Low Risk", 2: "Medium Risk"}
        
        return jsonify({
            'status': 'success',
            'risk_level': res,
            'label': mapping.get(res),
            'advice': get_medical_advice(data, res)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ----------------- الخدمة الثانية: الشات بوت (Chat) -----------------
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    msg = data.get('message', '').lower()
    
    # ردود سريعة
    if any(word in msg for word in ["سلام", "أهلا", "hi"]):
        reply = "أهلاً بك في PulseKey، كيف يمكنني مساعدتك؟"
    elif "حالة" in msg or "status" in msg:
        reply = "برجاء إتمام الفحص في صفحة التحاليل أولاً لأتمكن من تشخيص حالتك."
    else:
        reply = "أنا مساعد PulseKey الصحي، يمكنك سؤالي عن السكر أو الضغط أو حالتك الصحية."
        
    return jsonify({'reply': reply})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)