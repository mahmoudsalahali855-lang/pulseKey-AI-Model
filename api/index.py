from flask import Flask, request, jsonify
import joblib
import os
import numpy as np
import pandas as pd

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FEATURES_ORDER = [
    'age', 'gender', 'diabetes', 'hypertension', 'heart_disease',
    'glucose_mg_dl', 'systolic_bp', 'diastolic_bp', 'heart_rate',
    'temperature_c', 'spo2'
]

NUMERIC_COLS = ['glucose_mg_dl', 'systolic_bp', 'diastolic_bp', 'heart_rate', 'temperature_c', 'spo2']

def load_resources():
    try:
        model_path  = os.path.join(BASE_DIR, 'pulsekey_model.pkl')
        scaler_path = os.path.join(BASE_DIR, 'pulsekey_scaler.pkl')

        if not os.path.exists(model_path):
            model_path  = 'api/pulsekey_model.pkl'
            scaler_path = 'api/pulsekey_scaler.pkl'

        if os.path.exists(model_path) and os.path.exists(scaler_path):
            m = joblib.load(model_path)
            s = joblib.load(scaler_path)
            print("Successfully loaded model and scaler.")
            return m, s
        else:
            print(f"Files not found at: {model_path} or {scaler_path}")
            return None, None
    except Exception as e:
        print(f"Error loading resources: {str(e)}")
        return None, None

model, scaler = load_resources()

# ─────────────────────────────────────────────
# دالة الشات
# ─────────────────────────────────────────────
def pulsekey_chatbot_reply(user_query, report_data):
    query = user_query.lower().strip()

    if not report_data or 'risk_level' not in report_data:
        return "محتاج أشوف علاماتك الحيوية الأول عشان أقدر أساعدك.\nI need your vitals first to help you."

    risk    = report_data.get('risk_level', 1)
    vitals  = report_data.get('vitals', {})
    advices = report_data.get('advice_list', [])

    # الحالة الصحية
    if any(w in query for w in ["status", "report", "حالة", "تقرير", "عامل ايه"]):
        status_map = {
            0: "Critical Risk 🚨 (خطر عالي)",
            1: "Stable ✅ (مستقر)",
            2: "Needs Monitoring 🟡 (خطر متوسط)"
        }
        return f"حالتك الصحية حالياً: {status_map.get(risk, 'غير معروف')}. اتبع التعليمات في تقريرك."

    # النصائح
    if any(w in query for w in ["advice", "do", "نصيحة", "اعمل ايه", "مساعدة"]):
        formatted = "\n".join([f"- {a}" for a in advices])
        return f"بناءً على بياناتك، أنصحك بالآتي:\n{formatted}"

    # السكر
    if any(w in query for w in ["sugar", "glucose", "سكر"]):
        return f"آخر قراءة للسكر هي: {vitals.get('glucose_mg_dl', 'N/A')} mg/dL."

    # الضغط
    if any(w in query for w in ["pressure", "bp", "ضغط"]):
        return f"ضغط الدم: {vitals.get('systolic_bp', 'N/A')}/{vitals.get('diastolic_bp', 'N/A')} mmHg."

    # النبض
    if any(w in query for w in ["heart rate", "pulse", "نبض", "قلب"]):
        return f"معدل ضربات القلب: {vitals.get('heart_rate', 'N/A')} bpm."

    # الحرارة
    if any(w in query for w in ["temp", "temperature", "حرارة"]):
        return f"درجة الحرارة: {vitals.get('temperature_c', 'N/A')} °C."

    # الأكسجين
    if any(w in query for w in ["spo2", "oxygen", "أكسجين", "اكسجين"]):
        return f"نسبة الأكسجين: {vitals.get('spo2', 'N/A')}%."

    # شرح السكري
    if any(w in query for w in ["ما هو السكر", "diabetes", "سكري"]):
        return "مرض السكري هو ارتفاع نسبة السكر في الدم نتيجة خلل في هرمون الإنسولين، ونحن هنا نساعدك في مراقبته."

    # شرح الضغط
    if any(w in query for w in ["ما هو الضغط", "blood pressure", "ضغط الدم"]):
        return "ضغط الدم هو قوة دفع الدم للشرايين؛ الارتفاع المستمر قد يرهق القلب، لذا نراقبه بدقة."

    # رد افتراضي
    return "أنا مساعد PulseKey. يمكنك سؤالي عن: حالتك الصحية، نصائح طبية، أو قراءات السكر والضغط والنبض."


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route('/')
def home():
    return "PulseKey API is Online and Ready to Predict"


@app.route('/predict', methods=['POST'])
def predict():
    global model, scaler

    if model is None or scaler is None:
        model, scaler = load_resources()
        if model is None:
            return jsonify({
                "status": "error",
                "message": "Model files missing on server. Check api folder."
            }), 500

    try:
        json_data = request.get_json()

        if not json_data:
            return jsonify({"status": "error", "message": "JSON body is empty"}), 400

        if 'data' not in json_data:
            return jsonify({"status": "error", "message": "No 'data' key provided"}), 400

        raw = json_data['data']

        if isinstance(raw, list):
            df = pd.DataFrame([raw], columns=FEATURES_ORDER)
        elif isinstance(raw, dict):
            df = pd.DataFrame([raw])
            df = df[FEATURES_ORDER]
        else:
            return jsonify({"status": "error", "message": "'data' must be a list or dict"}), 400

        df = df.apply(pd.to_numeric, errors='coerce')
        df[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])

        prediction = model.predict(df)
        res = int(prediction[0])

        mapping = {0: "High Risk", 1: "Low Risk", 2: "Medium Risk"}

        return jsonify({
            "status":     "success",
            "risk_level": res,
            "label":      mapping.get(res, "Unknown")
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route('/chat', methods=['POST'])
def chat():
    try:
        json_data = request.get_json()

        if not json_data:
            return jsonify({"status": "error", "message": "JSON body is empty"}), 400

        user_query  = json_data.get('message', '').strip()
        report_data = json_data.get('report', {})

        if not user_query:
            return jsonify({"status": "error", "message": "No 'message' key provided"}), 400

        reply = pulsekey_chatbot_reply(user_query, report_data)

        return jsonify({
            "status": "success",
            "reply":  reply
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


# ✅ مهم لـ Railway
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
