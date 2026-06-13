from flask import Flask, request, jsonify
import joblib
import os
import numpy as np
import pandas as pd
import random
import anthropic

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
# دالة الشات بـ Claude AI
# ─────────────────────────────────────────────
def pulsekey_chatbot_reply(user_query, report_data):

    # لو مفيش داتا
    if 'risk_level' not in report_data:
        return random.choice([
            "👋 أهلاً! محتاج تدخل علاماتك الحيوية الأول عشان أقدر أساعدك.\nHello! Please enter your vitals first so I can help you.",
            "😊 مرحباً! ابدأ بإدخال بياناتك الصحية وأنا هساعدك.\nWelcome! Start by entering your health data and I'll assist you.",
            "🩺 لو عايز تعرف حالتك، ادخل قراءاتك الحيوية الأول.\nTo check your health status, please enter your vitals first."
        ])

    risk    = report_data.get('risk_level', 1)
    vitals  = report_data.get('vitals', {})
    advices = report_data.get('advice_list', [])

    risk_map = {
        0: ("Critical Risk 🚨", "خطر عالي — لازم تتواصل مع الطبيب فوراً!"),
        1: ("Stable ✅",        "مستقر — استمر في نمط حياتك الصحي!"),
        2: ("Medium Risk 🟡",   "خطر متوسط — خد بالك من نفسك وراقب القراءات.")
    }
    status_en, status_ar = risk_map.get(risk, ("Unknown", "غير معروف"))

    advices_text = "\n".join([f"• {a}" for a in advices]) if advices else "لا توجد نصائح محددة حالياً."

    system_prompt = f"""أنت PulseKey، مساعد طبي ذكي ومتخصص بتساعد المرضى يفهموا حالتهم الصحية بناءً على قراءاتهم الحيوية الفعلية.

═══════════════════════════════
بيانات المريض الحالية:
═══════════════════════════════
• الحالة العامة     : {status_ar} ({status_en})
• السكر             : {vitals.get('glucose_mg_dl', 'غير متاح')} mg/dL
• ضغط الدم          : {vitals.get('systolic_bp', 'غير متاح')}/{vitals.get('diastolic_bp', 'غير متاح')} mmHg
• النبض             : {vitals.get('heart_rate', 'غير متاح')} bpm
• الحرارة           : {vitals.get('temperature_c', 'غير متاح')} °C
• الأكسجين (SpO2)   : {vitals.get('spo2', 'غير متاح')}%
• النصائح المقترحة  :
{advices_text}
═══════════════════════════════

قواعد يجب اتباعها دايماً:
1. رد دايماً بالعربي والإنجليزي مع بعض
2. اعتمد على بيانات المريض الفعلية في كل رد
3. لو المريض قال عرض (دوخة، ضيق تنفس، تعب، صداع، غيره) اربطه بأقرب قراءة في بياناته
4. لو الحالة Critical 🚨 — اقترح يتصل بطوارئ أو طبيب في كل رد
5. متقدمش تشخيص نهائي أبداً، بس وضّح العلاقة بين العرض والقراءات
6. ردودك تكون واضحة ومختصرة وإنسانية
7. استخدم الإيموجي بشكل مناسب
8. لو سألك عن نصيحة عامة، استخدم النصائح المقترحة في بياناته"""

    try:
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_query}
            ]
        )

        return message.content[0].text

    except Exception as e:
        return f"🤖 عندي مشكلة تقنية دلوقتي، حاول تاني بعد شوية.\nTechnical issue, please try again shortly.\n\nError: {str(e)}"


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
