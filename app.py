from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import numpy as np
import pandas as pd
import google.generativeai as genai

app = Flask(__name__)
CORS(app)

# ─── API Key من Railway Variables ──────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FEATURES_ORDER = [
    'age', 'gender', 'diabetes', 'hypertension', 'heart_disease',
    'glucose_mg_dl', 'systolic_bp', 'diastolic_bp', 'heart_rate',
    'temperature_c', 'spo2'
]

NUMERIC_COLS = ['glucose_mg_dl', 'systolic_bp', 'diastolic_bp', 'heart_rate', 'temperature_c', 'spo2']


# ─────────────────────────────────────────────
# تحميل الموديل والسكيلر
# ─────────────────────────────────────────────
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
# الشات بوت — الدكتور PulseKey
# ─────────────────────────────────────────────
def pulsekey_chatbot_reply(user_query, report_data):

    if not report_data or 'risk_level' not in report_data:
        return (
            "مرحباً! أنا الدكتور PulseKey. "
            "لم أجد بياناتك الصحية بعد — "
            "يرجى إدخال قراءاتك الحيوية أولاً حتى أتمكن من مساعدتك."
        )

    risk    = report_data.get('risk_level', 1)
    vitals  = report_data.get('vitals',     {})
    advices = report_data.get('advice_list', report_data.get('advice', []))

    glucose = vitals.get('glucose_mg_dl', 'N/A')
    sys_bp  = vitals.get('systolic_bp',   'N/A')
    dia_bp  = vitals.get('diastolic_bp',  'N/A')
    hr      = vitals.get('heart_rate',    'N/A')
    spo2    = vitals.get('spo2',          'N/A')
    temp    = vitals.get('temperature_c', 'N/A')
    age     = vitals.get('age',           'N/A')
    gender  = 'ذكر'  if vitals.get('gender')       == 1 else 'أنثى'
    diab    = 'نعم'  if vitals.get('diabetes')           else 'لا'
    htn     = 'نعم'  if vitals.get('hypertension')       else 'لا'
    hd      = 'نعم'  if vitals.get('heart_disease')      else 'لا'

    risk_map = {
        0: "خطر عالٍ (High Risk) — يستلزم تدخلاً طبياً فورياً",
        1: "مستقر (Stable)        — الحالة جيدة",
        2: "خطر متوسط (Medium)   — يحتاج متابعة دقيقة"
    }

    advice_text = '\n'.join([f"  - {a}" for a in advices]) if advices else "  - لا توجد نصائح إضافية"

    system_prompt = f"""أنت الدكتور PulseKey، طبيب متخصص في أمراض القلب والأمراض المزمنة بخبرة أكثر من 20 سنة.
أنت تتحدث مباشرةً مع مريضك بناءً على نتائج فحوصاته الحقيقية.

══════════════════════════════════
ملف المريض الطبي
══════════════════════════════════
العمر: {age} سنة | الجنس: {gender}
السكري: {diab} | ارتفاع الضغط: {htn} | أمراض القلب: {hd}

القراءات الحيوية:
  السكر في الدم  : {glucose} mg/dL
  ضغط الدم       : {sys_bp}/{dia_bp} mmHg
  معدل القلب     : {hr} bpm
  تشبع الأكسجين  : {spo2}%
  درجة الحرارة   : {temp} درجة مئوية

تقييم الخطورة: {risk_map.get(risk, 'غير محدد')}

النصائح الطبية:
{advice_text}
══════════════════════════════════

تعليمات أسلوبك كطبيب:
- تكلم المريض بصيغة المخاطب المباشر: "ضغطك"، "سكرك"، "قلبك"، "حالتك"
- اذكر الأرقام الفعلية دائماً في ردك — لا كلام عام أبداً
- اشرح ما تعنيه القراءة ولماذا هي جيدة أو مقلقة
- قدّم نصائح طبية دقيقة ومخصصة لحالته تحديداً
- إذا كان الخطر عالياً (risk=0): قل له يروح الطوارئ فوراً مع شرح السبب
- إذا كان الخطر متوسطاً (risk=2): قل له يتصل بطبيبه خلال 24-48 ساعة
- إذا كان مستقراً (risk=1): اثنِ عليه وشجّعه على الاستمرار
- اختم دائماً بجملة واضحة تلخص وضعه وتوجيهك له
- إذا تكلم بالعربية رد بالعربية، وإذا تكلم بالإنجليزية رد بالإنجليزية
- لا تقل أبداً "أنا ذكاء اصطناعي" — أنت الدكتور PulseKey
- الرد بين 4 و7 جمل — مركّز ومفيد
"""

    try:
        model_ai = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction=system_prompt
        )
        response = model_ai.generate_content(user_query)
        return response.text

    except Exception as e:
        err_str = str(e)
        if "API_KEY_INVALID" in err_str or "API key not valid" in err_str:
            return "خطأ في مفتاح API — يرجى التحقق من الإعدادات."
        if "429" in err_str or "quota" in err_str.lower():
            return "الخدمة مشغولة حالياً، يرجى المحاولة بعد لحظات."
        return f"حدث خطأ: {err_str}"


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

        raw_vitals = json_data['data'] if isinstance(raw, dict) else dict(zip(FEATURES_ORDER, raw))
        advices = []
        if raw_vitals.get('glucose_mg_dl', 0) > 140:
            advices.append("السكر مرتفع: قلل السكريات والكربوهيدرات")
        if raw_vitals.get('systolic_bp', 0) > 130:
            advices.append("الضغط مرتفع: قلل الأملاح وتجنب التوتر")
        if raw_vitals.get('spo2', 100) < 95:
            advices.append("الأكسجين منخفض: تنفس بعمق واستشر الطبيب")
        if raw_vitals.get('heart_rate', 75) > 100:
            advices.append("النبض مرتفع: استرح وتجنب المجهود")
        if res == 0:
            advices.append("حالة حرجة: توجه للطوارئ فوراً")
        elif res == 2:
            advices.append("حالة متوسطة: راجع طبيبك خلال 24-48 ساعة")
        if not advices:
            advices.append("علاماتك الحيوية مستقرة — استمر في نمط حياتك الصحي")

        return jsonify({
            "status"    : "success",
            "risk_level": res,
            "label"     : mapping.get(res, "Unknown"),
            "vitals"    : raw_vitals,
            "advice"    : advices
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
        report_data = json_data.get('report_context', json_data.get('report', {}))

        if not user_query:
            return jsonify({"status": "error", "message": "No 'message' key provided"}), 400

        reply = pulsekey_chatbot_reply(user_query, report_data)

        return jsonify({
            "status": "success",
            "reply":  reply
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route('/api/ai/chat-with-context', methods=['POST'])
def chat_with_context():
    try:
        json_data = request.get_json()

        if not json_data:
            return jsonify({"status": "error", "message": "JSON body is empty"}), 400

        user_query  = json_data.get('message', '').strip()
        report_data = json_data.get('report_context', {})

        if not user_query:
            return jsonify({"status": "error", "message": "No 'message' key provided"}), 400

        reply = pulsekey_chatbot_reply(user_query, report_data)

        return jsonify({
            "status": "success",
            "reply":  reply,
            "timestamp": pd.Timestamp.utcnow().isoformat()
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


# مهم لـ Railway
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
