from flask import Flask, request, jsonify
import joblib
import os
import numpy as np
import pandas as pd
import random

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
# دالة الشات المحسّنة
# ─────────────────────────────────────────────
def pulsekey_chatbot_reply(user_query, report_data):
    query = user_query.lower().strip()

    # لو مفيش داتا
    if 'risk_level' not in report_data:
        return random.choice([
            "👋 أهلاً! محتاج تدخل علاماتك الحيوية الأول عشان أقدر أساعدك.\nHello! Please enter your vitals first so I can help you.",
            "😊 مرحباً! ابدأ بإدخال بياناتك الصحية وأنا هساعدك.\nWelcome! Start by entering your health data and I'll assist you.",
            "🩺 لو عايز تعرف حالتك، ادخل قراءاتك الحيوية الأول.\nTo check your health status, please enter your vitals first."
        ])

    risk   = report_data.get('risk_level', 1)
    vitals = report_data.get('vitals', {})
    advices = report_data.get('advice_list', [])

    status_map = {
        0: ("Critical Risk 🚨", "خطر عالي — لازم تتواصل مع الطبيب فوراً!"),
        1: ("Stable ✅",        "مستقر — استمر في نمط حياتك الصحي!"),
        2: ("Medium Risk 🟡",   "خطر متوسط — خد بالك من نفسك وراقب القراءات.")
    }
    status_en, status_ar = status_map.get(risk, ("Unknown", "غير معروف"))

    # تحيات
    if any(w in query for w in ["hi", "hello", "مرحبا", "أهلا", "اهلا", "السلام", "هاي", "هلو"]):
        return random.choice([
            f"👋 أهلاً وسهلاً! أنا PulseKey مساعدك الصحي. حالتك دلوقتي: {status_ar}\nHello! I'm PulseKey, your health assistant. Your current status: {status_en}",
            f"😊 هلا! كيف أقدر أساعدك النهارده؟ حالتك: {status_ar}\nHey! How can I help you today? Your status: {status_en}",
            f"🩺 أهلاً! أنا هنا أساعدك. حالتك الحالية: {status_ar}\nHi there! I'm here to help. Your current status: {status_en}"
        ])

    # شكر
    if any(w in query for w in ["شكرا", "شكراً", "thanks", "thank you", "تسلم", "يسلمو"]):
        return random.choice([
            "😊 العفو! أنا دايماً هنا لو محتاج أي حاجة.\nYou're welcome! I'm always here if you need anything.",
            "🙏 بالعافية! صحتك تهمنا.\nMy pleasure! Your health matters to us.",
            "💙 أي خدمة! اهتم بصحتك.\nAnytime! Take care of yourself."
        ])

    # الحالة الصحية
    if any(w in query for w in ["status", "report", "حالة", "تقرير", "عامل ايه", "ازيك", "أزيك", "how am i", "عاملة ايه"]):
        responses = [
            f"📊 حالتك الصحية دلوقتي: {status_ar}\n💡 {random.choice(advices) if advices else 'استمر في متابعة قراءاتك.'}\nYour health status: {status_en}",
            f"🩺 بناءً على قراءاتك الأخيرة، حالتك: {status_ar}\nBased on your latest readings, your status is: {status_en}",
            f"📋 تقريرك الصحي يقول: {status_ar}\n{random.choice(advices) if advices else ''}\nYour health report says: {status_en}"
        ]
        return random.choice(responses)

    # النصائح
    if any(w in query for w in ["advice", "نصيحة", "اعمل ايه", "مساعدة", "help", "ايه اللي", "what should"]):
        formatted = "\n".join([f"• {a}" for a in advices]) if advices else "✅ حالتك كويسة، استمر في نمط حياتك الصحي."
        return random.choice([
            f"💡 بناءً على بياناتك، أنصحك بالآتي:\n{formatted}\n\nBased on your data, here's my advice:\n{formatted}",
            f"🩺 إليك نصائحي الطبية:\n{formatted}\n\nHere are my medical tips:\n{formatted}",
            f"📋 خطوات مهمة ليك:\n{formatted}\n\nImportant steps for you:\n{formatted}"
        ])

    # السكر
    if any(w in query for w in ["sugar", "glucose", "سكر", "جلوكوز"]):
        val = vitals.get('glucose_mg_dl', 'N/A')
        if val != 'N/A':
            if float(val) > 140:
                comment = "⚠️ مرتفع شوية، قلل السكريات. / High, reduce sugar intake."
            elif float(val) < 70:
                comment = "⚠️ منخفض، تناول شيء حلو. / Low, have something sweet."
            else:
                comment = "✅ طبيعي وتمام. / Normal and good."
        else:
            comment = ""
        return random.choice([
            f"🩸 قراءة السكر: {val} mg/dL — {comment}",
            f"📊 مستوى الجلوكوز عندك: {val} mg/dL\n{comment}",
            f"💉 آخر قراءة للسكر في دمك: {val} mg/dL\n{comment}"
        ])

    # الضغط
    if any(w in query for w in ["pressure", "bp", "ضغط", "blood pressure"]):
        sys = vitals.get('systolic_bp', 'N/A')
        dia = vitals.get('diastolic_bp', 'N/A')
        if sys != 'N/A' and float(sys) > 130:
            comment = "⚠️ مرتفع، قلل الأملاح والتوتر. / High, reduce salt and stress."
        else:
            comment = "✅ طبيعي. / Normal."
        return random.choice([
            f"💓 ضغط الدم: {sys}/{dia} mmHg — {comment}",
            f"📊 قراءة الضغط عندك: {sys}/{dia} mmHg\n{comment}",
            f"🩺 ضغطك الانقباضي/الانبساطي: {sys}/{dia} mmHg\n{comment}"
        ])

    # النبض
    if any(w in query for w in ["heart rate", "pulse", "نبض", "قلب", "heart"]):
        val = vitals.get('heart_rate', 'N/A')
        if val != 'N/A':
            comment = "⚠️ مرتفع، استرح شوية. / High, take some rest." if float(val) > 100 else "✅ طبيعي. / Normal."
        else:
            comment = ""
        return random.choice([
            f"❤️ معدل ضربات القلب: {val} bpm — {comment}",
            f"💗 نبضك: {val} bpm\n{comment}",
            f"🫀 قراءة القلب عندك: {val} bpm\n{comment}"
        ])

    # الحرارة
    if any(w in query for w in ["temp", "temperature", "حرارة", "درجة"]):
        val = vitals.get('temperature_c', 'N/A')
        if val != 'N/A':
            comment = "⚠️ حرارتك مرتفعة، خد مسكن. / Fever detected, take a painkiller." if float(val) > 37.5 else "✅ طبيعية. / Normal."
        else:
            comment = ""
        return random.choice([
            f"🌡️ درجة حرارتك: {val} °C — {comment}",
            f"🌡️ الحرارة: {val} °C\n{comment}",
            f"📊 قراءة الحرارة: {val} °C\n{comment}"
        ])

    # الأكسجين
    if any(w in query for w in ["spo2", "oxygen", "أكسجين", "اكسجين", "o2"]):
        val = vitals.get('spo2', 'N/A')
        if val != 'N/A':
            comment = "⚠️ منخفض، خد نفس عميق واستشر الطبيب. / Low, breathe deeply and consult a doctor." if float(val) < 95 else "✅ ممتاز. / Excellent."
        else:
            comment = ""
        return random.choice([
            f"🫁 نسبة الأكسجين: {val}% — {comment}",
            f"💨 تشبع الأكسجين عندك: {val}%\n{comment}",
            f"📊 قراءة الـ SpO2: {val}%\n{comment}"
        ])

    # شرح السكري
    if any(w in query for w in ["ما هو السكر", "diabetes", "سكري", "what is diabetes"]):
        return random.choice([
            "🩸 مرض السكري هو ارتفاع نسبة السكر في الدم نتيجة خلل في هرمون الإنسولين. نحن هنا نساعدك في مراقبته.\nDiabetes is high blood sugar caused by insulin issues. We're here to help you monitor it.",
            "💉 السكري مرض مزمن بيأثر على طريقة معالجة الجسم للسكر. ممكن يتحكم فيه بالغذاء والدواء والرياضة.\nDiabetes affects how your body processes sugar. It can be managed with diet, medication, and exercise."
        ])

    # شرح الضغط
    if any(w in query for w in ["ما هو الضغط", "blood pressure", "hypertension", "ضغط الدم"]):
        return random.choice([
            "💓 ضغط الدم هو قوة دفع الدم على جدران الشرايين. الارتفاع المستمر فوق 130/80 يحتاج متابعة.\nBlood pressure is the force of blood on artery walls. Consistently above 130/80 needs monitoring.",
            "🩺 ارتفاع ضغط الدم ممكن يؤثر على القلب والكلى. الغذاء الصحي والرياضة بيساعدوا في تنظيمه.\nHigh blood pressure can affect heart and kidneys. Healthy diet and exercise help regulate it."
        ])

    # رد افتراضي
    return random.choice([
        "🤖 أنا مساعد PulseKey الصحي! اسألني عن: حالتك، السكر، الضغط، النبض، الحرارة، أو الأكسجين.\nI'm PulseKey health assistant! Ask me about: your status, sugar, pressure, pulse, temperature, or oxygen.",
        "😊 مش فاهم سؤالك كويس! حاول تسأل عن: حالتك الصحية، السكر، الضغط، النبض، أو نصيحة طبية.\nI didn't quite get that! Try asking about: your health status, sugar, pressure, pulse, or medical advice.",
        "🩺 يمكنك سؤالي عن أي من قراءاتك الحيوية أو حالتك الصحية العامة.\nYou can ask me about any of your vital readings or your general health status."
    ])


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
