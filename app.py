from flask import Response
import pickle
import numpy as np
import cv2
import os
import re
import nltk
import logging
import time
from tensorflow.keras.models import load_model
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import requests
from flask import Flask, render_template, request, jsonify


# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("mindcare")
log.info("=== MindCare AI starting up ===")
# ────────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"

# Create uploads folder if not exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ---------------- LOAD MODELS ----------------
log.info("Loading text model (text_model.pkl)...")
text_model = pickle.load(open("models/text_model.pkl", "rb"))
log.info("Loading TF-IDF vectorizer (tfidf.pkl)...")
vectorizer = pickle.load(open("models/tfidf.pkl", "rb"))
log.info("Loading image emotion model (emotion_7class_model.h5)...")
image_model = load_model("models/emotion_7class_model.h5")
log.info("All models loaded successfully.")

# Emotion Labels — must match training order in retrain_image_model.py
# Training order: angry(0), disgust(1), fear(2), happy(3), sad(4), surprise(5), neutral(6)
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Text Labels — actual classes from Combined_Data.csv (status column)
# The model predicts these directly, no manual index mapping needed
text_labels = ["Anxiety", "Bipolar", "Depression", "Normal", "Personality disorder", "Stress", "Suicidal"]
log.info(f"emotion_labels : {emotion_labels}")
log.info(f"text_labels    : {text_labels}")

nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
log.info("NLTK stopwords and lemmatizer ready.")



# ---------------- TEXT CLEANING ----------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|[^a-z\s]", "", text)
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(words)

# ---------------- TEXT PREDICTION ----------------
# Per-label confidence thresholds — calibrated against the Combined_Data.csv dataset biases.
# The dataset is mental-health-forum text so positive everyday words skew toward Depression.
# Lower threshold for serious labels, higher for ambiguous ones.
TEXT_THRESHOLDS = {
    "Suicidal":             0.40,   # critical — keep even at lower confidence
    "Depression":           0.45,   # important — forum data makes this accurate at 0.45+
    "Anxiety":              0.50,   # keep at moderate confidence
    "Bipolar":              0.55,   # needs clearer signal
    "Stress":               0.55,   # easily confused with Normal
    "Personality disorder": 0.60,   # very specific — needs higher confidence
}

def predict_text(text):
    log.debug(f"predict_text called | input length={len(text)}")
    cleaned = clean_text(text)
    log.debug(f"predict_text cleaned: '{cleaned[:100]}'")
    vec = vectorizer.transform([cleaned])

    prediction = text_model.predict(vec)[0]
    proba       = text_model.predict_proba(vec)[0]
    confidence  = float(np.max(proba))
    classes     = list(text_model.classes_)

    # Model returns label strings directly from the CSV status column
    if isinstance(prediction, (int, np.integer)):
        label = text_labels[int(prediction)] if int(prediction) < len(text_labels) else str(prediction)
        log.warning(f"predict_text: got integer prediction {prediction}, mapped to '{label}'")
    else:
        label = str(prediction)

    # If the top prediction is not Normal but confidence is too low,
    # check if the SECOND best label also fails its threshold → default Normal
    if label != "Normal":
        threshold = TEXT_THRESHOLDS.get(label, 0.50)
        if confidence < threshold:
            # Check if the model mislabelled: look at all class probabilities
            # and pick the highest-confidence class that meets its own threshold
            sorted_probs = sorted(zip(classes, proba), key=lambda x: -x[1])
            chosen = "Normal"
            for cls, prob in sorted_probs:
                if cls == "Normal":
                    chosen = "Normal"
                    break
                cls_thr = TEXT_THRESHOLDS.get(cls, 0.50)
                if prob >= cls_thr:
                    chosen = cls
                    break
            log.info(f"predict_text: '{label}' conf={confidence:.2f} < thr={threshold} → reassigned to '{chosen}'")
            label = chosen

    log.debug(f"predict_text result: label='{label}', confidence={confidence:.4f}")
    return label, confidence

# ---------------- IMAGE PREDICTION ----------------
def predict_image(image_path):
    log.debug(f"predict_image called | path={image_path}")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        log.error(f"predict_image: cv2 could not read image at '{image_path}'")
        return "Neutral", 0.0
    img = cv2.resize(img, (48, 48))
    img = img / 255.0
    img = img.reshape(1, 48, 48, 1)

    pred = image_model.predict(img)
    label_index = np.argmax(pred)
    confidence = np.max(pred)

    label = emotion_labels[label_index]
    log.debug(f"predict_image result: label={label}, confidence={confidence:.4f}")
    return label, confidence

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def generate_frames():
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    

    while True:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]


            roi = cv2.equalizeHist(roi)

            roi = cv2.resize(roi, (48, 48))
            roi = roi / 255.0
            roi = roi.reshape(1, 48, 48, 1)


            prediction = image_model.predict(roi)
            confidence = np.max(prediction)
            label_index = np.argmax(prediction)

            if confidence < 0.5:
                 label = "Uncertain"
            else:
                 label = emotion_labels[label_index]


            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0,255,0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    camera.release()

# ===============================
# AI GENERATED SUMMARY FUNCTION
# ===============================

def generate_ai_summary(predicted_emotion):
    
    summaries = {
        "Happy": "The analysis indicates a positive emotional state. The individual appears mentally stable and emotionally balanced.",
        
        "Sad": "The facial/textual indicators suggest emotional distress or low mood. Continuous sadness may require attention and emotional support.",
        
        "Angry": "The detected emotion suggests elevated stress or frustration levels. Stress management strategies may be beneficial.",
        
        "Fear": "The analysis indicates signs of anxiety or fear-related emotional response. Relaxation techniques may help.",
        
        "Disgust": "The detected emotion may indicate dissatisfaction or discomfort toward a situation.",
        
        "Surprise": "The emotion suggests heightened emotional sensitivity or sudden reaction.",
        
        "Neutral": "The emotional state appears balanced without strong positive or negative indicators.",
        
        "Normal": 
        "The textual indicators suggest a stable emotional condition. No significant signs of psychological distress are detected.",

        "Anxiety": 
        "The analysis suggests symptoms related to anxiety, including excessive worry, restlessness, or nervous thoughts. Persistent anxiety may affect daily functioning.",

        "Stress": 
        "The system detected signs of psychological stress. Stress may be related to workload, personal issues, or emotional pressure.",

        "Depression": 
        "The analysis indicates possible depressive patterns such as low mood, hopelessness, or emotional withdrawal. Continuous depressive symptoms require attention.",

        "Bipolar": 
        "The textual patterns suggest mood instability that may align with bipolar-related symptoms, including emotional highs and lows.",

        "Personality disorder": 
        "The analysis suggests behavioral or emotional patterns that may be associated with personality-related disturbances affecting social or emotional regulation.",

        "Suicidal": 
        "⚠️ The system detected high-risk language patterns that may indicate suicidal thoughts or self-harm tendencies. Immediate attention is strongly recommended."
    }
    

    recommendations = {
        "Happy": "Maintain healthy habits and social engagement to sustain positive mental health.",
        
        "Sad": "Consider mindfulness exercises, talking to trusted individuals, or engaging in relaxing activities.",
        
        "Angry": "Deep breathing, meditation, or physical activity may help regulate emotional levels.",
        
        "Fear": "Practice grounding techniques and reduce exposure to stressful triggers.",
        
        "Disgust": "Identify the source of discomfort and approach it calmly.",
        
        "Surprise": "Ensure emotional balance and adequate rest.",
        
        "Neutral": "Continue maintaining a stable and balanced lifestyle.",

           "Normal":
        "Maintain a healthy lifestyle, regular sleep schedule, exercise, and positive social interactions to preserve mental well-being.",

        "Anxiety":
        "Practice deep breathing exercises, meditation, structured routines, and limit exposure to stress triggers. Consider professional support if symptoms persist.",

        "Stress":
        "Adopt stress management techniques such as time management, physical activity, relaxation exercises, and proper rest.",

        "Depression":
        "Engage in small daily activities, maintain social connections, and consider speaking with a mental health professional for guidance.",

        "Bipolar":
        "Mood tracking, structured daily routines, and professional psychiatric evaluation are recommended for effective management.",

        "Personality disorder":
        "Long-term therapy and structured psychological support can help improve emotional regulation and interpersonal functioning.",

        "Suicidal":
        "🚨 Immediate professional help is strongly advised. Please contact a mental health professional, trusted individual, or local emergency services immediately."
    }
    

    return summaries.get(predicted_emotion, ""), recommendations.get(predicted_emotion, "")


# ---------------- ROUTES ----------------

# Home Page
@app.route("/")
def home():
    return render_template("home.html")

# Analyze Page (Tabs Page)
@app.route("/analyze")
def analyze():
    return render_template("analyze.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/webcam")
def webcam():
    return render_template("webcam.html")

@app.route("/video")
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# Prediction Route
@app.route("/predict", methods=["POST"])
def predict():
    log.info("POST /predict called")
    text_result = None
    image_result = None
    final_emotion = None

    # -------- TEXT INPUT --------
    user_text = request.form.get("text")

    if user_text and user_text.strip():
        log.info(f"/predict: text input received, length={len(user_text)}")
        text_label, text_conf = predict_text(user_text)
        text_result = (text_label, round(text_conf * 100, 2))
        final_emotion = text_label
        log.info(f"/predict: text result → label={text_label}, confidence={text_conf*100:.2f}%")

    # -------- IMAGE INPUT --------
    image_file = request.files.get("image")

    if image_file and image_file.filename != "":
        log.info(f"/predict: image input received, filename={image_file.filename}")
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], image_file.filename)
        image_file.save(image_path)
        log.debug(f"/predict: image saved to {image_path}")

        img_label, img_conf = predict_image(image_path)
        image_result = (img_label, round(img_conf * 100, 2))
        final_emotion = img_label
        log.info(f"/predict: image result → label={img_label}, confidence={img_conf*100:.2f}%")

    # If nothing predicted
    if final_emotion is None:
        log.warning("/predict: no text or image input — defaulting to Neutral")
        final_emotion = "Neutral"

    summary, recommendation = generate_ai_summary(final_emotion)
    log.info(f"/predict: final_emotion={final_emotion} → rendering result.html")

    return render_template(
        "result.html",
        text_result=text_result,
        image_result=image_result,
        summary=summary,
        recommendation=recommendation
    )
import requests
import zipfile
import io
from collections import defaultdict
from datetime import datetime

# ── WhatsApp chat parser ──
def parse_whatsapp_chat(text):
    """
    Parse WhatsApp exported chat — handles all known export formats:
      DD/MM/YYYY, HH:MM - Name: msg          (Android 24h)
      DD/MM/YYYY, HH:MM AM - Name: msg       (Android 12h)
      [DD/MM/YYYY, HH:MM:SS AM] Name: msg    (iOS bracket)
      M/D/YY, H:MM AM - Name: msg            (US locale)
    """
    import re

    # Strip BOM / zero-width chars that WhatsApp sometimes adds
    text = text.lstrip('\ufeff\u200e\u200f')

    # One master pattern covering all variants
    pattern = re.compile(
        r'[\["]?'                                   # optional [ or "
        r'(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4})'    # date
        r',?\s+'                                    # separator
        r'(\d{1,2}:\d{2}(?::\d{2})?'               # HH:MM or HH:MM:SS
        r'(?:\s?[AaPp][Mm])?)'                      # optional AM/PM
        r'[\]"]?\s*[-\u2013\u2014]\s*'              # ] or " then dash (–, —, -)
        r'([^:]+?):\s*'                             # sender name (non-greedy)
        r'(.*)'                                     # message body
    )

    messages = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = pattern.match(line)
        if m:
            date_str, time_str, sender, msg = m.groups()
            sender = sender.strip()
            msg = msg.strip()
            # Skip system messages
            if sender.lower() in ('messages and calls are end-to-end encrypted',
                                  'you', '') or not msg:
                continue
            messages.append({'date': date_str, 'sender': sender, 'message': msg})
    return messages

def analyze_whatsapp_messages(messages, target_person=None):
    """Run text model on each message and aggregate per sender."""
    from collections import Counter
    persons = defaultdict(lambda: {'messages': [], 'emotions': Counter()})

    for item in messages:
        sender = item['sender']
        if target_person and target_person.lower() not in sender.lower():
            continue
        msg = item['message']
        if len(msg) < 3 or msg.lower() in ['<media omitted>', 'null', 'this message was deleted']:
            continue
        persons[sender]['messages'].append(msg)

    results = []
    for name, data in persons.items():
        msgs = data['messages']
        if not msgs:
            continue
        # Sample up to 200 messages for speed
        sample = msgs[:200]
        emotion_counts = Counter()
        for m in sample:
            try:
                label, _ = predict_text(m)
                emotion_counts[label] += 1
            except Exception:
                pass

        total = sum(emotion_counts.values()) or 1
        emotion_pcts = {k: round(v/total*100, 1) for k, v in emotion_counts.most_common()}

        # Determine stability
        positive = emotion_counts.get('Healthy', 0) + emotion_counts.get('Normal', 0)
        negative = sum(emotion_counts.get(e, 0) for e in ['Depressed', 'Anxiety', 'Stress', 'Depression', 'Suicidal', 'Bipolar'])
        ratio = positive / (positive + negative + 1)
        if ratio > 0.6:
            stability = 'Stable'
        elif ratio > 0.35:
            stability = 'Moderate'
        else:
            stability = 'Unstable'

        dominant = emotion_counts.most_common(1)[0][0] if emotion_counts else 'Neutral'
        summary, _ = generate_ai_summary(dominant)

        results.append({
            'name': name,
            'message_count': len(msgs),
            'emotions': emotion_pcts,
            'stability': stability,
            'dominant_emotion': dominant,
            'summary': summary or f"Analysis based on {len(msgs)} messages."
        })

    return results

@app.route("/whatsapp")
def whatsapp():
    return render_template("whatsapp.html")

@app.route("/analyze_whatsapp", methods=["POST"])
def analyze_whatsapp():
    file = request.files.get("chat_zip")
    target = request.form.get("target_person", "").strip()

    if not file:
        return jsonify({"error": "No file uploaded."})

    try:
        raw_text = ""
        filename = file.filename.lower()

        if filename.endswith(".zip"):
            zf = zipfile.ZipFile(io.BytesIO(file.read()))
            for name in zf.namelist():
                if name.endswith(".txt"):
                    raw_text = zf.read(name).decode("utf-8", errors="ignore")
                    break
            if not raw_text:
                return jsonify({"error": "No .txt file found inside the ZIP."})
        elif filename.endswith(".txt"):
            raw_text = file.read().decode("utf-8", errors="ignore")
        else:
            return jsonify({"error": "Please upload a .zip or .txt file."})

        messages = parse_whatsapp_chat(raw_text)
        if not messages:
            # Return first 5 lines for debugging
            preview = '\n'.join(raw_text.splitlines()[:5])
            return jsonify({"error": f"Could not parse chat. First lines of file:\n{preview}\n\nMake sure it is a valid WhatsApp export (.txt or .zip without media)."})

        persons_data = analyze_whatsapp_messages(messages, target if target else None)
        if not persons_data:
            return jsonify({"error": "No messages found for the specified person." if target else "No messages could be analyzed."})

        # Date range
        dates = [m['date'] for m in messages]
        date_range = f"{dates[0]} – {dates[-1]}" if dates else "N/A"

        # Overall stability
        stabs = [p['stability'] for p in persons_data]
        overall = 'Stable' if stabs.count('Stable') > len(stabs)/2 else ('Moderate' if stabs.count('Unstable') < len(stabs)/2 else 'Unstable')

        return jsonify({
            "total_messages": len(messages),
            "participants": len(persons_data),
            "date_range": date_range,
            "overall_stability": overall,
            "persons": persons_data
        })

    except Exception as e:
        return jsonify({"error": f"Processing error: {str(e)}"})

# Separate chatbot page
@app.route("/chatbot")
def chatbot():
    return render_template("chatbot.html")

OLLAMA_URL = "http://localhost:11434"
OLLAMA_EXE = r"C:\Users\Lenovo\AppData\Local\Programs\Ollama\ollama.exe"
SYSTEM_PROMPT = (
    "You are a compassionate and empathetic mental health AI assistant. "
    "Your role is to provide emotional support, active listening, and healthy coping strategies. "
    "Keep responses warm, concise (3-5 sentences), and supportive. "
    "Never provide medical diagnoses. If the user expresses crisis or suicidal thoughts, "
    "gently but clearly encourage them to contact a mental health professional or helpline immediately."
)

def get_available_model():
    """Return the first available Ollama model, or None."""
    log.debug(f"get_available_model: checking {OLLAMA_URL}/api/tags ...")
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        log.debug(f"get_available_model: /api/tags status={resp.status_code}")
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            log.debug(f"get_available_model: models found = {[m['name'] for m in models]}")
            if models:
                chosen = models[0]["name"]
                log.debug(f"get_available_model: returning '{chosen}'")
                return chosen
            else:
                log.warning("get_available_model: Ollama is running but NO models are installed!")
    except requests.exceptions.ConnectionError:
        log.warning("get_available_model: Ollama is NOT running (connection refused)")
    except requests.exceptions.Timeout:
        log.warning("get_available_model: Ollama /api/tags timed out after 5s")
    except Exception as e:
        log.error(f"get_available_model: unexpected error — {e}")
    return None

# Check Ollama status endpoint (called by frontend)
@app.route("/check_ollama")
def check_ollama():
    log.info("GET /check_ollama called")
    model = get_available_model()
    if model:
        log.info(f"/check_ollama: Ollama online, model='{model}'")
        return jsonify({"online": True, "model": model})
    log.warning("/check_ollama: Ollama offline or no models available")
    return jsonify({"online": False, "model": None})

# Chat API — real Ollama responses
@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "").strip()
    log.info(f"POST /chat | user message: '{user_message[:100]}'")

    if not user_message:
        log.warning("/chat: empty message received")
        return jsonify({"reply": None, "error": "Empty message received."}), 400

    model = get_available_model()
    if not model:
        log.error("/chat: Ollama not available — cannot generate response")
        return jsonify({"reply": None, "error": "Ollama is not running. Please start Ollama and run: ollama pull llama3.2"}), 503

    log.info(f"/chat: sending prompt to Ollama model='{model}'")
    t_start = time.time()
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model,
                "system": SYSTEM_PROMPT,
                "prompt": user_message,
                "stream": False,
                "options": {"temperature": 0.75, "num_predict": 200}
            },
            timeout=60
        )
        elapsed = time.time() - t_start
        log.info(f"/chat: Ollama responded in {elapsed:.2f}s | HTTP status={resp.status_code}")

        if resp.status_code == 200:
            data = resp.json()
            reply = data.get("response", "").strip()
            if reply:
                log.info(f"/chat: reply received, length={len(reply)} chars")
                log.debug(f"/chat: reply preview = '{reply[:120]}...'")
                return jsonify({"reply": reply, "error": None})
            else:
                log.error(f"/chat: Ollama returned 200 but empty 'response' field. Full JSON: {data}")
                return jsonify({"reply": None, "error": "Ollama returned an empty response."}), 502
        else:
            log.error(f"/chat: Ollama HTTP error {resp.status_code} — {resp.text[:200]}")
            return jsonify({"reply": None, "error": f"Ollama HTTP {resp.status_code}"}), 502

    except requests.exceptions.Timeout:
        elapsed = time.time() - t_start
        log.error(f"/chat: Ollama request timed out after {elapsed:.1f}s (limit=60s)")
        return jsonify({"reply": None, "error": "Ollama took too long to respond (>60s). Try again."}), 504
    except requests.exceptions.ConnectionError:
        log.error("/chat: Lost connection to Ollama mid-request")
        return jsonify({"reply": None, "error": "Lost connection to Ollama."}), 503
    except Exception as e:
        log.exception(f"/chat: Unexpected exception — {e}")
        return jsonify({"reply": None, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
