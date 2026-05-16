from flask import Response
import pickle
import numpy as np
import cv2
import os
import re
import nltk
from tensorflow.keras.models import load_model
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import requests
from flask import Flask, render_template, request, jsonify


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"

# Create uploads folder if not exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ---------------- LOAD MODELS ----------------
text_model = pickle.load(open("models/text_model.pkl", "rb"))
vectorizer = pickle.load(open("models/tfidf.pkl", "rb"))
image_model = load_model("models/emotion_7class_model.h5")

# Emotion Labels for Image Model (EDIT if needed)
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Text Labels (EDIT according to your dataset)
text_labels = ["Healthy", "Depressed"]

nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()



# ---------------- TEXT CLEANING ----------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|[^a-z\s]", "", text)
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(words)

# ---------------- TEXT PREDICTION ----------------
def predict_text(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])

    prediction = text_model.predict(vec)[0]
    confidence = np.max(text_model.predict_proba(vec))

    # If your model outputs 0/1
    if isinstance(prediction, (int, np.integer)):
        label = text_labels[prediction]
    else:
        label = prediction

    return label, confidence

# ---------------- IMAGE PREDICTION ----------------
def predict_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48))
    img = img / 255.0
    img = img.reshape(1, 48, 48, 1)

    pred = image_model.predict(img)
    label_index = np.argmax(pred)
    confidence = np.max(pred)

    label = emotion_labels[label_index]

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

    text_result = None
    image_result = None
    final_emotion = None

    # -------- TEXT INPUT --------
    user_text = request.form.get("text")

    if user_text and user_text.strip():
        text_label, text_conf = predict_text(user_text)
        text_result = (text_label, round(text_conf * 100, 2))
        final_emotion = text_label   # store for summary

    # -------- IMAGE INPUT --------
    image_file = request.files.get("image")

    if image_file and image_file.filename != "":
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], image_file.filename)
        image_file.save(image_path)

        img_label, img_conf = predict_image(image_path)
        image_result = (img_label, round(img_conf * 100, 2))
        final_emotion = img_label   # store for summary

    # If nothing predicted
    if final_emotion is None:
        final_emotion = "Neutral"

    summary, recommendation = generate_ai_summary(final_emotion)

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
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            if models:
                return models[0]["name"]
    except Exception:
        pass
    return None

# Check Ollama status endpoint (called by frontend)
@app.route("/check_ollama")
def check_ollama():
    model = get_available_model()
    if model:
        return jsonify({"online": True, "model": model})
    return jsonify({"online": False, "model": None})

# Chat API — real Ollama responses
@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "").strip()
    if not user_message:
        return jsonify({"reply": "Error: empty message."})

    model = get_available_model()
    if not model:
        return jsonify({"reply": "Error: Ollama not available."})

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
        if resp.status_code == 200:
            reply = resp.json().get("response", "").strip()
            if reply:
                return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"reply": f"Error: {str(e)}"})

    return jsonify({"reply": "Error: no response from model."})

if __name__ == "__main__":
    app.run(debug=True)
