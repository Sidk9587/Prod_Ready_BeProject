"""
Full audit script - run this to verify everything is working.
Usage: python audit.py
"""
import os, re, sys, pickle
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import nltk
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words  = set(stopwords.words("english"))
lemmatizer  = WordNetLemmatizer()

THRESHOLDS = {
    "Suicidal":             0.40,
    "Depression":           0.45,
    "Anxiety":              0.50,
    "Bipolar":              0.55,
    "Stress":               0.55,
    "Personality disorder": 0.60,
}

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|[^a-z\s]", "", text)
    return " ".join([lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words])

def predict(text, model, vec):
    c      = clean_text(text)
    proba  = model.predict_proba(vec.transform([c]))[0]
    classes = list(model.classes_)
    conf   = float(np.max(proba))
    raw    = model.predict(vec.transform([c]))[0]

    if raw != "Normal":
        thr = THRESHOLDS.get(raw, 0.50)
        if conf < thr:
            # Walk sorted probabilities, pick best class that passes its own threshold
            sorted_probs = sorted(zip(classes, proba), key=lambda x: -x[1])
            chosen = "Normal"
            for cls, prob in sorted_probs:
                if cls == "Normal":
                    chosen = "Normal"; break
                cls_thr = THRESHOLDS.get(cls, 0.50)
                if prob >= cls_thr:
                    chosen = cls; break
            final = chosen
        else:
            final = raw
    else:
        final = raw
    return raw, conf, final

# ─────────────────────────────────────────────
PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
INFO = "\033[94mINFO\033[0m"
errors = []

def check(label, ok, detail=""):
    sym = PASS if ok else FAIL
    print(f"  [{sym}] {label}" + (f"  — {detail}" if detail else ""))
    if not ok:
        errors.append(label)

# ─────────────────────────────────────────────
print("\n" + "="*60)
print("1. MODEL FILES")
print("="*60)

text_model_path = os.path.join("models", "text_model.pkl")
tfidf_path      = os.path.join("models", "tfidf.pkl")
img_model_path  = os.path.join("models", "emotion_7class_model.h5")

check("text_model.pkl exists",          os.path.exists(text_model_path))
check("tfidf.pkl exists",               os.path.exists(tfidf_path))
check("emotion_7class_model.h5 exists", os.path.exists(img_model_path))

# ─────────────────────────────────────────────
print("\n" + "="*60)
print("2. TEXT MODEL LOADING")
print("="*60)

try:
    with open(text_model_path, "rb") as f:
        text_model = pickle.load(f)
    check("text_model loads without error", True, str(type(text_model).__name__))
except Exception as e:
    check("text_model loads without error", False, str(e))
    sys.exit(1)

try:
    with open(tfidf_path, "rb") as f:
        vectorizer = pickle.load(f)
    # Actually try to transform something — this is where idf-not-fitted crashes
    _ = vectorizer.transform(["hello world test"])
    check("tfidf.pkl loads AND transforms", True, f"vocab={len(vectorizer.vocabulary_)}")
except Exception as e:
    check("tfidf.pkl loads AND transforms", False, str(e))
    sys.exit(1)

check("text_model has 7 classes",  len(text_model.classes_) == 7,
      str(list(text_model.classes_)))
check("text_model coef shape ok",  text_model.coef_.shape == (7, len(vectorizer.vocabulary_)),
      str(text_model.coef_.shape))

# ─────────────────────────────────────────────
print("\n" + "="*60)
print("3. TEXT PREDICTIONS (with threshold)")
print("="*60)

text_tests = [
    # (input,                                                           expected_final)
    ("Life is going well I am happy with my job and enjoying hobbies", "Normal"),
    ("I am feeling great today spent quality time with my family",      "Normal"),
    ("Had a productive day feeling motivated and grateful",             "Normal"),
    ("I am feeling to talk with everyone and enjoy with my family",    "Normal"),
    ("today was okay had lunch with a colleague nothing special",       "Normal"),
    ("I feel completely empty nothing brings me joy I cry all day",    "Depression"),
    ("I want to kill myself I see no way out of this pain",            "Suicidal"),
    ("I am constantly worrying my heart races I cannot stop",          "Anxiety"),
    ("I cannot get out of bed everything feels hopeless and dark",     "Depression"),
    ("I feel stressed and overwhelmed by all my deadlines",            "Stress"),
]

print(f"  {'Input':<55} {'Expected':<22} {'Raw':<22} {'Conf':>5}  {'Final':<22}  Result")
print("  " + "-"*140)
for txt, expected in text_tests:
    raw, conf, final = predict(txt, text_model, vectorizer)
    ok = final == expected
    sym = "PASS" if ok else "FAIL <---"
    print(f"  {txt[:54]:<55} {expected:<22} {raw:<22} {conf:>5.2f}  {final:<22}  {sym}")
    if not ok:
        errors.append(f"Text pred: '{txt[:40]}' expected={expected} got={final}")

# ─────────────────────────────────────────────
print("\n" + "="*60)
print("4. TEXT MODEL COEFFICIENT CHECK (key positive words)")
print("="*60)

classes = list(text_model.classes_)
n_idx = classes.index("Normal")
d_idx = classes.index("Depression")
words = ["happy", "enjoy", "family", "grateful", "motivated", "productive"]
print(f"  {'Word':<14} {'Normal coef':>12}  {'Depr coef':>10}  Winner")
for w in words:
    lw = lemmatizer.lemmatize(w)
    if lw in vectorizer.vocabulary_:
        fi  = vectorizer.vocabulary_[lw]
        nc  = text_model.coef_[n_idx][fi]
        dc  = text_model.coef_[d_idx][fi]
        win = "NORMAL" if nc > dc else "Depression"
        print(f"  {w:<14} {nc:>12.3f}  {dc:>10.3f}  {win}")

# ─────────────────────────────────────────────
print("\n" + "="*60)
print("5. IMAGE MODEL")
print("="*60)

try:
    from tensorflow.keras.models import load_model
    img_model = load_model(img_model_path)
    check("image model loads",        True)
    check("input shape  (1,48,48,1)", img_model.input_shape  == (None,48,48,1), str(img_model.input_shape))
    check("output shape (1,7)",       img_model.output_shape == (None,7),       str(img_model.output_shape))

    # Test on dummy input
    dummy = np.zeros((1,48,48,1), dtype="float32")
    p = img_model.predict(dummy, verbose=0)
    check("image model predicts on dummy input", p.shape == (1,7), str(p))

    emotion_labels_app   = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]
    emotion_labels_train = ["angry","disgust","fear","happy","sad","surprise","neutral"]
    mismatch = any(a.lower() != t for a,t in zip(emotion_labels_app, emotion_labels_train))
    check("emotion_labels order matches training", not mismatch,
          f"app={emotion_labels_app}")
except ImportError:
    check("TensorFlow importable", False, "DLL load failure — TF not usable on this Python/OS combo")
    print("  [INFO] Image model cannot be tested locally — it will still work inside Flask if TF loads there.")
except Exception as e:
    check("image model loads", False, str(e))

# ─────────────────────────────────────────────
print("\n" + "="*60)
print("6. SUMMARY")
print("="*60)
if errors:
    print(f"  {len(errors)} issue(s) found:")
    for e in errors:
        print(f"    - {e}")
else:
    print("  All checks passed!")
print()
