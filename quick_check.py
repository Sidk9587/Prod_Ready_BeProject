import pickle, re, numpy as np
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

THRESHOLDS = {
    "Suicidal":             0.45,
    "Depression":           0.55,
    "Anxiety":              0.55,
    "Bipolar":              0.60,
    "Stress":               0.60,
    "Personality disorder": 0.65,
}

def clean(text):
    text = text.lower()
    text = re.sub(r'http\S+|[^a-z\s]', '', text)
    return ' '.join([lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words])

def predict(text):
    c = clean(text)
    raw = m.predict(v.transform([c]))[0]
    proba = m.predict_proba(v.transform([c]))[0]
    conf = float(np.max(proba))
    if raw != 'Normal':
        thr = THRESHOLDS.get(raw, 0.55)
        final = raw if conf >= thr else 'Normal'
    else:
        final = raw
    return raw, conf, final

with open('models/text_model.pkl','rb') as f: m = pickle.load(f)
with open('models/tfidf.pkl','rb') as f: v = pickle.load(f)

tests = [
    # From your logs — SHOULD be Normal
    ("Life is going well I am happy with my job and enjoying hobbies",    "Normal"),
    ("I am feeling great today spent quality time with my family",         "Normal"),
    ("Had a productive day feeling motivated and grateful for everything",  "Normal"),
    ("I am feeling to talk with everyone and enjoy with my family",        "Normal"),
    # Mental health — SHOULD be detected
    ("I feel completely empty inside nothing brings me joy I cry all day", "Depression"),
    ("I want to kill myself I see no way out",                             "Suicidal"),
    ("I am constantly worrying my heart races and I cannot stop",          "Anxiety"),
    ("Too many deadlines I feel overwhelmed and burned out",               "Stress"),
    ("My mood swings are extreme euphoric one week crash the next",        "Bipolar"),
    # Edge cases
    ("today was okay had lunch with a colleague nothing special",           "Normal"),
    ("I went shopping and had a nice evening at home",                      "Normal"),
]

print(f"{'Input':<55} {'Expected':<22} {'Raw':<22} {'Conf':>5}  {'Final':<22}  OK?")
print('-'*140)
ok_count = 0
for txt, expected in tests:
    raw, conf, final = predict(txt)
    ok = 'YES' if final == expected else 'NO <---'
    if final == expected: ok_count += 1
    print(f"{txt[:54]:<55} {expected:<22} {raw:<22} {conf:>5.2f}  {final:<22}  {ok}")
print(f"\n{ok_count}/{len(tests)} tests passed")
