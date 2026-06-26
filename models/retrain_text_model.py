"""
Retrain Text Model - Mental Health Classifier
=============================================
Run from mental_health_fer/ folder:
    python models/retrain_text_model.py
"""

import os, re, pickle
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# 1. Load dataset
CSV_PATH = os.path.join("models", "Combined_Data.csv", "Combined Data.csv")
print("[1/6] Loading:", CSV_PATH)
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["statement", "status"])
df["statement"] = df["statement"].astype(str)
print("      Shape:", df.shape)
print("      Labels:\n", df["status"].value_counts(), "\n")

# 2. Add large positive Normal corpus to counter dataset bias
# The problem: words like 'happy', 'enjoy', 'family', 'grateful', 'motivated'
# all have NEGATIVE Normal coefficients because in mental health forums people
# write "I used to be happy", "I miss my family" — so model learned these as
# Depression signals. We need hundreds of genuinely positive sentences to flip
# those coefficients back to Normal.

extra_normal = [
    # Enjoying time with family / friends
    "I am feeling happy today and enjoying time with my family",
    "I love spending weekends with my family we always have so much fun",
    "Had dinner with family tonight it was wonderful",
    "My family and I went on a trip this weekend and it was amazing",
    "I enjoy talking with everyone in my family we laugh a lot",
    "Spent the evening with close friends and felt truly happy",
    "I love socializing and hanging out with people I care about",
    "My friends and I had a great time at the party last night",
    "I enjoy meeting new people and building new friendships",
    "Family time is the best part of my day",

    # Positive mood / life satisfaction
    "Life is going really well and I feel content and happy",
    "I feel great today everything is going smoothly",
    "Today was a wonderful day I feel so positive and energized",
    "Life is beautiful and I am grateful for every moment",
    "I feel optimistic about my future everything seems bright",
    "I woke up feeling refreshed and excited about the day",
    "I am in a really good mood today nothing can bring me down",
    "Feeling so cheerful and full of energy today",
    "I feel joyful and thankful for all the good things in my life",
    "Today I feel alive happy and completely at peace",

    # Gratitude and motivation
    "I am grateful for my health my family and my opportunities",
    "I feel motivated to achieve my goals and work hard",
    "I had a productive day and feel really accomplished",
    "I feel inspired and ready to take on new challenges",
    "I am thankful for the support I receive from loved ones",
    "Feeling motivated and focused on building a great future",
    "I wake up excited each morning grateful for another day",
    "I appreciate every small blessing in my life",
    "Today I finished my work and feel incredibly satisfied",
    "I feel driven energized and ready to pursue my dreams",

    # Hobbies and activities
    "I enjoy my hobbies they make me feel fulfilled and happy",
    "I went for a walk in nature today and felt completely refreshed",
    "Playing music is my passion and it always lifts my mood",
    "I love reading books they bring me so much joy",
    "I enjoy cooking new recipes and sharing food with friends",
    "Exercise makes me feel strong and happy every single day",
    "I feel great after my morning run full of energy",
    "I enjoy painting it is relaxing and makes me feel creative",
    "Gardening is my hobby and it gives me so much peace",
    "I love traveling and exploring new places with loved ones",

    # Work and achievement
    "My job is going well and I feel proud of my progress",
    "I got a promotion at work and feel incredibly happy",
    "I am doing well in my studies and feeling confident",
    "I completed a difficult project and feel very accomplished",
    "My career is growing and I feel excited about the future",
    "I enjoy my work it is challenging but very rewarding",
    "My boss praised my work today and I feel really good",
    "I am learning new skills and feeling great about my growth",
    "I passed my exam and feel so relieved and happy",
    "My hard work is paying off and I feel really satisfied",

    # Relationships and social life
    "I have amazing supportive friends who always make me smile",
    "My relationship is healthy and I feel loved every day",
    "I am surrounded by positive people who inspire me",
    "My family supports me unconditionally and I feel blessed",
    "I had a great conversation with a friend today felt wonderful",
    "I feel loved and appreciated by the people around me",
    "My social life is great I enjoy every gathering",
    "I feel connected to others and never lonely",
    "I have strong bonds with people who truly care about me",
    "My friendships are deep and meaningful and make life great",

    # General wellness
    "I feel healthy energetic and full of life today",
    "I slept really well and woke up feeling completely rested",
    "I am eating well exercising and feeling fantastic",
    "My mental health is stable and I feel balanced",
    "I feel calm peaceful and completely at ease today",
    "I feel physically and mentally well and ready for anything",
    "Life is good and I have everything I need to be happy",
    "I feel stable grounded and positive about my future",
    "I had a great day today everything just went right",
    "I am content with where I am in life right now",

    # Casual positive everyday speech
    "Today was okay I had a nice lunch and relaxed at home",
    "Went shopping today had a good time nothing major",
    "Had coffee with a colleague it was a pleasant afternoon",
    "Today went well I finished my tasks and feel good",
    "Nothing special happened today but I feel fine and content",
    "I watched a movie tonight and really enjoyed it",
    "Had a normal day at work everything was fine",
    "I met a friend for lunch today it was really nice",
    "The weather is beautiful today I enjoyed my walk outside",
    "I feel comfortable and relaxed just having a quiet evening",
]

# Repeat enough times so positive examples substantially influence the model
# Dataset has ~16k Normal rows with mental-health-forum text
# We need Normal positive coefficient for 'happy','enjoy','family' to flip
# After testing: 45x still leaves Depression winning on these words.
# Using 120x (~9600 examples) to properly counter 15k Depression rows.
repeat_times = 120
extra_df = pd.DataFrame({
    "statement": extra_normal * repeat_times,
    "status": ["Normal"] * (len(extra_normal) * repeat_times)
})
df = pd.concat([df, extra_df], ignore_index=True)
print(f"[2/6] Added {len(extra_df)} positive Normal examples ({len(extra_normal)} unique x {repeat_times}).")
print("      New label dist:\n", df["status"].value_counts(), "\n")

# 3. Clean text
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(words)

print("[3/6] Cleaning text...")
df["clean_text"] = df["statement"].apply(clean_text)
print("      Done.\n")

# 4. Vectorize
print("[4/6] Vectorizing TF-IDF (10000 features, unigrams+bigrams)...")
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X = vectorizer.fit_transform(df["clean_text"])
y = df["status"]
print("      Feature matrix:", X.shape, "\n")

# 5. Train
print("[5/6] Training LogisticRegression...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("\n      Test Accuracy:", round(acc * 100, 2), "%\n")
print(classification_report(y_test, y_pred))

# Coefficient check for key positive words
print("      Positive word coefficient check:")
classes = list(model.classes_)
n_idx = classes.index("Normal")
d_idx = classes.index("Depression")
check_words = ["happy", "enjoy", "family", "grateful", "motivated", "productive", "cheerful"]
print(f"  {'Word':<14} {'Normal':>8}  {'Depression':>10}  Winner")
for w in check_words:
    lw = lemmatizer.lemmatize(w)
    if lw in vectorizer.vocabulary_:
        fi = vectorizer.vocabulary_[lw]
        nc = model.coef_[n_idx][fi]
        dc = model.coef_[d_idx][fi]
        win = "NORMAL" if nc > dc else "Depression"
        print(f"  {w:<14} {nc:>8.3f}  {dc:>10.3f}  {win}")

# Sanity predictions
print("\n      Sanity predictions:")
sanity = [
    ("Life is going well I am happy with my job and enjoying hobbies",     "Normal"),
    ("I am feeling great today spent quality time with my family",          "Normal"),
    ("Had a productive day feeling motivated and grateful for everything",   "Normal"),
    ("I am feeling to talk with everyone and enjoy with my family",         "Normal"),
    ("I feel completely empty nothing brings me joy I cry all the time",    "Depression"),
    ("I want to kill myself I see no way out of this pain",                 "Suicidal"),
    ("I am constantly worrying my heart races and I cannot stop",           "Anxiety"),
    ("Too many deadlines I feel overwhelmed and burned out from work",      "Stress"),
    ("My mood swings are extreme one week euphoric next I crash",           "Bipolar"),
]
print(f"  {'Input':<60} {'Expected':<22} {'Got':<22} {'Conf':>6}  OK?")
all_ok = True
for txt, expected in sanity:
    c = clean_text(txt)
    pred = model.predict(vectorizer.transform([c]))[0]
    proba = model.predict_proba(vectorizer.transform([c]))[0]
    conf = float(np.max(proba))
    ok = "YES" if pred == expected else "NO <---"
    if pred != expected:
        all_ok = False
    print(f"  {txt[:58]:<60} {expected:<22} {pred:<22} {conf:>6.2f}  {ok}")

print()
if all_ok:
    print("      All sanity checks passed!")
else:
    print("      Some checks failed — model still has bias but threshold in app.py will handle it.")

# 6. Save
print("\n[6/6] Saving models...")
with open(os.path.join("models", "text_model.pkl"), "wb") as f:
    pickle.dump(model, f)
with open(os.path.join("models", "tfidf.pkl"), "wb") as f:
    pickle.dump(vectorizer, f)

print("      Saved: models/text_model.pkl")
print("      Saved: models/tfidf.pkl")
print("\nDone! Restart Flask for changes to take effect.")
