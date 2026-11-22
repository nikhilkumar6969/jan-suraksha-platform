import json
import os
import random

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

# ============================================================
# Create folders
# ============================================================

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ============================================================
# Synthetic Dataset Generation
# ============================================================

random.seed(42)

# ------------------------------
# Scam Templates (English)
# ------------------------------

scam_english = [
    "Your UPI account is blocked. Verify immediately at {}",
    "KYC expired. Update bank details now at {}",
    "OTP required to unlock your account. Share OTP immediately.",
    "Digital arrest notice issued. Pay penalty today.",
    "Courier held by customs. Pay processing fee at {}",
    "Loan approved instantly. Click {} to receive funds.",
    "Your bank account will be suspended. Verify at {}",
    "Income tax refund pending. Confirm details at {}",
    "UPI reward pending. Claim now at {}",
    "Security alert. Reconfirm account through {}",
]

# ------------------------------
# Scam Templates (Hindi)
# ------------------------------

scam_hindi = [
    "आपका यूपीआई खाता बंद होने वाला है। तुरंत सत्यापन करें {}",
    "आपकी केवाईसी समाप्त हो गई है। अभी अपडेट करें {}",
    "खाता अनलॉक करने के लिए ओटीपी साझा करें।",
    "डिजिटल गिरफ्तारी नोटिस जारी हुआ है। तुरंत भुगतान करें।",
    "आपका पार्सल कस्टम में रुका है। शुल्क जमा करें {}",
    "तुरंत लोन प्राप्त करें। आवेदन करें {}",
    "बैंक खाता निलंबित होने वाला है। सत्यापन करें {}",
    "रिफंड प्राप्त करने के लिए जानकारी अपडेट करें {}",
    "इनाम प्राप्त करने हेतु लिंक खोलें {}",
    "सुरक्षा जांच हेतु तुरंत लॉगिन करें {}",
]

# ------------------------------
# Scam Templates (Hinglish)
# ------------------------------

scam_hinglish = [
    "Dear customer, aapka UPI block ho gaya hai. Verify now {}",
    "KYC update nahi hua. Account freeze ho sakta hai {}",
    "OTP share karo warna account band ho jayega.",
    "Digital arrest notice. Fine pay karo abhi.",
    "Courier customs me atka hai. Fee bhar do {}",
    "Instant loan approved. Paisa lene ke liye click {}",
    "Reward jeeto. Link kholo {}",
    "Bank verification pending hai {}",
    "Aapka account suspend hone wala hai {}",
    "UPI cashback claim karo {}",
]

# ------------------------------
# Genuine Templates (English)
# ------------------------------

genuine_english = [
    "Your account was credited with Rs {}.",
    "Your account was debited with Rs {}.",
    "Monthly salary has been credited.",
    "ATM cash withdrawal successful.",
    "Electricity bill payment received.",
    "Thanks for shopping with us.",
    "Meeting at 5 PM today.",
    "Dinner is ready at home.",
    "Please call when you reach.",
    "Train ticket confirmed successfully.",
]

# ------------------------------
# Genuine Templates (Hindi)
# ------------------------------

genuine_hindi = [
    "आपके खाते में ₹{} जमा किए गए हैं।",
    "आपके खाते से ₹{} निकाले गए हैं।",
    "मासिक वेतन जमा कर दिया गया है।",
    "बिजली बिल का भुगतान प्राप्त हुआ।",
    "कृपया घर पहुँचकर फोन करें।",
    "आज शाम 5 बजे बैठक है।",
    "यात्रा टिकट सफलतापूर्वक बुक हो गई।",
    "राशि प्राप्त होने की पुष्टि की जाती है।",
    "आपका बैंक स्टेटमेंट उपलब्ध है।",
    "परिवार के साथ रात का भोजन 8 बजे है।",
]

# ------------------------------
# Genuine Templates (Hinglish)
# ------------------------------

genuine_hinglish = [
    "Salary account me credit ho gayi hai.",
    "Rs {} successfully debit hua hai.",
    "Meeting kal 11 baje hai.",
    "Papa ghar pahunch gaye hain.",
    "UPI payment successful of Rs {}.",
    "Lunch ke liye 2 baje milte hain.",
    "Ticket confirm ho gaya hai.",
    "EMI payment successfully received.",
    "Account balance updated successfully.",
    "Family function Sunday ko hai.",
]

fake_domains = [
    "verify-now.net",
    "secure-wallet-check.com",
    "bank-alerts.xyz",
    "kyc-update-fast.net",
    "upi-safe-check.online",
    "claim-bonus.click",
    "refund-security.site",
]

amounts = [150, 499, 999, 1200, 2500, 5000, 7800, 12000]

records = []

# ============================================================
# 400 Scam Messages
# ============================================================

for _ in range(134):
    template = random.choice(scam_english)
    url = f"http://{random.choice(fake_domains)}"
    records.append(
        [template.format(url), "Scam"]
    )

for _ in range(133):
    template = random.choice(scam_hindi)
    url = f"http://{random.choice(fake_domains)}"
    records.append(
        [template.format(url), "Scam"]
    )

for _ in range(133):
    template = random.choice(scam_hinglish)
    url = f"http://{random.choice(fake_domains)}"
    records.append(
        [template.format(url), "Scam"]
    )

# ============================================================
# 400 Genuine Messages
# ============================================================

for _ in range(134):
    template = random.choice(genuine_english)
    records.append(
        [template.format(random.choice(amounts)), "Genuine"]
        if "{}" in template
        else [template, "Genuine"]
    )

for _ in range(133):
    template = random.choice(genuine_hindi)
    records.append(
        [template.format(random.choice(amounts)), "Genuine"]
        if "{}" in template
        else [template, "Genuine"]
    )

for _ in range(133):
    template = random.choice(genuine_hinglish)
    records.append(
        [template.format(random.choice(amounts)), "Genuine"]
        if "{}" in template
        else [template, "Genuine"]
    )

random.shuffle(records)

df = pd.DataFrame(records, columns=["message", "label"])

dataset_path = "data/scam_dataset.csv"
df.to_csv(dataset_path, index=False, encoding="utf-8")

print(f"\nDataset saved: {dataset_path}")
print(f"Total Samples: {len(df)}")

# ============================================================
# Train/Test Split (80/20 Stratified)
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    df["message"],
    df["label"],
    test_size=0.20,
    random_state=42,
    stratify=df["label"],
)

# ============================================================
# Model Pipeline
# ============================================================

word_features = TfidfVectorizer(
    ngram_range=(1, 2),
    lowercase=True,
)

char_features = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
)

features = FeatureUnion(
    [
        ("word_tfidf", word_features),
        ("char_tfidf", char_features),
    ]
)

pipeline = Pipeline(
    [
        ("features", features),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
            ),
        ),
    ]
)

# ============================================================
# Train
# ============================================================

pipeline.fit(X_train, y_train)

# ============================================================
# Evaluate
# ============================================================

preds = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, preds)
precision = precision_score(
    y_test,
    preds,
    pos_label="Scam"
)
recall = recall_score(
    y_test,
    preds,
    pos_label="Scam"
)
f1 = f1_score(
    y_test,
    preds,
    pos_label="Scam"
)

cm = confusion_matrix(
    y_test,
    preds,
    labels=["Genuine", "Scam"]
)

metrics = {
    "accuracy": round(float(accuracy), 4),
    "precision": round(float(precision), 4),
    "recall": round(float(recall), 4),
    "f1_score": round(float(f1), 4),
    "confusion_matrix": cm.tolist(),
}

# ============================================================
# Save Metrics
# ============================================================

metrics_path = "models/metrics.json"

with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)

# ============================================================
# Save Model
# ============================================================

model_path = "models/scam_model.joblib"
joblib.dump(pipeline, model_path)

# ============================================================
# Print Report
# ============================================================

print("\n" + "=" * 60)
print("CYBER FRAUD MESSAGE DETECTOR")
print("=" * 60)

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, preds))

print("\nSaved Files:")
print(f"Dataset : {dataset_path}")
print(f"Model   : {model_path}")
print(f"Metrics : {metrics_path}")
print("=" * 60)