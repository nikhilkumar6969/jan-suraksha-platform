"""
utils.py

Offline utilities for Streamlit Cyber Fraud Detection App
- Zero external API calls
- Local ML inference
- Offline Whisper transcription
- Local SQLite reports DB
- OpenCV QR analysis
"""
import os
import random
import re
import sqlite3
import urllib.parse
from datetime import datetime

import cv2
import joblib
import numpy as np
import pandas as pd
from faster_whisper import WhisperModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================================
# CONFIG
# ==========================================================

DB_PATH = "data/scam_reports.db"

CHHATTISGARH_DISTRICTS = [
    "Raipur",
    "Bastar",
    "Bilaspur",
    "Durg",
    "Raigarh",
    "Korba",
]

SCAM_TYPES = [
    "OTP Scam",
    "Fake QR",
    "Digital Arrest",
    "Scam Call",
    "Loan App",
    "Fake KYC",
]

SUSPICIOUS_NAMES = [
    "payment",
    "merchant",
    "reward",
    "cashback",
    "support",
    "helpdesk",
    "service",
]

SCAM_PHRASES = {
    "Digital Arrest": [
        "digital arrest",
        "digital गिरफ्तारी",
        "video call police",
        "ed notice",
        "cbi warrant",
        "police warrant",
    ],
    "OTP Scam": [
        "share otp",
        "tell otp",
        "otp batao",
        "otp batayiye",
        "verification code",
    ],
    "Fake KYC": [
        "kyc update",
        "complete kyc",
        "bank kyc",
        "re-kyc",
        "verify your kyc",
    ],
    "Customs Parcel": [
        "customs parcel",
        "courier hold",
        "parcel detained",
        "shipment blocked",
    ],
    "Scam Call": [
        "urgently pay",
        "account freeze",
        "verify account",
        "penalty payment",
    ],
}


# ==========================================================
# PII SCRUBBING
# ==========================================================

def scrub_pii(text: str) -> str:
    """
    Redact sensitive information before storage.
    """

    if not isinstance(text, str):
        return ""

    # Email
    text = re.sub(
        r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",
        "[REDACTED]",
        text,
    )

    # Mobile numbers
    text = re.sub(
        r"(?<!\d)(?:\+91[- ]?)?[6-9]\d{9}(?!\d)",
        "[REDACTED]",
        text,
    )

    # Aadhaar-like 12-digit
    text = re.sub(
        r"(?<!\d)\d{12}(?!\d)",
        "[REDACTED]",
        text,
    )

    # Bank account/card numbers (13-19 digits)
    text = re.sub(
        r"(?<!\d)\d{13,19}(?!\d)",
        "[REDACTED]",
        text,
    )

    return text


# ==========================================================
# FEATURE EXPLANATION HELPERS
# ==========================================================

def _extract_top_features(text, model, top_n=5):
    """
    Explain LogisticRegression prediction by obtaining
    TF-IDF feature contributions.
    """

    try:
        features_step = model.named_steps["features"]
        classifier = model.named_steps["classifier"]

        x = features_step.transform([text])

        prediction = classifier.predict([text])[0]

        class_index = list(classifier.classes_).index(prediction)

        coef = classifier.coef_[class_index]

        contributions = x.multiply(coef)

        values = contributions.toarray()[0]

        feature_names = features_step.get_feature_names_out()

        top_idx = np.argsort(values)[::-1][:top_n]

        top_features = []

        for idx in top_idx:
            if values[idx] > 0:
                top_features.append(
                    {
                        "feature": str(feature_names[idx]),
                        "weight": round(float(values[idx]), 4),
                    }
                )

        return top_features

    except Exception:
        return []


# ==========================================================
# PREDICTION + EXPLANATION
# ==========================================================

def predict_and_explain(text, model):
    """
    Returns:
    {
        prediction,
        confidence,
        top_triggers,
        explanation
    }
    """

    text = str(text)

    probs = model.predict_proba([text])[0]
    classes = model.classes_

    idx = np.argmax(probs)

    prediction = classes[idx]
    confidence = round(float(probs[idx] * 100), 2)

    top_features = _extract_top_features(text, model)

    triggers = [x["feature"] for x in top_features]

    if prediction == "Scam":
        explanation = (
            "Message contains linguistic patterns commonly "
            "associated with phishing, KYC fraud, OTP requests, "
            "account verification scams, suspicious URLs, or coercive pressure."
        )
    else:
        explanation = (
            "Message resembles normal banking notifications, "
            "personal communication, or trusted transactional alerts."
        )

    return {
        "prediction": prediction,
        "confidence": confidence,
        "top_triggers": top_features,
        "explanation": explanation,
    }


# ==========================================================
# AUDIO ANALYSIS (OFFLINE)
# ==========================================================

_whisper_model = None


def _get_whisper():
    global _whisper_model

    if _whisper_model is None:
        _whisper_model = WhisperModel(
            "tiny",
            device="cpu",
            compute_type="int8",
        )

    return _whisper_model


def analyze_call_audio(audio_file_path, model):
    """
    Offline scam-call detector
    """

    whisper = _get_whisper()

    segments, _ = whisper.transcribe(
        audio_file_path,
        beam_size=1,
    )

    transcript = " ".join(
        segment.text for segment in segments
    ).strip()

    result = predict_and_explain(
        transcript,
        model,
    )

    detected_phrases = []

    transcript_lower = transcript.lower()

    for category, phrase_list in SCAM_PHRASES.items():
        for phrase in phrase_list:
            if phrase.lower() in transcript_lower:
                detected_phrases.append(
                    {
                        "category": category,
                        "phrase": phrase,
                    }
                )

    return {
        "transcript": transcript,
        "verdict": result["prediction"],
        "confidence": result["confidence"],
        "explanation": result["explanation"],
        "trigger_words": result["top_triggers"],
        "triggered_phrases": detected_phrases,
    }


# ==========================================================
# QR / UPI ANALYZER
# ==========================================================

def check_upi_qr(image_bytes, expected_name=""):
    """
    Decode UPI QR offline.

    Returns:
    {
        success,
        upi_id,
        payee_name,
        amount,
        suspicious,
        reasons
    }
    """

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8,
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR,
    )

    detector = cv2.QRCodeDetector()

    data, _, _ = detector.detectAndDecode(image)

    if not data:
        return {
            "success": False,
            "reason": "No QR detected"
        }

    parsed = urllib.parse.urlparse(data)

    if parsed.scheme.lower() != "upi":
        return {
            "success": False,
            "reason": "Not a UPI QR"
        }

    query = urllib.parse.parse_qs(parsed.query)

    upi_id = query.get("pa", [""])[0]
    payee_name = query.get("pn", [""])[0]
    amount = query.get("am", [""])[0]

    suspicious = False
    reasons = []

    if expected_name:
        if expected_name.lower().strip() not in payee_name.lower():
            suspicious = True
            reasons.append(
                "Payee name mismatch"
            )

    for keyword in SUSPICIOUS_NAMES:
        if keyword in payee_name.lower():
            suspicious = True
            reasons.append(
                f"Suspicious payee naming pattern: {keyword}"
            )

    return {
        "success": True,
        "raw_data": data,
        "upi_id": upi_id,
        "payee_name": payee_name,
        "amount": amount,
        "suspicious": suspicious,
        "reasons": reasons,
    }


# ==========================================================
# DATABASE
# ==========================================================

def init_db():
    """
    Create database and preload
    120 anonymized reports.
    """

    os.makedirs("data", exist_ok=True)

    conn = sqlite3.connect(DB_PATH)

    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS reports(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            district TEXT,
            scam_type TEXT,
            text TEXT
        )
    """)

    conn.commit()

    count = cur.execute(
        "SELECT COUNT(*) FROM reports"
    ).fetchone()[0]

    if count >= 120:
        conn.close()
        return

    templates = [
        "Caller asked for OTP to unblock account.",
        "Received suspicious KYC verification message.",
        "Fake parcel customs payment request.",
        "Loan app threatening contacts.",
        "Digital arrest video call demand.",
        "QR scan payment fraud attempt.",
        "Unknown caller requested bank details.",
        "Reward points redemption scam.",
        "Account freeze warning from fake bank.",
        "Suspicious UPI collect request received.",
    ]

    rows = []

    for _ in range(120):
        rows.append(
            (
                datetime.utcnow().isoformat(),
                random.choice(CHHATTISGARH_DISTRICTS),
                random.choice(SCAM_TYPES),
                scrub_pii(random.choice(templates)),
            )
        )

    cur.executemany(
        """
        INSERT INTO reports
        (
            timestamp,
            district,
            scam_type,
            text
        )
        VALUES (?, ?, ?, ?)
        """,
        rows,
    )

    conn.commit()
    conn.close()


def add_report(
    district,
    scam_type,
    text
):
    """
    Save new report
    """

    init_db()

    conn = sqlite3.connect(DB_PATH)

    cleaned = scrub_pii(text)

    conn.execute(
        """
        INSERT INTO reports
        (
            timestamp,
            district,
            scam_type,
            text
        )
        VALUES (?, ?, ?, ?)
        """,
        (
            datetime.utcnow().isoformat(),
            district,
            scam_type,
            cleaned,
        ),
    )

    conn.commit()
    conn.close()


def get_reports_df():

    init_db()

    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql_query(
        """
        SELECT *
        FROM reports
        ORDER BY timestamp DESC
        """,
        conn,
    )

    conn.close()

    return df


# ==========================================================
# EMERGING SCAM DETECTION
# ==========================================================

def detect_emerging_scams(
    df,
    similarity_threshold=0.70,
):
    """
    Detect clusters of repeated scams
    in same district.
    """

    alerts = []

    if len(df) < 3:
        return alerts

    for district in df["district"].unique():

        group = df[
            df["district"] == district
        ].copy()

        if len(group) < 3:
            continue

        texts = (
            group["text"]
            .fillna("")
            .astype(str)
            .tolist()
        )

        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
        )

        matrix = vectorizer.fit_transform(texts)

        sim = cosine_similarity(matrix)

        visited = set()

        for i in range(len(texts)):

            if i in visited:
                continue

            cluster = [i]

            for j in range(
                i + 1,
                len(texts)
            ):
                if sim[i, j] >= similarity_threshold:
                    cluster.append(j)

            if len(cluster) >= 3:

                visited.update(cluster)

                sample_msg = texts[
                    cluster[0]
                ][:150]

                alerts.append(
                    {
                        "district": district,
                        "count": len(cluster),
                        "example_message": sample_msg,
                        "alert": (
                            f"Potential emerging scam "
                            f"campaign detected in "
                            f"{district}"
                        ),
                    }
                )

    return alerts


# ==========================================================
# CONVENIENCE LOADER
# ==========================================================

def load_model(
    model_path="models/scam_model.joblib"
):
    return joblib.load(model_path)