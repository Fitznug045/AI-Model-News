import re
import pickle
import requests
import pandas as pd
import tensorflow as tf
import json

from bs4 import BeautifulSoup
from urllib.parse import quote
from flask import Flask, request, jsonify

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing.sequence import pad_sequences

model_clickbait = tf.keras.models.load_model("model_clickbait_final.keras")
model_stance = tf.keras.models.load_model("model_stance_functional.keras")
pers = pd.read_csv("final-pers.csv")

with open("tokenizer_clickbait.pkl", "rb") as f:
    tokenizer_clickbait = pickle.load(f)

with open("claim_tokenizer.pkl", "rb") as f:
    claim_tokenizer = pickle.load(f)

with open("article_tokenizer.pkl", "rb") as f:
    article_tokenizer = pickle.load(f)

with open("stance_config.pkl", "rb") as f:
    stance_config = pickle.load(f)

with open("negatif-indonesia.txt", "r", encoding="utf-8") as f:
    negative_words = set(line.strip().lower() for line in f if line.strip())

with open("positif-indonesia.txt", "r", encoding="utf-8") as f:
    positive_words = set(line.strip().lower() for line in f if line.strip())

print("Semua model, tokenizer, dan file pendukung berhasil diload.")

strong_bantah_keywords = [
    "hoaks",
    "hoax",
    "[hoaks]",
    "cek fakta",
    "dibantah",
    "klarifikasi",
    "tidak benar",
    "bukan fakta",
    "fakta sebenarnya",
    "fact check",
    "[hoaks]",
    "[salah]",
    "penipuan",
    "[penipuan]",
    "keliru"
]

weak_bantah_keywords = [
    "palsu",
    "bohong",
    "fitnah",
    "misinformasi",
    "disinformasi",
    "fake",
    "menyesatkan",
    "tidak terbukti",
    "tidak jadi",
    "tidak akan",
    "diragukan",
    "belum terbukti"
]

strong_dukung_keywords = [
    "dikonfirmasi",
    "terbukti",
    "dipastikan",
    "menegaskan",
    "didakwa",
    "dituntut",
    "putusan",
    "vonis",
    "terdakwa",
    "umumkan",
    "mengumumkan",
    "resmi",
    "sidang"
]

weak_dukung_keywords = [
    "benar",
    "valid",
    "official",
    "nyata",
    "jadi",
    "mulai berlaku",
    "berlangsung",
    "terjadi",
    "mengakui",
    "menyatakan",
    "mengonfirmasi",
    "pengadilan",
    "kejaksaan",
    "jaksa tuntut",
]

MAX_LEN_CLICKBAIT = 40

MAX_CLAIM_LEN = stance_config["MAX_CLAIM_LEN"]
MAX_ARTICLE_LEN = stance_config["MAX_ARTICLE_LEN"]

THRESHOLD_CLICKBAIT = 0.45

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s!?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def search_google_news(headline, max_articles=10):
    query_encoded = quote(headline)

    rss_url = (
        "https://news.google.com/rss/search?"
        f"q={query_encoded}&hl=id&gl=ID&ceid=ID:id"
    )

    response = requests.get(
        rss_url,
        timeout=10,
        headers={
            "User-Agent": "Mozilla/5.0"
        }
    )

    soup = BeautifulSoup(response.content, "xml")

    items = soup.find_all("item")[:max_articles]

    articles = []

    for item in items:
        title = item.title.text if item.title else ""
        link = item.link.text if item.link else ""
        source = item.source.text if item.source else ""
        published = item.pubDate.text if item.pubDate else ""

        description = item.description.text if item.description else ""
        snippet = BeautifulSoup(description, "html.parser").get_text(" ", strip=True)

        content = snippet if snippet else title

        articles.append({
            "title": title,
            "url": link,
            "source": source,
            "published": published,
            "content": content
        })

    return pd.DataFrame(articles)

def predict_clickbait(headline):
    clean = clean_text(headline)

    seq = tokenizer_clickbait.texts_to_sequences([clean])

    pad = pad_sequences(
        seq,
        maxlen=MAX_LEN_CLICKBAIT,
        padding="post",
        truncating="post"
    )

    score = model_clickbait.predict(pad, verbose=0)[0][0]

    return float(score)

# hitung stance dari model + kamus
def sentiment_score(text):
    words = clean_text(text).split()

    pos = sum(1 for w in words if w in positive_words)
    neg = sum(1 for w in words if w in negative_words)

    total = pos + neg

    if total == 0:
        return 0

    return (pos - neg) / total

def predict_stance(claim, article_text):

    clean_claim = clean_text(claim)
    clean_article = clean_text(article_text)

    claim_seq = claim_tokenizer.texts_to_sequences([clean_claim])
    article_seq = article_tokenizer.texts_to_sequences([clean_article])

    claim_pad = pad_sequences(
        claim_seq,
        maxlen=MAX_CLAIM_LEN,
        padding="post",
        truncating="post"
    )

    article_pad = pad_sequences(
        article_seq,
        maxlen=MAX_ARTICLE_LEN,
        padding="post",
        truncating="post"
    )

    model_score = model_stance.predict(
        [claim_pad, article_pad],
        verbose=0
    )[0][0]

    text = clean_article

    strong_bantah_hit = sum(
        k in text for k in strong_bantah_keywords
    )

    weak_bantah_hit = sum(
        k in text for k in weak_bantah_keywords
    )

    strong_dukung_hit = sum(
        k in text for k in strong_dukung_keywords
    )

    weak_dukung_hit = sum(
        k in text for k in weak_dukung_keywords
    )


    sent_score = sentiment_score(text)

    boost = 0

    if strong_bantah_hit > 0:
        boost -= 0.45

    elif strong_dukung_hit > 0:
        boost += 0.35

    elif weak_bantah_hit > weak_dukung_hit:
        boost -= 0.15

    elif weak_dukung_hit > weak_bantah_hit:
        boost += 0.15

    # sentiment negatif
    if sent_score < -0.2:
        boost -= 0.10

    # sentiment positif
    elif sent_score > 0.2:
        boost += 0.10

    final_score = model_score + boost
    final_score = max(0, min(1, final_score))

    if strong_bantah_hit > 0:
        final_score = min(final_score, 0.25)

    elif strong_dukung_hit > 0:
        final_score = max(final_score, 0.75)

    elif weak_bantah_hit > weak_dukung_hit:
        final_score = min(final_score, 0.35)

    elif weak_dukung_hit > weak_bantah_hit:
        final_score = max(final_score, 0.55)

    if final_score < 0.4:
        label = "membantah"
    else:
        label = "mendukung"

    return label, float(final_score)

def get_similarity(headline, article_text):
    texts = [
        clean_text(headline),
        clean_text(article_text)
    ]

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(texts)

    sim = cosine_similarity(
        tfidf[0:1],
        tfidf[1:2]
    )[0][0]

    return float(sim)

def is_trusted_source(source):
    source = str(source).lower().strip()

    all_pers_text = " ".join(
        pers.astype(str)
        .apply(lambda x: " ".join(x), axis=1)
        .str.lower()
        .tolist()
    )

    return source in all_pers_text

def analyze_news_claim(headline, max_articles=10):
    clickbait_score = predict_clickbait(headline)

    if clickbait_score >= THRESHOLD_CLICKBAIT:
        return {
            "headline": headline,
            "is_clickbait": True,
            "clickbait_score": clickbait_score,
            "verdict": "CLICKBAIT"
        }
    df_articles = search_google_news(headline, max_articles=max_articles)

    results = []

    for _, row in df_articles.iterrows():
        title = row["title"]
        content = row["content"]
        source = row["source"]

        article_text = title + " " + content

        article_clickbait_score = predict_clickbait(title)

        article_clickbait_label = (
            "yes" if article_clickbait_score >= THRESHOLD_CLICKBAIT else "no"
        )


        similarity = get_similarity(headline, article_text)

        stance_label, stance_model_score = predict_stance(
            headline,
            article_text
        )

        trusted = is_trusted_source(source)

        stance_risk = 1.0 - stance_model_score

        trusted_bantah_score = 1.0 if trusted and stance_label == "membantah" else 0.0
        trusted_dukung_score = 1.0 if trusted and stance_label == "mendukung" else 0.0

        article_hoax_score = (
            0.55 * stance_risk +
            0.30 * (1 - similarity) +
            0.15 * trusted_bantah_score
        )

        article_hoax_score -= 0.20 * trusted_dukung_score

        article_hoax_score = max(0, min(1, article_hoax_score))

        results.append({
            "title": title,
            "source": source,
            "url": row["url"],
            "published": row["published"],
            "similarity": round(similarity, 4),
            "stance_model_score": round(stance_model_score, 4),
            "stance": stance_label,
            "article_clickbait_score": round(article_clickbait_score, 4),
            "article_clickbait": article_clickbait_label,
            "article_hoax_score": round(article_hoax_score, 4),
            "trusted_source": trusted
        })

    df_result = pd.DataFrame(results)

    if len(df_result) == 0:
        return {
            "headline": headline,
            "message": "Tidak ditemukan artikel pembanding."
        }

    relevant_df = df_result[
        (df_result["article_clickbait"] == "no") &
        (df_result["similarity"] >= 0.15)
    ]

    if len(relevant_df) == 0:
      return {
          "headline": headline,
          "message": "Tidak ditemukan artikel non-clickbait yang relevan.",
          "articles": df_result
      }

    avg_similarity = relevant_df["similarity"].mean()
    avg_stance_model_score = relevant_df["stance_model_score"].mean()
    avg_article_hoax_score = relevant_df["article_hoax_score"].mean()

    membantah_count = (relevant_df["stance"] == "membantah").sum()
    mendukung_count = (relevant_df["stance"] == "mendukung").sum()

    trusted_bantah_ratio = (
        (relevant_df["trusted_source"] == True) &
        (relevant_df["stance"] == "membantah")
    ).mean()

    trusted_dukung_ratio = (
        (relevant_df["trusted_source"] == True) &
        (relevant_df["stance"] == "mendukung")
    ).mean()

    final_score = avg_article_hoax_score
    final_score = max(0, min(1, final_score))

    if final_score >= 0.60:
        verdict = "Kemungkinan Hoax"
    elif final_score >= 0.40:
        verdict = "Perlu Verifikasi Lanjutan"
    else:
        verdict = "Kemungkinan Asli"

    output = {
        "headline": headline,
        "verdict": verdict,
        "final_hoax_score": round(final_score, 4),
        "avg_similarity": round(avg_similarity, 4),
        "avg_stance_model_score": round(avg_stance_model_score, 4),
        "avg_article_hoax_score": round(avg_article_hoax_score, 4),
        "trusted_bantah_ratio": round(float(trusted_bantah_ratio), 4),
        "trusted_dukung_ratio": round(float(trusted_dukung_ratio), 4),
        "membantah_count": int(membantah_count),
        "mendukung_count": int(mendukung_count),
        "articles": df_result
    }

    return output

# FORMAT RESPONSE UNTUK BACKEND
def format_response_for_backend(hasil):
    if hasil.get("is_clickbait"):
        return {
            "skor_clickbait": round(
                hasil["clickbait_score"] * 100,
                2
            ),
            "verdict": "CLICKBAIT",
            "keyakinan": "tinggi",
            "berita_terkait": []
        }
    df_articles = hasil["articles"].copy()

    df_articles = df_articles[
        (df_articles["article_clickbait"] == "no") &
        (df_articles["similarity"] >= 0.15)
    ]

    if "final_hoax_score" not in hasil:

        berita_terkait = []

        return {
            "skor_hoax": None,
            "verdict": "TIDAK DAPAT DIANALISIS",
            "keyakinan": "rendah",
            "message": hasil.get("message"),
            "berita_terkait": berita_terkait
        }

    berita_terkait = []

    for i, (_, row) in enumerate(df_articles.iterrows(), start=1):
        berita_terkait.append({
            "id": i,
            "judul": row["title"],
            "sumber": row["source"],
            "similarity": None if pd.isna(row["similarity"]) else round(row["similarity"] * 100, 2),
            "status": None if pd.isna(row["stance"]) else str(row["stance"]).capitalize(),
            "trusted_source": None if pd.isna(row["trusted_source"]) else bool(row["trusted_source"]),
            "skor_hoax_artikel": None if pd.isna(row["article_hoax_score"]) else round(row["article_hoax_score"] * 100, 2),
            "url": row["url"]
        })

    # hitungan skor hoax dijadiin rentang 0-100
    skor_hoax = 0

    total_konfirmasi = hasil["membantah_count"] + hasil["mendukung_count"]

    if total_konfirmasi > 0:
        rasio_bantah = hasil["membantah_count"] / total_konfirmasi
    else:
        rasio_bantah = 0

    skor_hoax += rasio_bantah * 45

    skor_hoax += hasil["avg_article_hoax_score"] * 30

    skor_hoax += (1 - hasil["avg_similarity"]) * 10

    skor_hoax += hasil["trusted_bantah_ratio"] * 15

    skor_hoax -= hasil["trusted_dukung_ratio"] * 10

    skor_hoax = max(0, min(100, skor_hoax))
    skor_hoax = round(skor_hoax)

    # verdict
    if skor_hoax >= 75:
        verdict = "SANGAT MUNGKIN HOAX"

    elif skor_hoax >= 55:
        verdict = "KEMUNGKINAN HOAX"

    elif skor_hoax >= 40:
        verdict = "TIDAK PASTI (PERLU VERIFIKASI)"

    elif skor_hoax >= 20:
        verdict = "KEMUNGKINAN BENAR"

    else:
        verdict = "SANGAT MUNGKIN BENAR"

    # Keyakinan
    if skor_hoax >= 75 or skor_hoax <= 19:
        keyakinan = "tinggi"
    elif skor_hoax >= 55 or skor_hoax <= 39:
        keyakinan = "sedang"
    else:
        keyakinan = "rendah"

    # Analisis semantik
    analisis_semantik = []

    # cek fakta
    if hasil["trusted_bantah_ratio"] > 0:
        cek_fakta_resmi = "Ditemukan sumber terpercaya yang membantah"
    elif hasil["trusted_dukung_ratio"] > 0:
        cek_fakta_resmi = "Ditemukan sumber terpercaya yang mendukung"
    else:
        cek_fakta_resmi = "Tidak ada cek fakta resmi ditemukan"

    response = {
        "skor_hoax": skor_hoax,
        "verdict": verdict,
        "keyakinan": keyakinan,
        "komponen_analisis": {
            "rasio_konfirmasi": {
                "membantah": hasil["membantah_count"],
                "mendukung": hasil["mendukung_count"]
            },
            "cek_fakta_resmi": cek_fakta_resmi,
            "analisis_semantik": analisis_semantik,
            "jangkauan_penelusuran": len(df_articles)
        },
        "berita_terkait": berita_terkait
    }

    return response


app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Hoax Detection API is running",
        "endpoint": "/predict",
        "method": "POST"
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if data is None:
            return jsonify({
                "error": "Request body harus JSON"
            }), 400

        headline = data.get("headline", "")

        if headline.strip() == "":
            return jsonify({
                "error": "headline kosong"
            }), 400

        max_articles = data.get("max_articles", 10)

        hasil = analyze_news_claim(
            headline=headline,
            max_articles=max_articles
        )

        if "articles" not in hasil:
            return jsonify({
                "skor_hoax": None,
                "verdict": "TIDAK DAPAT DIANALISIS",
                "keyakinan": "rendah",
                "message": hasil.get("message", "Tidak ditemukan artikel pembanding."),
                "berita_terkait": []
            }), 200

        response_backend = format_response_for_backend(hasil)

        return jsonify(response_backend), 200

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
