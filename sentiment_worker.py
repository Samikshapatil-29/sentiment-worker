# sentiment_worker.py

import os
import io
import time
import json
from datetime import datetime

from supabase import create_client, Client


# ==========================================
# CONFIG - USE VADER FOR RAILWAY (LIGHTWEIGHT)
# ==========================================
USE_HF = False  # Set True only if Railway has enough memory


# ==========================================
# TRANSFORMER MODELS (IF ENABLED)
# ==========================================
if USE_HF:
    try:
        from transformers import pipeline
        hf_sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        hf_summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    except Exception as e:
        print("HF unavailable, switching to VADER:", e)
        USE_HF = False


# ==========================================
# VADER + GENSIM (LIGHTWEIGHT FALLBACK)
# ==========================================
if not USE_HF:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from gensim.summarization import summarize as gensim_summarize

    nltk.download("vader_lexicon", quiet=True)
    vader = SentimentIntensityAnalyzer()


# ==========================================
# WORDCLOUD
# ==========================================
from wordcloud import WordCloud
from PIL import Image


# ==========================================
# SUPABASE CONNECTION
# ==========================================
SUPABASE_URL = os.environ.get("https://bnpmufjeoyitvafhilde.supabase.co")
SUPABASE_KEY = os.environ.get("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJucG11Zmplb3lpdHZhZmhpbGRlIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM3ODcyNjgsImV4cCI6MjA3OTM2MzI2OH0.BWe0BxLdu25h-sgoTyrNXhGmnj-FUZ_neq6PMlMKxIw")
WORDCLOUD_BUCKET = os.environ.get("WORDCLOUD_BUCKET", "wordclouds")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# ==========================================
# FETCH UNPROCESSED ROWS
# ==========================================
def fetch_unprocessed(limit=100):
    try:
        resp = supabase.table("uploaded_csv_data") \
                       .select("*") \
                       .eq("processed", False) \
                       .limit(limit) \
                       .execute()
        return resp.data or []
    except Exception as e:
        print("Fetch error:", e)
        return []


# ==========================================
# ANALYZE TEXT
# ==========================================
def analyze_text_vader(text):
    score = vader.polarity_scores(text)
    comp = score["compound"]

    if comp >= 0.05:
        label = "positive"
    elif comp <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    try:
        summary = gensim_summarize(text, word_count=30)
        if not summary:
            summary = text.split(".")[0]
    except:
        summary = text.split(".")[0]

    return {"label": label, "score": comp, "summary": summary}


def analyze_text_hf(text):
    s = hf_sentiment(text[:512])
    label = s[0]["label"].lower()
    score = float(s[0]["score"])

    try:
        summary_out = hf_summarizer(text, max_length=60, min_length=20, do_sample=False)
        summary = summary_out[0]["summary_text"]
    except:
        summary = text.split(".")[0]

    return {"label": label, "score": score, "summary": summary}


# ==========================================
# WORDCLOUD GENERATION
# ==========================================
def generate_wordcloud(text):
    wc = WordCloud(width=800, height=400, background_color="white")
    img = wc.generate(text)
    buffer = io.BytesIO()
    img.to_image().save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


# ==========================================
# UPLOAD TO SUPABASE STORAGE
# ==========================================
def upload_wordcloud(buf, filename):
    try:
        bucket = supabase.storage().from_(WORDCLOUD_BUCKET)
        file_bytes = buf.read()

        bucket.upload(filename, file_bytes)

        signed_url = bucket.create_signed_url(filename, 60 * 60 * 24 * 365 * 5)
        return signed_url.get("signedURL")
    except Exception as e:
        print("Upload failed:", e)
        return None


# ==========================================
# SAVE ANALYSIS TO DB
# ==========================================
def save_analysis(row, analysis, wordcloud_url):
    payload = {
        "source_id": row["id"],
        "file_name": row["file_name"],
        "row_number": row["row_number"],
        "original_text": row["text"],
        "sentiment_label": analysis["label"],
        "sentiment_score": analysis["score"],
        "summary": analysis["summary"],
        "wordcloud_url": wordcloud_url,
        "model_info": json.dumps({
            "method": "hf" if USE_HF else "vader",
            "timestamp": datetime.utcnow().isoformat()
        })
    }

    supabase.table("analyzed_results").insert(payload).execute()
    supabase.table("uploaded_csv_data").update({"processed": True}).eq("id", row["id"]).execute()


# ==========================================
# BATCH PROCESS
# ==========================================
def process_batch():
    rows = fetch_unprocessed()

    if not rows:
        print("No unprocessed rows.")
        return

    for row in rows:

        print("Analyzing id:", row.get("id"))

        text = (row.get("text") or "").strip()
        if not text:
            print("Empty text, skipping.")
            supabase.table("uploaded_csv_data").update({"processed": True}).eq("id", row["id"]).execute()
            continue

        analysis = analyze_text_hf(text) if USE_HF else analyze_text_vader(text)

        wc_buf = generate_wordcloud(text)
        filename = f"wc_{row['id']}.png"
        wc_url = upload_wordcloud(wc_buf, filename)

        save_analysis(row, analysis, wc_url)

        print(f"Processed ID {row['id']} → {analysis['label']} ({analysis['score']})")


# ==========================================
# MAIN LOOP
# ==========================================
if __name__ == "__main__":
    while True:
        process_batch()
        print("Sleeping for 30 sec...")
        time.sleep(30)
