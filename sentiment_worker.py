# sentiment_worker.py

import os
import io
import time
import json
from datetime import datetime

from supabase import create_client, Client

# -------- Lightweight sentiment setup (Railway-safe) --------
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download("vader_lexicon", quiet=True)
vader = SentimentIntensityAnalyzer()

# -------- Wordcloud --------
from wordcloud import WordCloud
from PIL import Image


# -------- Supabase Setup --------
SUPABASE_URL = os.environ.get("https://bnpmufjeoyitvafhilde.supabase.co")
SUPABASE_KEY = os.environ.get("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJucG11Zmplb3lpdHZhZmhpbGRlIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjM3ODcyNjgsImV4cCI6MjA3OTM2MzI2OH0.BWe0BxLdu25h-sgoTyrNXhGmnj-FUZ_neq6PMlMKxIw")
WORDCLOUD_BUCKET = os.environ.get("WORDCLOUD_BUCKET", "wordclouds")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# -------- Simple Summarizer (no gensim) --------
def simple_summary(text):
    sentences = text.split(".")
    if len(sentences) > 1:
        return sentences[0].strip()
    return text[:120]  # fallback


# -------- Fetch Rows --------
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


# -------- Analyze Sentiment --------
def analyze_text(text):
    score = vader.polarity_scores(text)
    comp = score["compound"]

    if comp >= 0.05:
        label = "positive"
    elif comp <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    summary = simple_summary(text)

    return {"label": label, "score": comp, "summary": summary}


# -------- Wordcloud --------
def generate_wordcloud(text):
    wc = WordCloud(width=800, height=400, background_color="white")
    img = wc.generate(text)
    buffer = io.BytesIO()
    img.to_image().save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


# -------- Upload to Supabase Storage --------
def upload_wordcloud(buf, filename):
    try:
        bucket = supabase.storage().from_(WORDCLOUD_BUCKET)
        file_bytes = buf.read()
        bucket.upload(filename, file_bytes)

        signed = bucket.create_signed_url(filename, 60 * 60 * 24 * 365 * 5)
        return signed.get("signedURL")
    except Exception as e:
        print("Upload failed:", e)
        return None


# -------- Save Analysis --------
def save_analysis(row, analysis, wc_url):
    payload = {
        "source_id": row["id"],
        "file_name": row["file_name"],
        "row_number": row["row_number"],
        "original_text": row["text"],
        "sentiment_label": analysis["label"],
        "sentiment_score": analysis["score"],
        "summary": analysis["summary"],
        "wordcloud_url": wc_url,
        "model_info": json.dumps({
            "method": "vader",
            "timestamp": datetime.utcnow().isoformat()
        })
    }

    supabase.table("analyzed_results").insert(payload).execute()
    supabase.table("uploaded_csv_data").update({"processed": True}).eq("id", row["id"]).execute()


# -------- Worker Loop --------
def process_batch():
    rows = fetch_unprocessed()
    if not rows:
        print("No unprocessed rows.")
        return

    for row in rows:
        print("Analyzing ID:", row["id"])
        text = row.get("text", "").strip()

        analysis = analyze_text(text)
        wc_buf = generate_wordcloud(text)

        filename = f"wc_{row['id']}.png"
        wc_url = upload_wordcloud(wc_buf, filename)

        save_analysis(row, analysis, wc_url)
        print(f"Processed {row['id']} → {analysis['label']} ({analysis['score']})")


# -------- Main Loop --------
if __name__ == "__main__":
    while True:
        process_batch()
        print("Sleeping 20 sec...")
        time.sleep(20)
