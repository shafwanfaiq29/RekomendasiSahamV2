import os
import re
import requests
import json
import numpy as np
from collections import Counter
from config import POSITIVE_WORDS, NEGATIVE_WORDS, MARKET_POSITIVE_WORDS, MARKET_NEGATIVE_WORDS, STOPWORDS_ID, POS_MINING, NEG_MINING, NEUTRAL_AMBIGUOUS

# ==============================================================================
# KONFIGURASI API — diambil dari environment variable, TIDAK hardcoded
# ==============================================================================
MODEL_API_URL = os.getenv("MODEL_API_URL", "")
HF_API_KEY    = os.getenv("HF_API_KEY", "")

# Nama model IndoBERT di HuggingFace Inference API
_INDOBERT_MODEL = "mdhugol/indonesia-bert-sentiment-classification"


def _call_indobert_api(text: str) -> dict:
    """
    Panggil IndoBERT via HuggingFace Inference API (requests.post).
    Kembalikan dict {'positive': float, 'neutral': float, 'negative': float}.

    Jika API tidak tersedia atau gagal → kembalikan dict netral {0, 1, 0}
    sebagai fallback agar app tidak crash.
    """
    if not HF_API_KEY:
        # Tidak ada API key → langsung pakai lexicon-only fallback
        return {"positive": 0.0, "neutral": 1.0, "negative": 0.0}

    # Gunakan MODEL_API_URL jika ada (custom endpoint), atau endpoint standar HF
    api_url = MODEL_API_URL if MODEL_API_URL else \
        f"https://api-inference.huggingface.co/models/{_INDOBERT_MODEL}"

    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = json.dumps({"inputs": text, "parameters": {"top_k": None}})

    try:
        response = requests.post(api_url, headers=headers, data=payload, timeout=15)

        if response.status_code == 200:
            raw = response.json()
            # HF Inference API mengembalikan [[{label, score}, ...]] atau [{label, score}, ...]
            if isinstance(raw, list) and len(raw) > 0:
                items = raw[0] if isinstance(raw[0], list) else raw
            else:
                return {"positive": 0.0, "neutral": 1.0, "negative": 0.0}

            probs = {}
            for x in items:
                lbl   = str(x.get('label', '')).lower().strip()
                score = float(x.get('score', 0.0))
                if lbl in ('positive', 'pos', 'label_0'):
                    probs['positive'] = score
                elif lbl in ('neutral', 'neu', 'label_1'):
                    probs['neutral'] = score
                elif lbl in ('negative', 'neg', 'label_2'):
                    probs['negative'] = score

            return {
                'positive': probs.get('positive', 0.0),
                'neutral':  probs.get('neutral',  1.0),
                'negative': probs.get('negative', 0.0),
            }

        elif response.status_code == 503:
            # Model sedang loading di HF (cold start) — tunggu dan coba sekali lagi
            import time
            time.sleep(5)
            response2 = requests.post(api_url, headers=headers, data=payload, timeout=20)
            if response2.status_code == 200:
                return _call_indobert_api(text)  # rekursi satu kali
            print(f"[WARN] IndoBERT API 503 setelah retry: {response2.text[:100]}")
            return {"positive": 0.0, "neutral": 1.0, "negative": 0.0}

        else:
            print(f"[WARN] IndoBERT API error {response.status_code}: {response.text[:150]}")
            return {"positive": 0.0, "neutral": 1.0, "negative": 0.0}

    except requests.exceptions.Timeout:
        print(f"[WARN] IndoBERT API timeout. Teks: '{text[:60]}...'")
        return {"positive": 0.0, "neutral": 1.0, "negative": 0.0}
    except Exception as e:
        print(f"[WARN] IndoBERT API gagal: {e}")
        return {"positive": 0.0, "neutral": 1.0, "negative": 0.0}


def bersihkan_teks(teks):
    """
    Preprocessing persis 100% kayak di pemodelan_nlp.ipynb.

    PENTING:
    - Lowercase dulu SEBELUM regex lainnya (IndoBERT sensitif kapital)
    - Pakai [^\w\s] bukan [^a-z\s] → angka tetap dipertahankan
      (angka penting buat konteks saham, e.g. "naik 20%")
    - Urutan operasi harus identik dengan notebook
    """
    teks = str(teks).lower()
    teks = re.sub(r'http\S+|www\S+|https\S+', '', teks, flags=re.MULTILINE)
    teks = re.sub(r'\@\w+|\#\w+', '', teks)
    teks = re.sub(r'&[a-z]+;', ' ', teks)
    teks = re.sub(r'[^\w\s]', ' ', teks)
    teks = re.sub(r'\s+', ' ', teks).strip()
    return teks


def _apply_lexicon_adjustment(prob_pos, prob_neu, prob_neg, teks_lower,
                               positive_words, negative_words, neutral_words=None):
    """
    Terapkan penyesuaian lexicon pada probabilitas IndoBERT.
    Logika identik dengan versi asli notebook — hanya dipisah jadi helper
    agar tidak duplikasi antara apply_sentiment dan apply_market_sentiment.
    """
    ada_positif = any(kata in teks_lower for kata in positive_words)
    ada_negatif = any(kata in teks_lower for kata in negative_words)
    ada_netral  = (
        any(kata in teks_lower for kata in neutral_words)
        if neutral_words else False
    )

    if ada_negatif and not ada_positif:
        prob_neg += (prob_neu * 0.6 + prob_pos * 0.8)
        prob_neu *= 0.4
        prob_pos *= 0.2
    elif ada_positif and not ada_negatif:
        prob_pos += (prob_neu * 0.6 + prob_neg * 0.8)
        prob_neu *= 0.4
        prob_neg *= 0.2
    elif ada_netral and neutral_words:
        prob_neu += (prob_pos * 0.4 + prob_neg * 0.4)
        prob_pos *= 0.6
        prob_neg *= 0.6
    elif ada_positif and ada_negatif:
        prob_neu += (prob_pos * 0.5 + prob_neg * 0.5)
        prob_pos *= 0.5
        prob_neg *= 0.5

    total_prob = prob_pos + prob_neu + prob_neg
    if total_prob > 0:
        prob_pos /= total_prob
        prob_neu /= total_prob
        prob_neg /= total_prob

    return prob_pos, prob_neu, prob_neg


def apply_sentiment(news_df):
    """
    Eksekusi NLP via HuggingFace API + Hybrid Lexicon + Agregasi Normalisasi Absolut
    buat Saham Individual.
    """
    if news_df.empty:
        return news_df, 0.0, "Neutral", 0

    list_prob_pos, list_prob_neu, list_prob_neg = [], [], []

    for judul in news_df["Title"]:
        teks_bersih = bersihkan_teks(judul)
        if not teks_bersih:
            list_prob_pos.append(0.0)
            list_prob_neu.append(1.0)
            list_prob_neg.append(0.0)
            continue

        probs    = _call_indobert_api(teks_bersih)
        prob_pos = probs['positive']
        prob_neu = probs['neutral']
        prob_neg = probs['negative']

        teks_lower = str(judul).lower()
        prob_pos, prob_neu, prob_neg = _apply_lexicon_adjustment(
            prob_pos, prob_neu, prob_neg,
            teks_lower,
            positive_words=POS_MINING,
            negative_words=NEG_MINING,
            neutral_words=NEUTRAL_AMBIGUOUS,
        )

        list_prob_pos.append(prob_pos)
        list_prob_neu.append(prob_neu)
        list_prob_neg.append(prob_neg)

    news_df = news_df.copy()
    news_df['Prob_Positif'] = list_prob_pos
    news_df['Prob_Netral']  = list_prob_neu
    news_df['Prob_Negatif'] = list_prob_neg

    tanggal_col = "Tanggal" if "Tanggal" in news_df.columns else "Date"
    df_harian = news_df.groupby(tanggal_col).agg(
        Total_Berita=('Title', 'count'),
        Avg_Prob_Positif=('Prob_Positif', 'mean'),
        Avg_Prob_Netral=('Prob_Netral', 'mean'),
        Avg_Prob_Negatif=('Prob_Negatif', 'mean'),
    ).reset_index()

    total_prob_harian = df_harian['Avg_Prob_Positif'] + df_harian['Avg_Prob_Netral'] + df_harian['Avg_Prob_Negatif']
    df_harian['Avg_Prob_Positif'] = np.where(total_prob_harian > 0, df_harian['Avg_Prob_Positif'] / total_prob_harian, 0)
    df_harian['Avg_Prob_Netral']  = np.where(total_prob_harian > 0, df_harian['Avg_Prob_Netral']  / total_prob_harian, 0)
    df_harian['Avg_Prob_Negatif'] = np.where(total_prob_harian > 0, df_harian['Avg_Prob_Negatif'] / total_prob_harian, 0)

    df_harian['Skor_Sentimen_Final'] = df_harian['Avg_Prob_Positif'] - df_harian['Avg_Prob_Negatif']
    df_harian = df_harian.sort_values(tanggal_col, ascending=False).reset_index(drop=True)

    skor_final = float(df_harian.iloc[0]['Skor_Sentimen_Final'])

    if skor_final > 0.02:
        label_final = "Positive"
    elif skor_final < -0.02:
        label_final = "Negative"
    else:
        label_final = "Neutral"

    news_df["Sentiment_Score"] = news_df['Prob_Positif'] - news_df['Prob_Negatif']
    news_df["Sentiment_Label"] = np.where(news_df["Sentiment_Score"] > 0.02, "Positive",
                                 np.where(news_df["Sentiment_Score"] < -0.02, "Negative", "Neutral"))

    return news_df, skor_final, label_final, len(news_df)


def apply_market_sentiment(news_df):
    """
    Eksekusi NLP via HuggingFace API + Hybrid Lexicon + Agregasi Normalisasi Absolut
    buat Market News Global.
    """
    if news_df.empty:
        return news_df, 0.0, "Neutral Market"

    list_prob_pos, list_prob_neu, list_prob_neg = [], [], []

    for judul in news_df["Title"]:
        teks_bersih = bersihkan_teks(judul)
        if not teks_bersih:
            list_prob_pos.append(0.0)
            list_prob_neu.append(1.0)
            list_prob_neg.append(0.0)
            continue

        probs    = _call_indobert_api(teks_bersih)
        prob_pos = probs['positive']
        prob_neu = probs['neutral']
        prob_neg = probs['negative']

        teks_lower = str(judul).lower()
        prob_pos, prob_neu, prob_neg = _apply_lexicon_adjustment(
            prob_pos, prob_neu, prob_neg,
            teks_lower,
            positive_words=MARKET_POSITIVE_WORDS,
            negative_words=MARKET_NEGATIVE_WORDS,
            neutral_words=None,  # Market sentiment tidak pakai NEUTRAL_AMBIGUOUS
        )

        list_prob_pos.append(prob_pos)
        list_prob_neu.append(prob_neu)
        list_prob_neg.append(prob_neg)

    news_df = news_df.copy()
    news_df['Prob_Positif'] = list_prob_pos
    news_df['Prob_Netral']  = list_prob_neu
    news_df['Prob_Negatif'] = list_prob_neg

    avg_prob_pos = news_df['Prob_Positif'].mean()
    avg_prob_neu = news_df['Prob_Netral'].mean()
    avg_prob_neg = news_df['Prob_Negatif'].mean()

    total_prob_global = avg_prob_pos + avg_prob_neu + avg_prob_neg
    if total_prob_global > 0:
        avg_prob_pos /= total_prob_global
        avg_prob_neu /= total_prob_global
        avg_prob_neg /= total_prob_global

    skor_final = avg_prob_pos - avg_prob_neg

    if skor_final > 0.02:
        label_final = "Positive Market"
    elif skor_final < -0.02:
        label_final = "Negative Market"
    else:
        label_final = "Neutral Market"

    news_df["Sentiment_Score"] = news_df['Prob_Positif'] - news_df['Prob_Negatif']
    news_df["Sentiment_Label"] = np.where(news_df["Sentiment_Score"] > 0.02, "Positive",
                                 np.where(news_df["Sentiment_Score"] < -0.02, "Negative", "Neutral"))

    return news_df, float(skor_final), label_final


def get_trending_topics(news_df):
    if news_df.empty:
        return []

    all_titles = " ".join(news_df["Title"].tolist()).lower()
    all_titles = re.sub(r'[^\w\s]', '', all_titles)
    words = all_titles.split()

    filtered_words = [w for w in words if w not in STOPWORDS_ID and len(w) > 2]
    word_counts = Counter(filtered_words)
    return [word for word, count in word_counts.most_common(10)]
