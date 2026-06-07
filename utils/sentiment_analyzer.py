import re
import numpy as np
from collections import Counter
from transformers import pipeline
from config import POSITIVE_WORDS, NEGATIVE_WORDS, MARKET_POSITIVE_WORDS, MARKET_NEGATIVE_WORDS, STOPWORDS_ID

def load_indobert():
    print("[*] Memuat Model IndoBERT ke RAM...")
    return pipeline(
        "text-classification",
        model="mdhugol/indonesia-bert-sentiment-classification",
        top_k=None,
        truncation=True,
        max_length=128,
    )

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

def apply_sentiment(news_df):
    """
    Eksekusi NLP + Hybrid Lexicon + Agregasi Normalisasi Absolut buat Saham Individual
    """
    if news_df.empty:
        return news_df, 0.0, "Neutral", 0

    indobert = load_indobert()
    list_prob_pos, list_prob_neu, list_prob_neg = [], [], []
    
    for judul in news_df["Title"]:
        teks_bersih = bersihkan_teks(judul)
        if not teks_bersih:
            list_prob_pos.append(0.0)
            list_prob_neu.append(1.0)
            list_prob_neg.append(0.0)
            continue
            
        hasil = indobert(teks_bersih)[0]
        probs = {}
        for x in hasil:
            lbl = str(x['label']).lower().strip()
            score = float(x['score'])
            if lbl in ('positive', 'pos', 'label_0'):
                probs['positive'] = score
            elif lbl in ('neutral', 'neu', 'label_1'):
                probs['neutral'] = score
            elif lbl in ('negative', 'neg', 'label_2'):
                probs['negative'] = score
                
        prob_pos = probs.get('positive', 0.0)
        prob_neu = probs.get('neutral', 0.0)
        prob_neg = probs.get('negative', 0.0)
        
        teks_lower = str(judul).lower()
        ada_positif = any(kata in teks_lower for kata in POSITIVE_WORDS)
        ada_negatif = any(kata in teks_lower for kata in NEGATIVE_WORDS)
        
        if ada_negatif and not ada_positif:
            prob_neg += (prob_neu * 0.7)
            prob_neu *= 0.3
        elif ada_positif and not ada_negatif:
            prob_pos += (prob_neu * 0.7)
            prob_neu *= 0.3
            
        total_prob = prob_pos + prob_neu + prob_neg
        prob_pos /= total_prob
        prob_neu /= total_prob
        prob_neg /= total_prob
                
        list_prob_pos.append(prob_pos)
        list_prob_neu.append(prob_neu)
        list_prob_neg.append(prob_neg)
        
    news_df = news_df.copy()
    news_df['Prob_Positif'] = list_prob_pos
    news_df['Prob_Netral'] = list_prob_neu
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
    df_harian['Avg_Prob_Netral'] = np.where(total_prob_harian > 0, df_harian['Avg_Prob_Netral'] / total_prob_harian, 0)
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
    Eksekusi NLP + Hybrid Lexicon + Agregasi Normalisasi Absolut buat Market News Global
    """
    if news_df.empty:
        return news_df, 0.0, "Neutral Market"

    indobert = load_indobert()
    list_prob_pos, list_prob_neu, list_prob_neg = [], [], []
    
    for judul in news_df["Title"]:
        teks_bersih = bersihkan_teks(judul)
        if not teks_bersih:
            list_prob_pos.append(0.0)
            list_prob_neu.append(1.0)
            list_prob_neg.append(0.0)
            continue
            
        hasil = indobert(teks_bersih)[0]
        probs = {}
        for x in hasil:
            lbl = str(x['label']).lower().strip()
            score = float(x['score'])
            if lbl in ('positive', 'pos', 'label_0'):
                probs['positive'] = score
            elif lbl in ('neutral', 'neu', 'label_1'):
                probs['neutral'] = score
            elif lbl in ('negative', 'neg', 'label_2'):
                probs['negative'] = score
                
        prob_pos = probs.get('positive', 0.0)
        prob_neu = probs.get('neutral', 0.0)
        prob_neg = probs.get('negative', 0.0)
        
        teks_lower = str(judul).lower()
        ada_positif = any(kata in teks_lower for kata in MARKET_POSITIVE_WORDS)
        ada_negatif = any(kata in teks_lower for kata in MARKET_NEGATIVE_WORDS)
        
        if ada_negatif and not ada_positif:
            prob_neg += (prob_neu * 0.7)
            prob_neu *= 0.3
        elif ada_positif and not ada_negatif:
            prob_pos += (prob_neu * 0.7)
            prob_neu *= 0.3
            
        total_prob = prob_pos + prob_neu + prob_neg
        prob_pos /= total_prob
        prob_neu /= total_prob
        prob_neg /= total_prob
                
        list_prob_pos.append(prob_pos)
        list_prob_neu.append(prob_neu)
        list_prob_neg.append(prob_neg)
        
    news_df = news_df.copy()
    news_df['Prob_Positif'] = list_prob_pos
    news_df['Prob_Netral'] = list_prob_neu
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
