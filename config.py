TICKER_MAP = {
    "ANTM": "ANTM.JK",
    "MDKA": "MDKA.JK",
    "BRMS": "BRMS.JK",
    "PSAB": "PSAB.JK"
}

COMPANY_NAMES = {
    "ANTM": "PT Aneka Tambang Tbk",
    "MDKA": "PT Merdeka Copper Gold Tbk",
    "BRMS": "PT Bumi Resources Minerals Tbk",
    "PSAB": "PT J Resources Asia Pasifik Tbk"
}

SCALER_FEATURES = [
    "Open", "High", "Low", "Close", "Volume",
    "Gold_Close",
    "MA7", "MA30",
    "Return",
    "Volatility",
    "Gold_Return",
]

MODEL_FEATURES = [
    "Open", "High", "Low", "Close", "Volume",
    "Return", "MA7", "MA30", "Volatility",
    "Gold_Close", "Gold_Return",
    "Sentiment_Score", "News_Count",
    "PBV_x_ROE", "Price_to_Equity_Discount", "Relative_PE_ratio",
    "EPS_Growth", "Debt_to_Total_Assets_Ratio", "Liquidity_Differential",
    "CCE", "Operating_Efficiency", "Dividend_Payout",
    "Yearly_Price_Change", "Composite_Rank", "Net_Debt_to_Equity"
]

GRU_LOOKBACK = 60

NEWS_KEYWORDS = {
    "ANTM": ["ANTM saham", "Antam emas", "Aneka Tambang saham", "ANTM emas"],
    "MDKA": ["MDKA saham", "Merdeka Copper Gold saham", "MDKA emas"],
    "BRMS": ["BRMS saham", "Bumi Resources Minerals saham", "BRMS emas"],
    "PSAB": ["PSAB saham", "J Resources Asia Pasifik saham", "PSAB emas", "J Resources gold", "J Resources tambang emas"]
}

MARKET_NEWS_CATEGORIES = {
    "Semua Berita": [],
    "Emas": ["harga emas hari ini", "emas dunia", "emas antam", "harga emas naik", "harga emas turun", "gold price today"],
    "Saham & IHSG": ["IHSG hari ini", "saham Indonesia", "pasar modal Indonesia", "Bursa Efek Indonesia", "rekomendasi saham hari ini"],
    "Ekonomi Indonesia": ["ekonomi Indonesia", "inflasi Indonesia", "suku bunga Bank Indonesia", "rupiah hari ini", "pertumbuhan ekonomi Indonesia"],
    "Ekonomi Global": ["ekonomi global", "The Fed suku bunga", "inflasi Amerika", "resesi global", "geopolitical risk market"],
    "Komoditas": ["harga komoditas", "harga minyak dunia", "harga batu bara", "harga tembaga", "commodity market"],
    "Tren Pasar": ["saham ramai dibahas", "market trend today", "investor sentiment", "fear of missing out saham", "berita ekonomi viral"]
}

MARKET_POSITIVE_WORDS = [
    "naik", "menguat", "positif", "tumbuh", "meningkat", "optimis", "rebound", 
    "bullish", "stabil", "surplus", "rekor", "cuan", "laba", "akumulasi", "prospek"
]

MARKET_NEGATIVE_WORDS = [
    "turun", "melemah", "negatif", "anjlok", "koreksi", "tertekan", "inflasi", 
    "resesi", "krisis", "rugi", "bearish", "risiko", "ketidakpastian", "perang", 
    "konflik", "gagal", "defisit"
]

POSITIVE_WORDS = [
    "naik", "menguat", "positif", "cuan", "untung", "laba", "melonjak",
    "tumbuh", "meningkat", "rekor", "prospek", "bagus", "cerah",
    "buy", "akumulasi", "bullish", "rebound", "mengkilap", "diburu",
    "rekomendasi", "target", "optimis", "ekspansi", "dividen"
]

NEGATIVE_WORDS = [
    "turun", "melemah", "negatif", "rugi", "anjlok", "merosot",
    "tertekan", "koreksi", "jatuh", "lesu", "beban", "risiko",
    "sell", "hindari", "bearish", "jeblok", "ambrol", "tekanan",
    "utang", "turunnya", "penurunan", "waspada"
]

STOPWORDS_ID = {
    "dan", "yang", "di", "ke", "dari", "untuk", "dengan", "dalam", "hari", 
    "ini", "terbaru", "adalah", "pada", "karena", "sebagai", "itu", "akan", 
    "bisa", "ada", "tidak", "juga", "sudah", "saja", "lagi", "atau", "oleh",
    "untuk", "kita", "kami", "saya", "kamu", "mereka", "dia", "saat", "bagi"
}
