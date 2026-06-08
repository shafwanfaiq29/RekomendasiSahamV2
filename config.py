TICKER_MAP = {
    "ANTM": "ANTM.JK",
    "MDKA": "MDKA.JK",
    "BRMS": "BRMS.JK",
    "PSAB": "PSAB.JK",
    'ARCI': 'ARCI.JK',
    'UNTR': 'UNTR.JK',
    'AMMN': 'AMMN.JK',
    'HRTA': 'HRTA.JK',
}

COMPANY_NAMES = {
    "ANTM": "PT Aneka Tambang Tbk",
    "MDKA": "PT Merdeka Copper Gold Tbk",
    "BRMS": "PT Bumi Resources Minerals Tbk",
    "PSAB": "PT J Resources Asia Pasifik Tbk",
    "ARCI": "PT Archi Indonesia Tbk",
    "UNTR": "PT United Tractors Tbk",
    "AMMN": "PT Amman Mineral Internasional Tbk",
    "HRTA": "PT Hartadinata Abadi Tbk",
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
    "PSAB": ["PSAB saham", "J Resources Asia Pasifik saham", "PSAB emas", "J Resources gold", "J Resources tambang emas"],
    "ARCI": ["ARCI saham", "Archi Indonesia tambang emas", "ARCI produksi emas Sulawesi", "Archi Indonesia kinerja", "ARCI Toka Tindung"],
    "UNTR": ["UNTR saham", "United Tractors tambang batubara", "UNTR emas DEWA", "United Tractors kinerja alat berat", "UNTR dividen emiten"],
    "AMMN": ["AMMN saham", "Amman Mineral tembaga emas", "AMMN smelter Sumbawa", "Amman Mineral Batu Hijau", "AMMN produksi ekspor"],
    "HRTA": ["HRTA saham", "Hartadinata perhiasan emas", "HRTA penjualan emas retail", "Hartadinata Abadi kinerja", "HRTA harga emas permintaan"],
}

MARKET_NEWS_CATEGORIES = {
    "Semua Berita": [],
    "Emas": ["harga emas hari ini", "emas dunia", "emas antam", "harga emas naik", "harga emas turun", "gold price today"],
    "Saham & IHSG": ["IHSG hari ini", "saham Indonesia", "pasar modal Indonesia", "Bursa Efek Indonesia", "rekomendasi saham hari ini"],
    "Ekonomi Global": ["ekonomi global", "The Fed suku bunga", "inflasi Amerika", "resesi global", "geopolitical risk market"],
    "Komoditas": ["harga komoditas", "harga minyak dunia", "harga batu bara", "harga tembaga", "commodity market"],
    "Tren Pasar": ["saham ramai dibahas", "market trend today", "investor sentiment", "fear of missing out saham", "berita ekonomi viral"],
    
    # --- TAMBAHIN EMITEN DI SINI CUY ---
    "ANTM": ["ANTM saham", "Antam emas", "Aneka Tambang saham", "ANTM emas"],
    "MDKA": ["MDKA saham", "Merdeka Copper Gold saham", "MDKA emas"],
    "BRMS": ["BRMS saham", "Bumi Resources Minerals saham", "BRMS emas"],
    "PSAB": ["PSAB saham", "J Resources Asia Pasifik saham", "PSAB emas"],
    "ARCI": ["Archi Indonesia", "Archi", "ARCI", "PT Archi", "Tambang Archi"],
    "UNTR": ["United Tractors", "United Tractor", "UNTR", "PT United Tractors", "UT"],
    "AMMN": ["Amman Mineral", "Amman", "AMMN", "PT Amman Mineral", "Amman Internasional"],
    "HRTA": ["Hartadinata", "Hartadinata Abadi", "HRTA", "PT Hartadinata", "Harta Abadi",],
    
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


POS_MINING = [
    # --- Operasional & Produksi ---
    "smelter", "ekspansi", "produksi meningkat",
    "kapasitas naik", "throughput tinggi", "cadangan bertambah",
    "sumber daya baru", "temuan baru", "eksplorasi berhasil",
    "grade tinggi", "kadar tinggi", "recovery rate tinggi",
    "efisiensi produksi","output meningkat", "target tercapai",
    "produksi melebihi target", "commissioning", "ramp up",
    "full operation", "on track", "sesuai jadwal", "ahead of schedule",
    "reserve upgrade", "resource upgrade", "feasibility study positif",
    "studi kelayakan lulus", "perpanjangan izin", "izin diterima",
    "IUP diperpanjang", "RKAB disetujui", "kuota ekspor naik", "kuota meningkat",

    # --- Keuangan & Profitabilitas ---
    "dividen", "dividen naik", "dividen spesial", "laba bersih naik",
    "laba melonjak", "profit meningkat", "margin naik", "EBITDA tinggi",
    "EBITDA positif","pendapatan naik", "revenue tumbuh", "cashflow positif",
    "arus kas kuat", "kas meningkat", "utang turun", "DER rendah", "rasio utang turun",
    "net cash", "zero debt", "bebas utang", "return on equity tinggi", "ROE tinggi",
    "ROA meningkat", "EPS naik", "earnings per share naik", "laba per saham meningkat",
    "buyback", "pembelian kembali saham", "stock buyback",

    # --- Harga & Pasar Saham ---
    "rebound", "rally", "bullish", "breakout", "all time high", "ATH",
    "naik signifikan", "melesat", "terbang", "melonjak", "cuan",
    "untung", "profit taking positif", "uptrend", "tren naik", "golden cross",
    "support kuat", "permintaan tinggi", "beli", "akumulasi", "oversold rebound",
    "potensi naik", "target harga naik", "upgrade", "rekomendasi beli", "strong buy",
    "outperform", "overweight", "katalis positif", "sentimen positif",

    # --- Harga Emas & Komoditas ---
    "harga emas naik", "gold price up", "emas menguat", "harga spot naik", "harga komoditas naik",
    "harga tembaga naik", "harga nikel naik", "harga logam mulia naik", "dolar melemah",
    "safe haven demand", "inflasi naik", "geopolitik mendorong emas", "fed dovish",
    "suku bunga turun", "interest rate cut", "quantitative easing", "risk off",

    # --- Korporasi & Strategis ---
    "akuisisi strategis", "merger menguntungkan", "joint venture", "kemitraan strategis",
    "kontrak baru", "offtake agreement", "kontrak jangka panjang", "investasi masuk",
    "FDI masuk", "divestasi menguntungkan", "listing baru", "rights issue sukses",
    "IPO sukses", "rating naik", "upgrade rating", "investment grade", "ESG positif",
    "sertifikasi lingkungan", "CSR award", "penghargaan", "manajemen baru positif",
]

NEG_MINING = [
    # --- Operasional & Produksi ---
    "ambrol", "longsor", "kecelakaan tambang", "fatality", "korban jiwa",
    "operasi dihentikan", "suspensi operasi", "force majeure", "banjir tambang",
    "gempa merusak", "kebakaran", "ledakan", "kadar rendah", "grade rendah",
    "dilusi bijih", "ore dilution", "throughput turun", "produksi turun",
    "output menurun", "target tidak tercapai", "miss target", "below target",
    "cadangan menipis", "sumber daya berkurang", "reserve revision turun",
    "umur tambang pendek", "mine life berkurang", "biaya produksi naik",
    "AISC tinggi", "all-in sustaining cost naik", "ongkos produksi melonjak",
    "overrun biaya", "cost overrun", "keterlambatan proyek", "delay proyek",
    "commissioning terlambat",

    # --- Regulasi & Perizinan ---
    "izin dicabut", "IUP dicabut", "suspensi izin", "moratorium", "larangan ekspor",
    "kuota ekspor dipotong", "royalti tinggi", "royalti naik", "pajak naik",
    "beban pajak meningkat", "sengketa lahan", "konflik komunitas", "penolakan warga",
    "demo warga", "gugatan hukum", "somasi", "perkara hukum", "denda lingkungan",
    "sanksi lingkungan", "pencemaran", "pelanggaran AMDAL", "audit negatif", "temuan BPK",

    # --- Keuangan & Saham ---
    "rugi", "kerugian", "merugi", "defisit","pendapatan turun","revenue turun",
    "laba turun", "profit turun", "margin tertekan", "EBITDA negatif", "cashflow negatif",
    "arus kas negatif", "utang naik", "DER tinggi", "rasio utang tinggi",
    "beban bunga tinggi", "gagal bayar", "default", "restrukturisasi utang", "pailit",
    "bangkrut","delisting", "suspensi saham", "auto reject bawah", "ARB", "koreksi tajam",
    "ambles","anjlok","terjun","crash","bearish","downtrend","tren turun",
    "dead cross","breakdown support","jual","tekanan jual","sell off",
    "profit taking negatif","downgrade","rekomendasi jual","strong sell",
    "underperform","underweight",

    # --- Harga Emas & Komoditas ---
    "harga emas turun","gold price down","emas melemah","harga spot turun",
    "harga komoditas turun","dolar menguat","fed hawkish","suku bunga naik",
    "interest rate hike","risk on","kenaikan yield obligasi",

    # --- Korporasi & Manajerial ---
    "pengunduran diri direksi","pergantian CEO mendadak","masalah tata kelola",
    "fraud","penggelapan","korupsi","manipulasi laporan","audit disclaimer",
    "opini tidak wajar","going concern","whistleblower","investigasi OJK",
    "investigasi BEI","restatement laporan keuangan",
]

NEUTRAL_AMBIGUOUS = [
    # --- Pasar & Bursa ---
    "bursa", "bursa efek", "BEI", "IDX", "IHSG", "indeks", "emiten", "saham", 
    "efek", "sekuritas", "market cap", "kapitalisasi pasar", 
    "volume perdagangan", "nilai transaksi", "frekuensi", "lot", "bid", "offer",
      "spread", "likuiditas", "free float",

    # --- Teknis Analisis ---
    "fluktuatif", "volatil", "volatilitas", "sideways", "konsolidasi", "ranging", 
    "koreksi", "penyesuaian", "retracement", "support", "resistance", 
    "moving average", "MA", "RSI", "MACD", "bollinger band", "fibonacci", 
    "chart", "grafik", "pola teknikal", "candlestick", "volume", 
    "open interest",

    # --- Fundamental & Valuasi ---
    "estimasi", "proyeksi", "konsensus", "target price", "fair value", 
    "PER", "price to earnings", "PBV", "price to book value", "EV/EBITDA", 
    "DCF", "discounted cash flow", "valuasi", "overvalued", "undervalued", 
    "wajar", "coverage", "inisiasi coverage", "laporan analis", "riset", 
    "research report",

    # --- Operasional Tambang (Teknis Netral) ---
    "kegiatan pertambangan", "kegiatan eksplorasi", "pengeboran", "drilling", 
    "infill drilling", "step out drilling", "sampling", "assay", "uji lab", 
    "JORC", "NI 43-101", "feasibility study", "pre-feasibility", 
    "scoping study", "ESIA", "AMDAL", "reklamasi", "pasca tambang", 
    "closure plan", "pit", "open pit", "underground", "stope", "haulage", 
    "mill", "processing plant", "heap leach", "CIL", "carbon in leach", 
    "flotasi", "concentrate", "tailing", "waste dump", "stripping ratio", 
    "bench",

    # --- Regulasi & Administrasi ---
    "RKAB", "IUP", "IUPK", "PKP2B", "kontrak karya", "divestasi", "divestment", 
    "kewajiban divestasi", "BPKP", "OJK", "kementerian ESDM", "ESDM", 
    "rapat umum pemegang saham", "RUPS", "RUPSLB", "paparan publik", 
    "keterbukaan informasi", "disclosure", "corporate action", "aksi korporasi", 
    "stock split", "reverse stock", "rights issue", "waran",

    # --- Makroekonomi ---
    "dinamis", "dinamika", "siklus", "siklus komoditas", "supercycle", 
    "inflasi", "deflasi", "suku bunga", "kebijakan moneter", "kebijakan fiskal", 
    "nilai tukar", "kurs", "rupiah", "dolar", "geopolitik", "resesi", 
    "pertumbuhan ekonomi", "GDP", "PMI", "data ekonomi", "rilis data", 
    "neraca perdagangan", "ekspor impor",
]