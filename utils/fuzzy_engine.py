# utils/fuzzy_engine.py
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def hitung_rekomendasi_fuzzy(prediksi_return, skor_sentimen, skor_fundamental):
    """
    Mengawinkan 3 Pilar Analisis menggunakan aturan logika fuzi.
    """
    # Regulasi Semesta Pembicaraan (Universe of Discourse)
    in_return = ctrl.Antecedent(np.arange(-10, 11, 0.1), 'return_ai')
    in_sentimen = ctrl.Antecedent(np.arange(-1, 1.1, 0.1), 'sentimen')
    in_fund = ctrl.Antecedent(np.arange(-1, 1.1, 0.1), 'fundamental')
    out_keputusan = ctrl.Consequent(np.arange(0, 101, 1), 'rekomendasi')
    
    # Keanggotaan otomatis (Membership Functions)
    in_return.automf(names=['bearish', 'stagnan', 'bullish'])
    in_sentimen.automf(names=['negatif', 'netral', 'positif'])
    in_fund.automf(names=['sakit', 'biasa', 'sehat'])
    out_keputusan.automf(names=['hindari', 'jangka_pendek', 'jangka_panjang'])
    
    # Basis Aturan Kontrol Fuzi (Fuzzy Rule Base)
    r1 = ctrl.Rule(in_fund['sakit'] | in_sentimen['negatif'], out_keputusan['hindari'])
    r2 = ctrl.Rule(in_fund['sehat'] & in_return['bullish'] & in_sentimen['positif'], out_keputusan['jangka_panjang'])
    r3 = ctrl.Rule(in_fund['biasa'] & in_sentimen['positif'], out_keputusan['jangka_pendek'])
    r4 = ctrl.Rule(in_return['bearish'] & in_fund['sakit'], out_keputusan['hindari'])
    r5 = ctrl.Rule(in_return['bullish'] & in_fund['sehat'], out_keputusan['jangka_panjang'])
    
    # Pemrosesan Mamdani
    sistem_kontrol = ctrl.ControlSystem([r1, r2, r3, r4, r5])
    simulasi = ctrl.ControlSystemSimulation(sistem_kontrol)
    
    simulasi.input['return_ai'] = np.clip(prediksi_return * 100, -10, 10)
    simulasi.input['sentimen'] = np.clip(skor_sentimen, -1, 1)
    simulasi.input['fundamental'] = np.clip(skor_fundamental, -1, 1)
    
    try:
        simulasi.compute()
        return simulasi.output['rekomendasi']
    except Exception:
        # Nilai default jika terjadi kendala defuzzifikasi diluar bounds
        return 50.0