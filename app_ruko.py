import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ============================================
# 1. KONFIGURASI HALAMAN & CSS
# ============================================
st.set_page_config(
    page_title="Dashboard Valuasi Ruko",
    page_icon="🏢",
    layout="wide"
)

# Styling CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        color: #1E3A8A;
        font-weight: 800;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6B7280;
        margin-top: -10px;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 20px;
        border-radius: 12px;
        border-left: 6px solid #2563EB;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================
# 2. LOAD MODEL
# ============================================
@st.cache_resource
def load_models():
    return {
        "Low": joblib.load("model_Low (2).joblib"),
        "Medium": joblib.load("model_Medium.joblib"),
        "High": joblib.load("model_High (2).joblib")
    }

models = load_models()

# ============================================
# 3. SIDEBAR (PENGATURAN)
# ============================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/609/609803.png", width=70)
    st.title("⚙️ Panel Kontrol")
    
    st.markdown("### Pilih Model")
    model_name = st.selectbox("Segmentasi Ruko", models.keys())
    model = models[model_name]
    
    st.info(f"Model Aktif: **{model_name} Segment**")
    st.markdown("---")
    st.caption("Dashboard Estimasi Nilai Pasar Wajar (NPW) Ruko.")

feature_names = list(model.feature_names_in_)

# ============================================
# 4. LOGIKA PARSING FITUR (UPDATE: JARAK -> KATEGORIK)
# ============================================

numeric_features = []
categorical_groups = {}

# UPDATE: Saya menghapus 'jarak' dari sini agar dia masuk ke kategori (dropdown)
numeric_keywords = [
    'luas', 'lebar', 'panjang', 'tinggi', 'row', 
    'lat', 'long', 'jumlah', 'nilai', 'harga', 'm2', 'tahun'
]

for f in feature_names:
    f_lower = f.lower()
    
    # Cek 1: Apakah mengandung kata kunci numerik?
    is_keyword_numeric = any(k in f_lower for k in numeric_keywords)
    
    # Cek 2: Apakah TIDAK ada underscore? (biasanya numerik murni)
    no_underscore = "_" not in f
    
    if is_keyword_numeric or no_underscore:
        numeric_features.append(f)
    else:
        # Masuk ke sini berarti Kategorik/OHE (termasuk Jarak)
        parts = f.split("_")
        prefix = parts[0] 
        categorical_groups.setdefault(prefix, []).append(f)

# Inisialisasi Dictionary Input dengan nilai 0
X_input = {f: 0 for f in feature_names}

# ============================================
# 5. UI UTAMA (INPUT)
# ============================================
st.markdown('<p class="main-header">🏢 Dashboard Valuasi Ruko</p>', unsafe_allow_html=True)
st.markdown(f'<p class="sub-header">Prediksi Harga per m² untuk Kategori: {model_name}</p>', unsafe_allow_html=True)

with st.container():
    tab1, tab2 = st.tabs(["📏 **Spesifikasi Fisik**", "📍 **Lokasi & Lingkungan**"])
    
    # --- TAB 1: INPUT NUMERIK (Luas, Lebar, ROW, dll) ---
    with tab1:
        st.write("Masukkan dimensi dan ukuran fisik:")
        if not numeric_features:
            st.info("Tidak ada input numerik.")
        
        num_cols = st.columns(3)
        for i, f in enumerate(numeric_features):
            with num_cols[i % 3]:
                clean_label = f.replace("_", " ").title().replace("M2", "(m²)").replace("M", "(m)")
                
                def_val = 0.0
                if 'luas' in f: def_val = 60.0
                elif 'lebar' in f or 'row' in f: def_val = 6.0
                
                X_input[f] = st.number_input(clean_label, min_value=0.0, value=def_val, key=f)

    # --- TAB 2: INPUT KATEGORIK (Jarak, Provinsi, Sertifikat, dll) ---
    with tab2:
        st.write("Pilih karakteristik lokasi, legalitas, dan akses:")
        
        if not categorical_groups:
            st.warning("Tidak ada variabel kategorik.")
        else:
            cat_cols = st.columns(3)
            for i, (prefix, cols) in enumerate(categorical_groups.items()):
                with cat_cols[i % 3]:
                    # Membersihkan opsi pilihan
                    options = [c.replace(prefix + "_", "") for c in cols]
                    
                    # Beautify text
                    display_options = [o.replace("_", " ").title() for o in options]
                    
                    label = prefix.replace("_", " ").title()
                    
                    # Dropdown Logic
                    choice_idx = st.selectbox(
                        label,
                        range(len(options)),
                        format_func=lambda x: display_options[x],
                        key=prefix
                    )
                    
                    actual_col_name = cols[choice_idx]
                    X_input[actual_col_name] = 1

st.markdown("---")

# ============================================
# 6. EKSEKUSI PREDIKSI & HASIL
# ============================================
c1, c2 = st.columns([1, 2])

with c1:
    st.markdown("### 🔮 Eksekusi")
    st.write("Klik tombol untuk kalkulasi harga.")
    predict_btn = st.button("Hitung Estimasi Harga", type="primary")

with c2:
    if predict_btn:
        X_df = pd.DataFrame([X_input])[feature_names]
        
        try:
            y_log = model.predict(X_df)[0]
            y_pred = np.exp(y_log)
            
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="margin:0; color:#4B5563;">Estimasi Harga Pasar Wajar:</h4>
                <h1 style="margin:0; color:#1E3A8A; font-size: 2.5em;">Rp {y_pred:,.0f} <span style="font-size:0.5em; color:#6B7280;">/ m²</span></h1>
            </div>
            """, unsafe_allow_html=True)
            
            if 'luas_tanah_m2' in X_input and X_input['luas_tanah_m2'] > 0:
                total_asset = y_pred * X_input['luas_tanah_m2']
                st.caption(f"Estimasi Total Aset (Tanah): **Rp {total_asset:,.0f}**")
                
        except Exception as e:
            st.error(f"Terjadi kesalahan pada model: {e}")
    else:
        st.info("👈 Silakan lengkapi data di atas dan klik tombol Hitung.")

# ============================================
# 7. FEATURE IMPORTANCE
# ============================================
st.markdown("### 📊 Faktor Penentu Harga")

with st.expander("Lihat Detail Grafik Pengaruh Variabel"):
    if hasattr(model, "feature_importances_"):
        fi = pd.DataFrame({
            "Variabel": feature_names,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False).head(10)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.barh(fi["Variabel"], fi["Importance"], color="#3B82F6")
        ax.invert_yaxis()
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('#DDDDDD')
        ax.set_xlabel("Tingkat Pengaruh (Importance)", color="#666666")
        
        st.pyplot(fig)
    else:
        st.info("Model ini tidak mendukung visualisasi Feature Importance.")
