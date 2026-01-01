import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config("Dashboard Ruko", layout="wide")

# ============================================
# LOAD MODEL
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
# SIDEBAR
# ============================================
st.sidebar.title("Pengaturan Model")
model_name = st.sidebar.selectbox("Segment Ruko", models.keys())
model = models[model_name]

feature_names = list(model.feature_names_in_)

# ============================================
# INISIALISASI INPUT
# ============================================
X_input = {f: 0 for f in feature_names}

# ============================================
# PISAHKAN NUMERIK & OHE
# ============================================
numeric_features = []
categorical_groups = {}

for f in feature_names:
    if "_" in f:
        prefix = f.split("_")[0]
        categorical_groups.setdefault(prefix, []).append(f)
    else:
        numeric_features.append(f)

# ============================================
# UI
# ============================================
st.title("Dashboard Prediksi Harga Ruko per m²")
st.subheader(f"Gradient Boosting Regressor – {model_name}")
st.markdown("---")

# ============================================
# INPUT NUMERIK
# ============================================
st.subheader("Variabel Numerik")

num_cols = st.columns(3)
for i, f in enumerate(numeric_features):
    with num_cols[i % 3]:
        X_input[f] = st.number_input(
            f.replace("_", " ").title(),
            min_value=0.0
        )

# ============================================
# INPUT KATEGORIK (DROPDOWN → OHE)
# ============================================
st.subheader("Variabel Kategorik")

cat_cols = st.columns(3)
for i, (prefix, cols) in enumerate(categorical_groups.items()):
    with cat_cols[i % 3]:

        options = [
            c.replace(prefix + "_", "") for c in cols
        ]

        choice = st.selectbox(
            prefix.replace("_", " ").title(),
            options
        )

        selected_col = f"{prefix}_{choice}"
        if selected_col in X_input:
            X_input[selected_col] = 1

# ============================================
# DATAFRAME SESUAI MODEL
# ============================================
X_df = pd.DataFrame([X_input])[feature_names]

# ============================================
# PREDIKSI
# ============================================
if st.button("🔮 Prediksi Harga"):
    y_log = model.predict(X_df)[0]
    y = np.exp(y_log)

    st.success("Prediksi berhasil")
    st.metric("Harga per m² (Rp)", f"{y:,.0f}")

# ============================================
# FEATURE IMPORTANCE
# ============================================
st.markdown("---")
st.subheader("Feature Importance")

fi = pd.DataFrame({
    "Variabel": feature_names,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

st.dataframe(fi, use_container_width=True)

fig, ax = plt.subplots(figsize=(6, 4))
ax.barh(fi["Variabel"], fi["Importance"])
ax.invert_yaxis()
ax.set_title("Feature Importance")
st.pyplot(fig)
