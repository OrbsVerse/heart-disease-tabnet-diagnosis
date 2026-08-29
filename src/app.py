import streamlit as st
import pandas as pd
import joblib
from pytorch_tabnet.tab_model import TabNetClassifier

# --- Konstanta path model dan scaler ---
MODEL_FILE = "best_tabnet_model.zip"
SCALER_FILE = "minmax_scaler.pkl"

# --- Daftar fitur ---
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope"]
FEATURE_ORDER = CATEGORICAL_FEATURES + NUMERIC_FEATURES


# --- Fungsi memuat model dan scaler ---
@st.cache_resource
def load_model_and_scaler():
    try:
        scaler = joblib.load(SCALER_FILE)
        model = TabNetClassifier()
        model.load_model(MODEL_FILE)
        return model, scaler
    except Exception as e:
        st.error(f"❌ Gagal memuat model atau scaler: {e}")
        st.stop()


model, scaler = load_model_and_scaler()

# --- Tampilan Utama ---
st.title("Prediksi Penyakit Jantung 🫀 dengan TabNet")
st.markdown(
    "Masukkan data pasien di sidebar untuk memprediksi risiko penyakit jantung."
)

# --- Input Sidebar ---
st.sidebar.header("Input Data Pasien")
input_data = {}

# Input numerik
numeric_defaults = {
    "age": (1, 120, 50),
    "trestbps": (80, 200, 120),
    "chol": (100, 600, 200),
    "thalach": (60, 250, 150),
    "oldpeak": (0.0, 10.0, 1.0, 0.1),
}

for feature, params in numeric_defaults.items():
    label = {
        "age": "Usia (tahun)",
        "trestbps": "Tekanan Darah Istirahat (mm Hg)",
        "chol": "Kolesterol Serum (mg/dl)",
        "thalach": "Detak Jantung Maksimum",
        "oldpeak": "ST Depression",
    }[feature]

    if len(params) == 3:
        input_data[feature] = st.sidebar.number_input(label, *params)
    else:
        input_data[feature] = st.sidebar.number_input(label, *params)

# Input kategorikal (dengan mapping)
category_options = {
    "sex": {"Wanita": 0, "Pria": 1},
    "cp": {
        "Typical Angina (1)": 1,
        "Atypical Angina (2)": 2,
        "Non-anginal Pain (3)": 3,
        "Asymptomatic (4)": 4,
    },
    "fbs": {"≤120 mg/dl (0)": 0, ">120 mg/dl (1)": 1},
    "restecg": {
        "Normal (0)": 0,
        "ST-T Abnormal (1)": 1,
        "LV Hypertrophy (2)": 2,
    },
    "exang": {"Tidak (0)": 0, "Ya (1)": 1},
    "slope": {"Upsloping (1)": 1, "Flat (2)": 2},
}

default_index = {
    "sex": 1,
    "cp": 3,
    "fbs": 0,
    "restecg": 0,
    "exang": 0,
    "slope": 1,
}

for feature, options in category_options.items():
    selected_label = st.sidebar.selectbox(
        label={
            "sex": "Jenis Kelamin",
            "cp": "Tipe Nyeri Dada",
            "fbs": "Gula Darah Puasa",
            "restecg": "Hasil EKG",
            "exang": "Angina karena Latihan",
            "slope": "Slope ST Segment",
        }[feature],
        options=list(options.keys()),
        index=default_index[feature],
    )
    input_data[feature] = options[selected_label]

# --- Proses Prediksi ---
if st.sidebar.button("Prediksi Risiko"):
    try:
        input_df = pd.DataFrame([input_data])
        input_df = input_df[FEATURE_ORDER]

        # Normalisasi hanya pada fitur numerik
        input_df[NUMERIC_FEATURES] = scaler.transform(input_df[NUMERIC_FEATURES])

        st.subheader("Data Pasien")
        st.dataframe(input_df)

        # Prediksi
        probability = model.predict_proba(input_df.values)[0][1]
        prediction = model.predict(input_df.values)[0]

        st.subheader("Hasil Prediksi")
        st.write(f"Probabilitas penyakit jantung: **{probability:.4f}**")

        if prediction == 1:
            st.error(
                "⚠️ Risiko Tinggi: Pasien diprediksi **MEMILIKI** penyakit jantung."
            )
        else:
            st.success(
                "✅ Risiko Rendah: Pasien diprediksi **TIDAK MEMILIKI** penyakit jantung."
            )

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat prediksi: {e}")
