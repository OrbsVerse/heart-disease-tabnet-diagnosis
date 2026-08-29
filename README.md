# Heart Disease Diagnosis System with TabNet

A deep learning–based diagnostic support system for heart disease risk prediction, built using **TabNet** — a deep learning architecture for tabular data with built-in attention-based feature selection. Deployed as an interactive Streamlit web app for real-time patient risk prediction.

## 🧪 Dataset

- 3,235-record clinical dataset
- Outliers handled via the **IQR method**

## 🧠 Modeling

- Preprocessing: numerical/categorical feature separation, `MinMaxScaler` normalization (tailored to TabNet's input requirements)
- Validated via **Stratified 5-Fold Cross Validation**, tuning learning rate, batch size, and early stopping
- Model persistence via `TabNetClassifier.save_model()` and `joblib`

## 📊 Results (average across folds)

| Metric | Score |
|---|---|
| Accuracy | 70.9% |
| Precision | 71.1% |
| Recall | 75.8% |

Consistent performance across folds with no signs of overfitting.

## 🖥️ Streamlit App (`src/app.py`)

- Sidebar form for patient feature input
- Loads the trained TabNet model and scaler automatically
- Displays real-time heart disease risk prediction

## 📂 Project Structure
```
data/ # Raw and cleaned datasets
notebook/ # Preprocessing and TabNet modeling notebooks
model/ # Saved TabNet model and scaler
src/ # Streamlit application
```

## ⚙️ Tech Stack

Python · PyTorch (TabNet) · Scikit-learn · Streamlit

## 🚀 Running Locally

```bash
pip install -r requirements.txt
streamlit run src/app.py
```
