 
# Blood Pressure Estimation Using PPG–ECG Features (MIMIC-IV Waveform)
 
Predicting systolic and diastolic blood pressure using **PPG** and **ECG** waveforms from the **MIMIC-IV Waveform Database**, combined with handcrafted physiological features and an **XGBoost Regressor** model.

> **⚠️ Research use only. Not intended for clinical diagnosis or treatment.**

---

## 📌 Overview
This project implements a complete pipeline for **non-invasive blood pressure estimation** using:
- PPG waveforms from a smartwatch-style sensor  
- ECG waveforms (single-lead)  
- Physiological features described in a 2022 *Nature Scientific Reports* paper  
- Machine-learning regression using **XGBoost**

The system aligns ECG–PPG signals, extracts features, trains two regression models, and produces SBP/DBP predictions.

---

## ⭐ Results
| Metric | Mean Absolute Error (MAE) |
|--------|----------------------------|
| **Systolic BP (SBP)** | **≈ 14 mmHg** |
| **Diastolic BP (DBP)** | **≈ 9 mmHg** |

These results are comparable to reported feature-based baselines in literature.

---

## 📂 Dataset

This work uses the **MIMIC-IV Waveform Database v0.1.0**, available on PhysioNet:

🔗 https://physionet.org/content/mimic4wdb/0.1.0/

Signals used:
- **PPG** (photoplethysmogram)
- **ECG** (single lead)
- Metadata for SBP/DBP reference

All signals are aligned, filtered, segmented, and converted into numerical features.

---

## 📘 Feature Extraction

### 📑 Source Paper  
Features follow the definitions from:  
**"Non-invasive Blood Pressure Estimation Using PPG Waveform Analysis" — Nature Scientific Reports (2022)**  
🔗 https://www.nature.com/articles/s41598-022-27170-2

### 🟥 PPG Features
feat_notch_amp
feat_reflective_idx
feat_delta_T
feat_crest_time
feat_T_sys
feat_T_dia
feat_T_ratio
feat_stt
feat_A1
feat_A2
feat_inflection_point_area
feat_width_25
feat_width_50
feat_skew
feat_kurtosis
feat_sys_mu
feat_sys_sigma
feat_dia_mu
feat_dia_sigma

### 🟦 ECG + PPG Combined Features
mean_rr_ms # RR-interval mean (heart rate variability)
feat_ptt # Pulse Transit Time (ECG R-peak → PPG foot)

Total engineered features: **21**.

---





---

## ⭐ Key Features

* Extracts rich **time-domain & frequency-domain PPG + ECG features**
* Uses **XGBoost Regression** models for SBP and DBP
* Full ML pipeline: **raw signal → features → model → predictions**
* Based on **MIMIC-IV Waveform dataset**
* Includes **training, inference, and evaluation** scripts
* Ready for **deployment on wearable devices**

---

## ⚙️ Example Model Configuration

```python
XGBRegressor(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.8,
    objective="reg:squarederror"
)
```

**Trained Models:**

* `xgb_sbp.json` — Predicts **Systolic Blood Pressure**
* `xgb_dbp.json` — Predicts **Diastolic Blood Pressure**

---

## 📦 Repository Structure

```
├── features.py
│── ecg_series_and_ppg_series_alignment_with_bp_changes_patrick_modification.py
│── onsets.py
│── train.py
├── xgb_sbp.json
│── xgb_dbp.json
│── features.csv 
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 2️⃣ Download MIMIC-IV Waveform Data

Requires a PhysioNet account.

```bash
pip install wfdb
wfdb-download -p mimic4wdb/0.1.0 -o data/raw/
```

---

### 3️⃣ Preprocess Signals and extract Features(ECG + PPG)

```bash
#Before running it add the data path and import features.py and onsets.py in it.

python ecg_series_and_ppg_series_alignment_with_bp_changes_patrick_modification.py 
```

---



---

### 6️⃣ Train the Model

```bash
#Train the model and add the features files csv in this for training.
python train.py 
```

---

## 📊 Evaluation

Includes visualizations:

* Shap values for the systolic and diastolic blood pressure

**Example Performance:**

```
SBP MAE: 14.1 mmHg
DBP MAE: 9.3 mmHg
```

---

## ⚠️ Medical Disclaimer

This software is for **research and educational purposes only**.
Do **not** use it for medical decisions, diagnosis, or clinical monitoring.

---

## 📜 License

MIT License — see `LICENSE`.

---

## 🙏 Acknowledgements

* PhysioNet & MIT Laboratory for Computational Physiology
* Nature Scientific Reports (2022) — PPG feature engineering
* WFDB Python package
* XGBoost authors

---

## 📬 Contact

**Your Name**
Email: [bilalzubairi031@gmail.com](mailto:bilalzubairi031@gmail.com)
GitHub: [https://github.com/bil21071(https://github.com/bil21071)

---



---

This version will display perfectly on GitHub.
