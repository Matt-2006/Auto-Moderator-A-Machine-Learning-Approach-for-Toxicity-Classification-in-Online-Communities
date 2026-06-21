# 🛡️ Auto-Moderator — Streamlit App

Multi-label Toxicity Classifier · COMP6885001 Natural Language Processing · Group 11

## Struktur Folder

```
auto_moderator_app/
├── app.py               ← Main Streamlit application
├── requirements.txt     ← Python dependencies
├── README.md            ← This file
└── models/
    └── distilbert_model.pt   ← (letakkan model di sini)
```

## Setup & Menjalankan

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Letakkan model checkpoint
Copy file `distilbert_model.pt` dari hasil training notebook (output di `Model Training/Training Model.ipynb`) ke folder `models/`:
```bash
mkdir models
cp /path/to/distilbert_model.pt models/
```

Atau set environment variable jika model ada di lokasi lain:
```bash
export MODEL_BASE_DIR=/path/to/your/models/folder
```

### 3. Jalankan aplikasi
```bash
streamlit run app.py
```

Aplikasi akan berjalan di `http://localhost:8501`

## Fitur Aplikasi

### Tab 1 — Single Comment
- Input teks bebas
- Klasifikasi real-time dengan DistilBERT
- Verdict: 🔴 Toxic / 🟠 Suspicious / 🟢 Safe
- Per-label score dengan progress bar dan threshold marker
- 6 contoh komentar siap klik

### Tab 2 — Batch Classification
- Input banyak komentar (satu per baris)
- Tabel hasil dengan score per label
- Summary statistik (total toxic / suspicious / safe)

### Tab 3 — Reference
- Definisi setiap label + threshold
- Penjelasan logika verdict
- Tabel perbandingan performa semua model

## Model

- **Architecture:** DistilBERT (`distilbert-base-uncased`) + Dropout(0.3) + Linear(768→6) + Sigmoid
- **Training:** BCEWithLogitsLoss · AdamW lr=2e-5 · 3 epochs · batch=32
- **Validation Macro ROC-AUC:** 0.9839 ✅ (target > 0.90)
- **Validation Macro F1:** 0.5229
- **Hamming Loss:** 0.0504

## Decision Threshold

Semua 6 label menggunakan **uniform threshold 0.50**, konsisten dengan evaluasi saat training.

| Verdict | Kondisi |
|---|---|
| 🔴 Toxic | Minimal 1 label score ≥ 0.50 |
| 🟠 Suspicious | Minimal 1 label score ≥ 0.25, tidak ada yang ≥ 0.50 |
| 🟢 Safe | Semua label score < 0.25 |

## Performa Semua Model

| Model | ROC-AUC | F1 | Precision | Recall | Hamming |
|---|---|---|---|---|---|
| **DistilBERT** ★ | **0.9839** | 0.5229 | 0.3747 | 0.9068 | 0.0504 |
| RoBERTa | 0.9826 | 0.5031 | 0.3551 | 0.9224 | 0.0526 |
| LinearSVC | 0.9747 | 0.5401 | 0.5848 | 0.5142 | 0.0267 |
| Logistic Regression | 0.9746 | 0.4725 | 0.3415 | 0.8445 | 0.0502 |

★ App menggunakan DistilBERT (ROC-AUC tertinggi). Semua model memenuhi target > 0.90 ✅

## Team

- Matthew Fitch Aurick — 2802389922
- Jayson Prasada Siswoyo — 2802389260
- Evan Chastya Pahan — 2802394185

Bina Nusantara University · 2025/2026
