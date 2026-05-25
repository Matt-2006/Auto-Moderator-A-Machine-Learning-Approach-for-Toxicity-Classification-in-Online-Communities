# 🛡️ Auto-Moderator — Streamlit App

Multi-label Toxicity Classifier · COMP6885001 Natural Language Processing · Group 11

## Struktur Folder

```
auto_moderator_app/
├── app.py               ← Main Streamlit application
├── requirements.txt     ← Python dependencies
├── README.md            ← This file
└── models/
    └── distilbert_model.pt   ← (letakkan checkpoint di sini)
```

## Setup & Menjalankan

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Letakkan model checkpoint
Copy file `distilbert_model.pt` dari hasil training notebook (output di `Model 3/distilbert_model.pt`) ke folder `models/`:
```bash
mkdir models
cp /path/to/output/Model\ 3/distilbert_model.pt models/
```

Atau set environment variable jika model ada di lokasi lain:
```bash
export MODEL_BASE_DIR=/path/to/your/output/folder
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
- **Test Macro ROC-AUC:** 0.9851 ✅ (target > 0.90)
- **Test Macro F1:** 0.6162
- **Hamming Loss:** 0.030008

## Per-Label Thresholds

| Label | Threshold | Alasan |
|---|---|---|
| toxic | 40% | Label utama, threshold moderat |
| severe_toxic | 30% | Sangat langka (1%), perlu threshold rendah |
| obscene | 40% | Cukup umum |
| threat | 25% | Paling langka (0.30%), threshold agresif |
| insult | 40% | Cukup umum |
| identity_hate | 25% | Sangat langka (0.88%), threshold agresif |

## Team

- Matthew Fitch Aurick — 2802389922
- Jayson Prasada Siswoyo — 2802389260
- Evan Chastya Pahan — 2802394185

Bina Nusantara University · 2025/2026
