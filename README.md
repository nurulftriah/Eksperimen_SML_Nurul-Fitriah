# Eksperimen Supervised Machine Learning - Nurul-Fitriah

Proyek ini memenuhi Kriteria 1 pada tingkat **Advanced** untuk submission Dicoding SML.  
Fokus pada analisis kelayakan kredit (Credit Scoring) menggunakan eksperimen preprocessing data, otomatisasi dengan Python script, dan CI/CD dengan GitHub Actions.

---

## Struktur Repositori

```
Eksperimen_SML_Nurul-Fitriah/
├── .github/
│   └── workflows/
│       ├── preprocess.yml          # Workflow otomatis preprocessing data
│       └── ci_mlflow.yml           # Workflow CI training & MLflow logging
├── .workflow/
│   ├── preprocess.yml              # Salinan workflow (advance)
│   └── ci_mlflow.yml               # Salinan workflow CI (advance)
├── namadataset_raw/
│   └── credit_scoring_raw.csv      # Dataset mentah
├── preprocessing/
│   ├── Eksperimen_Nurul_Fitriah.ipynb   # Notebook EDA & Preprocessing
│   ├── automate_Nurul-Fitriah.py        # Script preprocessing otomatis
│   └── namadataset_preprocessing/
│       └── credit_scoring_preprocessing.csv  # Dataset hasil preprocessing
├── conda.yaml                      # Environment untuk MLflow
├── modelling_tuning.py             # Script training & logging MLflow
└── README.md
```

---

## Cara Menjalankan

### A. Preprocessing Manual
```bash
python preprocessing/automate_Nurul-Fitriah.py
```
Output tersimpan di `preprocessing/namadataset_preprocessing/credit_scoring_preprocessing.csv`

### B. Trigger Workflow Otomatis
Workflow `preprocess.yml` akan otomatis jalan ketika:
- Ada perubahan di folder `namadataset_raw/`
- Ada perubahan di file `automate_Nurul-Fitriah.py`
- Di-trigger manual via tab Actions di GitHub

---

## GitHub Actions Workflow

| Workflow | Trigger | Fungsi |
|---|---|---|
| `preprocess.yml` | Push ke `namadataset_raw/` atau `automate_Nurul-Fitriah.py` | Jalankan preprocessing otomatis & commit hasilnya |
| `ci_mlflow.yml` | Push ke `main` | Training model & logging ke DagsHub MLflow |

---

## Secrets GitHub yang Diperlukan

Masuk ke **Settings → Secrets and variables → Actions**, tambahkan:

| Secret | Keterangan |
|---|---|
| `DAGSHUB_USERNAME` | Username DagsHub |
| `DAGSHUB_TOKEN` | Token akses DagsHub |
| `DOCKERHUB_USERNAME` | Username Docker Hub |
| `DOCKERHUB_TOKEN` | Token akses Docker Hub |

---

## Ringkasan Kriteria

| Kriteria | Level | Keterangan |
|---|---|---|
| Kriteria 1 | **Advanced** | Notebook EDA lengkap, script preprocessing otomatis, workflow `preprocess.yml` berjalan dan mengembalikan dataset hasil preprocessing |
