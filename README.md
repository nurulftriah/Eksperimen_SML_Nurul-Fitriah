# Eksperimen Supervised Machine Learning - Nurul-Fitriah

Proyek ini memenuhi Kriteria 1, 2, 3, dan 4 pada tingkat **Advanced** untuk submission Dicoding SML. Fokus pada analisis kelayakan kredit (Credit Scoring) menggunakan Random Forest, logging otomatis dengan MLflow & DagsHub, CI/CD dengan GitHub Actions, serta monitoring dengan FastAPI, Prometheus, dan Grafana.

---

## 1. Struktur Repositori

```
Eksperimen_SML_Nurul-Fitriah/
├── .workflow/
│   ├── preprocess.yml              # Workflow otomatis preprocessing data
│   └── ci_mlflow.yml               # Workflow CI training, MLflow logging, & Docker push
├── namadataset_raw/
│   └── credit_scoring_raw.csv      # Dataset mentah
├── preprocessing/
│   ├── Eksperimen_Nurul-Fitriah.ipynb  # Notebook EDA & Preprocessing
│   ├── automate_Nurul-Fitriah.py       # Script preprocessing otomatis
│   └── namadataset_preprocessing/
│       └── credit_scoring_preprocessing.csv  # Dataset hasil preprocessing
├── monitoring/
│   ├── prometheus.yml              # Konfigurasi scraping Prometheus
│   ├── grafana_datasources.yml     # Provisioning datasource Grafana
│   ├── grafana_dashboards.yml      # Provisioning dashboard Grafana
│   ├── dashboard_template.json     # Template dashboard Grafana
│   └── docker-compose.yml          # Docker Compose (API + Prometheus + Grafana)
├── app/
│   ├── app.py                      # FastAPI + Prometheus exporter
│   ├── Dockerfile                  # Dockerfile untuk FastAPI
│   └── requirements.txt            # Dependensi API
├── conda.yaml                      # Environment Anaconda untuk MLflow
├── generate_raw_data.py            # Script generate dataset mentah
├── MLproject                       # Konfigurasi MLflow Project
├── modelling_tuning.py             # Script training, tuning, & logging MLflow
├── requirements.txt                # Dependensi training
└── README.md
```

---

## 2. Cara Menjalankan Lokal

### Prasyarat
- Python 3.12.7
- Docker Desktop

### A. Generate Data & Preprocessing (Kriteria 1)
```bash
# 1. Generate dataset mentah
python generate_raw_data.py

# 2. Jalankan preprocessing otomatis
python preprocessing/automate_Nurul-Fitriah.py
```
Output tersimpan di `preprocessing/namadataset_preprocessing/credit_scoring_preprocessing.csv`

### B. Training & MLflow Logging (Kriteria 2)
```bash
# 1. Install dependensi
pip install -r requirements.txt

# 2. Jalankan MLflow server (terminal terpisah)
mlflow server --port 5000

# 3. Jalankan training
python modelling_tuning.py
```
Buka `http://127.0.0.1:5000` untuk melihat hasil di MLflow UI lokal.

### C. Serving & Monitoring (Kriteria 4)
```bash
# Jalankan dari folder monitoring/
cd monitoring
docker-compose up --build
```

| Service | URL |
|---|---|
| API FastAPI | http://localhost:5005 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

Login Grafana: `admin` / `admin`

### D. Kirim Request Prediksi
```bash
curl -X POST "http://localhost:5005/predict" \
  -H "Content-Type: application/json" \
  -d "{\"age\":0.5,\"annual_income\":1.2,\"credit_score\":0.8,\"loan_amount\":-0.2,\"marital_status_Married\":1,\"marital_status_Single\":0,\"education_level_High_School\":0,\"education_level_Master\":1,\"education_level_PhD\":0,\"age_group_Middle_Aged\":1,\"age_group_Senior\":0}"
```

---

## 3. Integrasi DagsHub & CI/CD (Kriteria 2 & 3)

### Secrets GitHub yang Diperlukan
Masuk ke **Settings → Secrets and variables → Actions**, tambahkan:

| Secret | Keterangan |
|---|---|
| `DAGSHUB_USERNAME` | Username DagsHub |
| `DAGSHUB_TOKEN` | Token akses DagsHub |
| `DOCKERHUB_USERNAME` | Username Docker Hub |
| `DOCKERHUB_TOKEN` | Token akses Docker Hub |

Setiap `git push` ke branch `main` akan otomatis:
1. Melatih model dan logging ke **DagsHub MLflow**
2. Upload artifact ke GitHub Actions
3. Build & push Docker image ke Docker Hub

---

## 4. Ringkasan Kriteria Terpenuhi

| Kriteria | Level | Keterangan |
|---|---|---|
| Kriteria 1 | Advanced | Notebook EDA, script preprocessing otomatis, workflow `preprocess.yml` |
| Kriteria 2 | Advanced | `modelling_tuning.py` dengan GridSearchCV, logging 5 artifact ke DagsHub MLflow |
| Kriteria 3 | Advanced | CI/CD lengkap: training → MLflow → upload artifact → build & push Docker |
| Kriteria 4 | Advanced | FastAPI + Prometheus (CPU, RAM, latency, throughput) + Grafana dashboard + alerting |