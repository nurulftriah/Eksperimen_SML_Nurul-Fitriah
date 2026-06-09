import json
import os

notebook_data = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Eksperimen Supervised Machine Learning (SML) - Nurul-Fitriah\n",
    "\n",
    "## 1. Perkenalan Dataset\n",
    "Tahap pertama, Anda harus mencari dan menggunakan dataset dengan ketentuan sebagai berikut:\n",
    "\n",
    "Sumber Dataset:\n",
    "Dataset dapat diperoleh dari berbagai sumber, seperti public repositories (Kaggle, UCI ML Repository, Open Data) atau data primer yang Anda kumpulkan sendiri.\n",
    "\n",
    "---\n",
    "### Informasi Dataset Credit Scoring (Dataset Eksperimen):\n",
    "Eksperimen ini menggunakan dataset **Credit Scoring**. Dataset ini dikembangkan secara sintetis untuk mensimulasikan data pengajuan pinjaman bank. Dataset ini memiliki beberapa masalah umum data dunia nyata seperti data kosong (missing values), outlier pada variabel umur dan pendapatan, baris duplikat, dan variabel kategorikal.\n",
    "\n",
    "**Variabel/Fitur pada Dataset:**\n",
    "- `customer_id`: ID unik nasabah (identifikasi)\n",
    "- `age`: Umur nasabah\n",
    "- `annual_income`: Pendapatan tahunan nasabah\n",
    "- `credit_score`: Skor kredit nasabah (berkisar antara 300 hingga 850)\n",
    "- `loan_amount`: Jumlah pinjaman yang diajukan\n",
    "- `marital_status`: Status pernikahan (Single, Married, Divorced)\n",
    "- `education_level`: Tingkat pendidikan (High School, Bachelor, Master, PhD)\n",
    "- `default`: Target variabel (1 = Gagal bayar/default, 0 = Lancar)\n",
    "\n",
    "Tujuan utama eksperimen ini adalah memuat data, melakukan Exploratory Data Analysis (EDA), dan membersihkan/mempersiapkan data (preprocessing) agar siap digunakan dalam pemodelan Machine Learning."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Import Library\n",
    "Pada tahap ini, Anda perlu mengimpor beberapa pustaka (library) Python yang dibutuhkan untuk analisis data dan pembangunan model machine learning atau deep learning."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "import os"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Memuat Dataset\n",
    "Pada tahap ini, Anda perlu memuat dataset ke dalam notebook. Jika dataset dalam format CSV, Anda bisa menggunakan pustaka pandas untuk membacanya. Pastikan untuk mengecek beberapa baris awal dataset untuk memahami strukturnya dan memastikan data telah dimuat dengan benar.\n",
    "\n",
    "Jika dataset berada di Google Drive, pastikan Anda menghubungkan Google Drive ke Colab terlebih dahulu. Setelah dataset berhasil dimuat, langkah berikutnya adalah memeriksa kesesuaian data dan siap untuk dianalisis lebih lanjut.\n",
    "\n",
    "Jika dataset berupa unstructured data, silakan sesuaikan dengan format seperti kelas Machine Learning Pengembangan atau Machine Learning Terapan."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Memuat dataset mentah\n",
    "raw_data_path = '../namadataset_raw/credit_scoring_raw.csv'\n",
    "df = pd.read_csv(raw_data_path)\n",
    "print(f\"Ukuran Dataset: {df.shape[0]} baris, {df.shape[1]} kolom\")\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Exploratory Data Analysis (EDA)\n",
    "Pada tahap ini, Anda akan melakukan Exploratory Data Analysis (EDA) untuk memahami karakteristik dataset.\n",
    "\n",
    "Tujuan dari EDA adalah untuk memperoleh wawasan awal yang mendalam mengenai data dan menentukan langkah selanjutnya dalam analisis atau pemodelan."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Informasi struktur dataset\n",
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Statistik deskriptif\n",
    "df.describe(include='all')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Memeriksa jumlah data kosong (missing values)\n",
    "print(\"Jumlah data kosong per kolom:\")\n",
    "print(df.isnull().sum())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Memeriksa jumlah baris duplikat\n",
    "print(f\"Jumlah baris duplikat: {df.duplicated().sum()}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualisasi Distribusi Target (Default)\n",
    "plt.figure(figsize=(6, 4))\n",
    "sns.countplot(x='default', data=df, palette='Set2')\n",
    "plt.title('Distribusi Gagal Bayar (Default) Nasabah')\n",
    "plt.xlabel('Default (0 = Lancar, 1 = Gagal Bayar)')\n",
    "plt.ylabel('Jumlah Nasabah')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualisasi Distribusi Umur dan Cek Pencilan\n",
    "plt.figure(figsize=(12, 5))\n",
    "plt.subplot(1, 2, 1)\n",
    "sns.histplot(df['age'].dropna(), bins=30, kde=True, color='skyblue')\n",
    "plt.title('Distribusi Umur Nasabah')\n",
    "plt.subplot(1, 2, 2)\n",
    "sns.boxplot(y=df['age'], color='lightgreen')\n",
    "plt.title('Deteksi Pencilan Umur (Outliers)')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualisasi Pendapatan Tahunan dan Cek Pencilan\n",
    "plt.figure(figsize=(12, 5))\n",
    "plt.subplot(1, 2, 1)\n",
    "sns.histplot(df['annual_income'].dropna(), bins=30, kde=True, color='salmon')\n",
    "plt.title('Distribusi Pendapatan Nasabah')\n",
    "plt.subplot(1, 2, 2)\n",
    "sns.boxplot(y=df['annual_income'], color='orange')\n",
    "plt.title('Deteksi Pencilan Pendapatan')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Data Preprocessing\n",
    "Pada tahap ini, data preprocessing adalah langkah penting untuk memastikan kualitas data sebelum digunakan dalam model machine learning.\n",
    "\n",
    "Jika Anda menggunakan data teks, data mentah sering kali mengandung nilai kosong, duplikasi, atau rentang nilai yang tidak konsisten, yang dapat memengaruhi kinerja model. Oleh karena itu, proses ini bertujuan untuk membersihkan dan mempersiapkan data agar analisis berjalan optimal.\n",
    "\n",
    "Berikut adalah tahapan-tahapan yang bisa dilakukan, tetapi tidak terbatas pada:\n",
    "- Menghapus atau Menangani Data Kosong (Missing Values)\n",
    "- Menghapus Data Duplikat\n",
    "- Normalisasi atau Standarisasi Fitur\n",
    "- Deteksi dan Penanganan Outlier\n",
    "- Encoding Data Categorical\n",
    "- Binning (Pengelompokan Data)\n",
    "\n",
    "Cukup sesuaikan dengan karakteristik data yang kamu gunakan yah. Khususnya ketika kami menggunakan data tidak terstruktur."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 1: Menghapus Data Duplikat\n",
    "df_cleaned = df.drop_duplicates().copy()\n",
    "print(f\"Ukuran dataset setelah menghapus duplikat: {df_cleaned.shape[0]} baris\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 2: Menangani Data Kosong (Missing Values)\n",
    "# Imputasi fitur numerik dengan nilai median\n",
    "df_cleaned['age'] = df_cleaned['age'].fillna(df_cleaned['age'].median())\n",
    "df_cleaned['annual_income'] = df_cleaned['annual_income'].fillna(df_cleaned['annual_income'].median())\n",
    "\n",
    "# Imputasi fitur kategorikal dengan nilai modus\n",
    "df_cleaned['marital_status'] = df_cleaned['marital_status'].fillna(df_cleaned['marital_status'].mode()[0])\n",
    "\n",
    "print(\"Jumlah data kosong setelah imputasi:\")\n",
    "print(df_cleaned.isnull().sum())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 3: Deteksi dan Penanganan Outlier menggunakan Metode IQR\n",
    "# Batasi outlier Umur\n",
    "q1_age = df_cleaned['age'].quantile(0.25)\n",
    "q3_age = df_cleaned['age'].quantile(0.75)\n",
    "iqr_age = q3_age - q1_age\n",
    "lower_age = max(18, q1_age - 1.5 * iqr_age)  # Umur minimal 18 tahun\n",
    "upper_age = min(100, q3_age + 1.5 * iqr_age) # Batasi maksimal 100 tahun\n",
    "df_cleaned['age'] = df_cleaned['age'].clip(lower_age, upper_age)\n",
    "\n",
    "# Batasi outlier Pendapatan Tahunan\n",
    "q1_inc = df_cleaned['annual_income'].quantile(0.25)\n",
    "q3_inc = df_cleaned['annual_income'].quantile(0.75)\n",
    "iqr_inc = q3_inc - q1_inc\n",
    "lower_inc = q1_inc - 1.5 * iqr_inc\n",
    "upper_inc = q3_inc + 1.5 * iqr_inc\n",
    "df_cleaned['annual_income'] = df_cleaned['annual_income'].clip(lower_inc, upper_inc)\n",
    "\n",
    "# Visualisasi setelah pembersihan outlier\n",
    "plt.figure(figsize=(12, 5))\n",
    "plt.subplot(1, 2, 1)\n",
    "sns.boxplot(y=df_cleaned['age'], color='lightgreen')\n",
    "plt.title('Umur setelah Penanganan Outlier')\n",
    "plt.subplot(1, 2, 2)\n",
    "sns.boxplot(y=df_cleaned['annual_income'], color='orange')\n",
    "plt.title('Pendapatan setelah Penanganan Outlier')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 4: Binning (Pengelompokan Data Umur)\n",
    "# Mengelompokkan umur menjadi Young (18-35), Middle-Aged (36-55), Senior (56+)\n",
    "bins = [17, 35, 55, np.inf]\n",
    "labels = ['Young', 'Middle-Aged', 'Senior']\n",
    "df_cleaned['age_group'] = pd.cut(df_cleaned['age'], bins=bins, labels=labels)\n",
    "print(\"Distribusi kelompok umur:\")\n",
    "print(df_cleaned['age_group'].value_counts())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 5: Encoding Data Kategorikal (One-hot encoding)\n",
    "# Drop customer_id karena tidak memiliki nilai prediktif\n",
    "df_preprocessed = df_cleaned.drop(columns=['customer_id'])\n",
    "categorical_cols = ['marital_status', 'education_level', 'age_group']\n",
    "df_preprocessed = pd.get_dummies(df_preprocessed, columns=categorical_cols, drop_first=True)\n",
    "\n",
    "# Mengonversi kolom boolean hasil dummy ke integer\n",
    "bool_cols = df_preprocessed.select_dtypes(include=['bool']).columns\n",
    "df_preprocessed[bool_cols] = df_preprocessed[bool_cols].astype(int)\n",
    "\n",
    "print(\"Tampilan data setelah One-hot Encoding:\")\n",
    "df_preprocessed.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 6: Standarisasi Fitur Numerik\n",
    "numerical_cols = ['age', 'annual_income', 'credit_score', 'loan_amount']\n",
    "scaler = StandardScaler()\n",
    "df_preprocessed[numerical_cols] = scaler.fit_transform(df_preprocessed[numerical_cols])\n",
    "\n",
    "print(\"Tampilan data final hasil standarisasi:\")\n",
    "df_preprocessed.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Menyimpan dataset hasil preprocessing\n",
    "output_dir = 'namadataset_preprocessing'\n",
    "os.makedirs(output_dir, exist_ok=True)\n",
    "output_path = os.path.join(output_dir, 'credit_scoring_preprocessing.csv')\n",
    "df_preprocessed.to_csv(output_path, index=False)\n",
    "print(f\"Preprocessed dataset berhasil disimpan di: {output_path}\")\n",
    "print(f\"Dimensi data final: {df_preprocessed.shape[0]} baris, {df_preprocessed.shape[1]} kolom\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

os.makedirs('preprocessing', exist_ok=True)
with open('preprocessing/Eksperimen_Nurul-Fitriah.ipynb', 'w') as f:
    json.dump(notebook_data, f, indent=1)

print("Jupyter Notebook created successfully at preprocessing/Eksperimen_Nurul-Fitriah.ipynb")
