
# Automated Healthy Diet Analytics: End-to-End MLOps Pipeline

  

Proyek ini adalah end-to-end MLOps pipeline yang dibangun dengan pendekatan production-first, menggunakan model **Support Vector Classifier (SVC)** sebagai core engine untuk klasifikasi prediksi.

Seluruh lifecycle machine learning diorkestrasi secara modular—mulai dari data preprocessing, experiment tracking dengan MLflow, hingga automated CI workflow untuk memastikan setiap perubahan model bisa diuji dan direproduksi dengan konsisten.

Di sisi deployment, model diekspos sebagai REST API yang siap menerima traffic real-time, sementara layer observability berjalan paralel melalui sistem monitoring & alerting untuk memantau performa server, latency, serta resource usage secara live.

Hasilnya adalah sebuah sistem ML yang tidak hanya “jalan”, tapi sudah siap scaling: observable, reproducible, dan production-minded.

  

## 📂 Struktur Direktori Proyek

  

Berdasarkan arsitektur repositori, proyek ini dibagi menjadi 4 tahapan utama:

  

```text

SUBMISSION_SML/
│
├── preprocessing/  
│   └── Tahap 1: Data ingestion, cleaning, dan preprocessing awal
│
├── membangun_model/  
│   └── Tahap 2: Eksperimen model, training, evaluasi, dan tracking dengan MLflow
│
├── Workflow-CI/  
│   └── Tahap 3: Automasi training & validasi melalui CI/CD pipeline
│
└── Monitoring dan Logging/  
    └── Tahap 4: Observability sistem (metrics, dashboard, dan alerting real-time)

  

```

  

## 🛠️ 4 Tahapan Utama Pipeline

  

## 1. [Preprocessing](https://github.com/noviardhana/Eksperimen_SML_Tegas-Gagas-Impian) (Pra-pemrosesan Data)

  

Tahap awal untuk memastikan kualitas data sebelum digunakan pada proses pemodelan machine learning.

  

-  **Eksplorasi dan Pembersihan Data:** Melakukan analisis data awal (EDA), menangani missing values, menghapus data duplikat, melakukan deteksi serta penanganan outlier, dan membersihkan data yang tidak relevan.

  

-  **Feature Engineering & Transformasi:** Melakukan encoding fitur kategorikal, normalisasi/standardisasi fitur numerik, serta menghapus fitur yang berpotensi menyebabkan data leakage.

  

-  **Output:** Dataset hasil preprocessing yang siap digunakan pada tahap training model, disimpan dalam file `healthy_diet_calorie_intake_preprocessing.csv`.

  

---

  

## 2. Membangun Model (Model Development & Experiment Tracking)

  

Tahap pengembangan model klasifikasi untuk memprediksi status kesehatan berdasarkan pola konsumsi dan gaya hidup.

  

-  **Baseline Modelling:** Melatih beberapa algoritma machine learning seperti:

    - Support Vector Classifier (SVC)

    - K-Nearest Neighbors (KNN)

    - Random Forest

    - Gradient Boosting

    - XGBoost

  

-  **Hyperparameter Tuning:** Melakukan optimasi parameter menggunakan `GridSearchCV` dan `RandomizedSearchCV` untuk memperoleh konfigurasi model terbaik berdasarkan metrik evaluasi.

  

-  **Evaluasi Model:** Mengukur performa model menggunakan:

    - Accuracy

    - Precision

    - Recall

    - F1-Score

    - Classification Report

    - Confusion Matrix

  

-  **Manajemen Eksperimen dengan MLflow:** Seluruh parameter, metrik evaluasi, model hasil training, dan artefak eksperimen dicatat secara otomatis menggunakan MLflow sehingga setiap eksperimen dapat direproduksi dan dibandingkan dengan mudah.

  

-  **Output Artefak Model:**

    - Folder `model_base/`

    - Model terlatih (`saved_model/`)

    - Classification Report (`.csv`)

    - Confusion Matrix (`.png`)

    - Ranking performa model (`.csv`)

  

---

  

## 3. [Workflow-CI](https://github.com/noviardhana/Workflow-CI) (Continuous Integration & Automated Training)

  

Tahap otomatisasi proses training dan deployment menggunakan GitHub Actions dan MLflow Project.

  

-  **GitHub Actions Automation:** Setiap perubahan kode yang di-*push* ke branch utama akan secara otomatis menjalankan pipeline CI/CD.

  

-  **MLflow Project Execution:** Menggunakan spesifikasi `MLproject` sebagai standar eksekusi sehingga proses training dapat dijalankan secara konsisten di berbagai lingkungan.

  

-  **Automated Model Training:** Workflow akan secara otomatis:

    - Menginstal dependency yang dibutuhkan

    - Menjalankan training model

    - Menghasilkan artefak evaluasi

    - Menyimpan model terbaik

  

-  **Artifact Management:** Hasil training seperti model, laporan evaluasi, dan confusion matrix dikumpulkan dan disimpan sebagai GitHub Artifact.

  

-  **Docker Containerization:** Model yang telah selesai dilatih akan dikemas menjadi Docker Image menggunakan MLflow sehingga siap untuk deployment.

  

-  **Docker Hub Integration:** Docker Image yang berhasil dibuat akan di-*push* secara otomatis ke Docker Hub untuk mempermudah proses distribusi dan deployment.

  



### 4. Monitoring dan Logging (Pemantauan Produksi)

Tahap pemantauan kesehatan sistem dan performa model saat melayani request prediksi secara live.

- **Exporter & Prometheus:**  
  Script `3.prometheus_exporter.py` berfungsi sebagai agent yang mengumpulkan metrik sistem (CPU, RAM, dan latency request API). Data ini kemudian di-scrape oleh Prometheus menggunakan konfigurasi `prometheus.yml`.

- **Grafana Dashboard:**  
  Grafana digunakan untuk visualisasi metrik dari Prometheus dalam bentuk dashboard real-time. Sistem ini juga dapat digunakan untuk monitoring anomali seperti lonjakan CPU atau RAM usage.

---

## 🚀 Panduan Menjalankan Ekosistem Lokal (Playbook)

Jalankan seluruh sistem ini menggunakan beberapa terminal terpisah dan pastikan virtual environment sudah aktif di setiap terminal.

---

### 🎛️ Terminal 1: MLflow Tracking UI

Menjalankan server untuk tracking eksperimen ML.

```bash
.\avicen\Scripts\activate
$env:MLFLOW_TRACKING_URI="http://127.0.0.1:5001"
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

Akses:  
[http://127.0.0.1:5001](http://127.0.0.1:5001)

----------

### ⚡ Terminal 2: Model Serving API

Menjalankan model sebagai REST API untuk inference.

```
mlflow models serve -m runs:/fffb4e0b482a49e69b7f98f9808676d1/model -p 5000 --env-manager=local
```

Endpoint:  
http://127.0.0.1:5000/invocations

----------

### 📡 Terminal 3: Prometheus Exporter

Mengaktifkan pengumpul metrik sistem dan API.

```
python 3.prometheus_exporter.py
```

Endpoint:  
[http://127.0.0.1:8000](http://127.0.0.1:8000)

----------

### 🗄️ Terminal 4: Prometheus Server

Menjalankan time-series database untuk scraping metrik.

```
.\prometheus.exe --config.file=prometheus.yml
```

Akses UI:  
http://127.0.0.1:9090

----------

### 📊 Terminal 5: Grafana Dashboard

Grafana berjalan sebagai service lokal.

Akses:  
[http://localhost:3000](http://localhost:3000)

Login:

-   user: admin
-   pass: admin

Data Source:

-   Prometheus URL: http://127.0.0.1:9090

----------

### 🎯 Terminal 6: Inference Load Test

Menjalankan simulasi request ke model API.

```
python 7.Inference.py
```

_Proyek ini dibangun sebagai bagian dari kepatuhan terhadap standarisasi MLOps lokal yang tangguh dan terotomatisasi._