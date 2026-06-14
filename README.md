# Automated Healthy Diet Analytics: End-to-End MLOps Pipeline

Proyek ini mengimplementasikan *pipeline* MLOps end-to-end untuk klasifikasi dan prediksi berbasis Machine Learning (menggunakan model **Support Vector Classifier - SVC**). Arsitektur sistem ini mencakup pemrosesan data, manajemen eksperimen model, integrasi otomatis (CI Workflow), hingga pemantauan metrik server (*monitoring & alerting*) secara *real-time*.

## 📂 Struktur Direktori Proyek

Berdasarkan arsitektur repositori, proyek ini dibagi menjadi 4 tahapan utama:

```text
SUBMISSION_SML/
├── preprocessing/             # Tahap 1: Ingesti & Pembersihan Data
├── membangun_model/           # Tahap 2: Eksperimen & Registrasi Model (MLflow)
├── Workflow-CI/               # Tahap 3: Otomatisasi & Pengujian Kontinu
└── Monitoring dan Logging/    # Tahap 4: Sensor Metrik, Dasbor Visual & Alerting

```

## 🛠️ 4 Tahapan Utama Pipeline

### 1. [Preprocessing](https://github.com/noviardhana/Eksperimen_SML_Tegas-Gagas-Impian) (Pra-pemrosesan Data)

Tahap awal untuk memastikan kualitas data sebelum masuk ke dalam proses _training_.

-   **Eksplorasi & Pembersihan:** Menangani _missing values_, duplikasi, serta melakukan rekayasa fitur (_feature engineering_) pada dataset diet sehat (`healthy_diet_calorie_intake.csv`).
    
-   **Output:** Dataset bersih yang siap digunakan untuk pemodelan (`healthy_diet_calorie_intake_preprocessing.csv`).
    

### 2. Membangun Model (Model Development & Tracking)

Tahap eksperimen untuk mencari performa model terbaik menggunakan arsitektur **MLflow**.

-   **Eksperimen & Tuning:** Kode `modelling.py` dan `modelling_tunning.py` digunakan untuk melatih model awal dan melakukan _hyperparameter tuning_ pada algoritma Random Forest dan SVC.
    
-   **Manajemen Artefak:** Seluruh performa metrik, parameter, dan file model fisik disimpan secara terstruktur di dalam folder `mlruns/` dan `mlartifacts/`.
    

### 3. [Workflow-CI](https://github.com/noviardhana/Workflow-CI) (Continuous Integration)

Sistem otomatisasi untuk menjamin bahwa setiap perubahan kode aman dan memenuhi standar sebelum dideploy.

-   **Otomatisasi GitHub Actions:** Folder `.github/workflows` menyimpan konfigurasi CI untuk menjalankan pengujian otomatis (_automated testing_) dan validasi kode setiap kali ada aktivitas _push_ ke repositori.
    
-   **MLProject:** Menggunakan spesifikasi `MLProject` untuk standarisasi lingkungan eksekusi agar _reproducible_.
    

### 4. Monitoring dan Logging (Pemantauan Produksi)

Tahap pemantauan kesehatan infrastruktur dan performa model saat melayani _request_ prediksi secara _live_.

-   **Exporter & Prometheus:** File `3.prometheus_exporter.py` bertindak sebagai agen sensor yang menyadap penggunaan CPU, RAM, serta trafik API, lalu dikumpulkan oleh `prometheus.exe` melalui konfigurasi `2.prometheus.yml`.
    
-   **Visualisasi Grafana:** Dasbor visual interaktif yang menyedot data dari Prometheus untuk memantau metrik secara grafikal dan mengirim notifikasi peringatan jika terjadi anomali sistem (seperti CPU > 85% atau RAM > 90%).
    

## 🚀 Panduan Menjalankan Ekosistem Lokal (Playbook)

Untuk mereplikasi atau menjalankan seluruh ekosistem ini di komputer lokal Anda, buka beberapa jendela terminal terpisah dan jalankan instruksi di bawah ini secara berurutan.

> ⚠️ **Penting:** Pastikan _Virtual Environment_ Python Anda (misal: `avicen`) sudah diaktifkan di setiap terminal baru yang Anda buka.

### 🎛️ Terminal 1: Manajemen Eksperimen (MLflow UI)

Mengaktifkan antarmuka pelacakan metrik eksperimen.

PowerShell

```
# Buka batasan keamanan berkas lokal Windows
$env:MLFLOW_ALLOW_FILE_STORE="true"

# Jalankan server MLflow UI pada port kustom
python -m mlflow ui --port 5001

```

_Akses UI:_ Buka `http://localhost:5001` di browser Anda.

### ⚡ Terminal 2: Melayani Model (Model Serving API)

Mengubah artefak model SVC statis menjadi API hidup menggunakan server Uvicorn bawaan MLflow.

Bash

```
python -m mlflow models serve -m "mlartifacts/431831200810355098/451b30357b8f48feadea09c639ecb968/artifacts/model" -p 5005 --env-manager=local

```

_(Biarkan terminal ini tetap terbuka untuk melayani request prediksi)._

### 📡 Terminal 3: Mengaktifkan Agen Sensor (Metrics Exporter)

Menjalankan script Python khusus yang mengumpulkan data hardware dan request API.

Bash

```
python .\3.prometheus_exporter.py

```

_Bahan Mentah Metrik:_ Dapat diintip secara raw pada alamat `http://localhost:8000`.

### 🗄️ Terminal 4: Database Pemantau (Prometheus Engine)

Menyalakan _time-series database_ Prometheus untuk menyedot (_scraping_) data dari _exporter_.

Bash

```
.\prometheus.exe --config.file=prometheus.yml

```

_Akses UI:_ Buka `http://localhost:9090` untuk melihat target monitoring.

### 📊 Tahap Akhir: Command Center (Grafana Dashboard)

Grafana secara otomatis berjalan sebagai _Windows Service_ di latar belakang jika diinstal menggunakan berkas `.msi`.

1.  Langsung buka _browser_ Anda dan arahkan ke alamat: **`http://localhost:3000`**
    
2.  Masuk menggunakan kredensial default (`admin` / `admin`).
    
3.  Hubungkan ke Prometheus (_Data Source URL:_ `http://localhost:9090`).
    
4.  Gunakan filter query PromQL berikut pada dashboard atau alert Anda:
    
    Code snippet
    
    ```
    {instance="127.0.0.1:8000"}
    
    ```
    

### 🎯 Langkah Pengujian: Simulasi Trafik Berkelanjutan

Untuk melihat pergerakan grafik pada dasbor Grafana atau memicu email peringatan (_alert notification_), buka **Terminal 5** dan jalankan pengujian prediksi:

Bash

```
python .\7.Inference.py

```

_Proyek ini dibangun sebagai bagian dari kepatuhan terhadap standarisasi MLOps lokal yang tangguh dan terotomatisasi._
