import requests
import time
import json
import random
import logging


logging.basicConfig(
    filename="api_model_logs.log", 
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Konfigurasi Global 
URL = "http://127.0.0.1:5005/invocations"
HEADERS = {"Content-Type": "application/json"}

# Kolom didefinisikan sekali di luar loop
COLUMNS = [
    "Age", "Activity_Level", "Daily_Calorie_Requirement", 
    "Daily_Calorie_Consumed", "Protein_Intake_g", "Carbohydrate_Intake_g", 
    "Fat_Intake_g", "Water_Intake_Liters", "Gender_Male", 
    "Gender_Other", "Diet_Type_High Protein", "Diet_Type_Keto", 
    "Diet_Type_Mediterranean", "Diet_Type_Vegan", "Diet_Type_Vegetarian"
]

# Kumpulan sampel data
BASE_DATA = [
    [0.063, 3, 1.218, 0.231, 1.036, -2.197, 2.253, 0.847, 1, 0, 0, 1, 0, 0, 0],
    [-1.700, 0, -0.819, -0.749, -0.547, 0.132, -1.179, -1.235, 0, 0, 0, 0, 0, 1, 0],
    [1.055, 1, -0.593, 0.087, 1.713, -0.079, -0.952, -0.492, 0, 0, 1, 0, 0, 0, 0],
    [-1.480, 2, 0.071, -0.661, -0.898, 0.076, -0.629, 0.252, 0, 0, 0, 0, 0, 0, 0],
    [-1.039, 0, -0.303, -0.929, -1.522, -0.043, -0.476, -0.789, 1, 0, 0, 0, 0, 0, 0]
]

def generate_payload():
    """
    Penjelasan: Memilih satu baris data secara acak dan membungkusnya dalam format JSON.
    """
    data_terpilih = random.choice(BASE_DATA)
    payload = {
        "dataframe_split": {
            "columns": COLUMNS,
            "data": [data_terpilih]
        }
    }
    return payload

def run_simulation(interval_detik=1):
    """
    Penjelasan: Menjalankan loop tanpa batas untuk menembak API model secara berulang.
    """
    print(f"Mulai menembak API di {URL} ... (Tekan Ctrl+C untuk stop)\n")
    
    while True:
        payload = generate_payload()
        
        # Mulai mencatat waktu eksekusi (seperti di kode pertama)
        start_time = time.time()
        
        try:
            # Mengirim request POST ke server model Docker
            response = requests.post(URL, headers=HEADERS, data=json.dumps(payload))
            
            # Hitung response time
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                # Coba parse ke JSON, jika gagal biarkan sebagai text biasa
                try:
                    prediction = response.json()
                except json.JSONDecodeError:
                    prediction = response.text.strip()
                
                # Logging hasil request ke file log
                logging.info(f"Request: {payload}, Response: {prediction}, Response Time: {response_time:.4f} sec")
                
                # Print hasil ke terminal sesuai format yang lu mau
                print(f"Prediction: {prediction}")
                print(f"Response Time: {response_time:.4f} sec\n")
                
            else:
                error_msg = f"Error {response.status_code}: {response.text}"
                logging.error(error_msg)
                print(error_msg + "\n")
                
        except requests.exceptions.ConnectionError:
            error_msg = "Gagal koneksi! Pastikan container Docker masih 'Up'."
            logging.error(f"Exception: {error_msg}")
            print(f"Exception: {error_msg}\n")
            
        except Exception as e:
            logging.error(f"Exception: {str(e)}")
            print(f"Exception: {str(e)}\n")
            
        # Jeda waktu antar request
        time.sleep(interval_detik)

if __name__ == "__main__":
    run_simulation()