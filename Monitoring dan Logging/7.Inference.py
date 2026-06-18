import requests
import time
import json
import random
import logging

logging.basicConfig(
    filename="api_model_logs.log",
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

URL = "http://127.0.0.1:5000/invocations"
HEADERS = {"Content-Type": "application/json"}

COLUMNS = [
    "Age", "Activity_Level", "Daily_Calorie_Requirement",
    "Daily_Calorie_Consumed", "Protein_Intake_g", "Carbohydrate_Intake_g",
    "Fat_Intake_g", "Water_Intake_Liters", "Gender_Male",
    "Gender_Other", "Diet_Type_High Protein", "Diet_Type_Keto",
    "Diet_Type_Mediterranean", "Diet_Type_Vegan", "Diet_Type_Vegetarian"
]

BASE_DATA = [
    [0.063, 3, 1.218, 0.231, 1.036, -2.197, 2.253, 0.847, 1, 0, 0, 1, 0, 0, 0],
    [-1.700, 0, -0.819, -0.749, -0.547, 0.132, -1.179, -1.235, 0, 0, 0, 0, 0, 1, 0],
    [1.055, 1, -0.593, 0.087, 1.713, -0.079, -0.952, -0.492, 0, 0, 1, 0, 0, 0, 0],
    [-1.480, 2, 0.071, -0.661, -0.898, 0.076, -0.629, 0.252, 0, 0, 0, 0, 0, 0, 0],
    [-1.039, 0, -0.303, -0.929, -1.522, -0.043, -0.476, -0.789, 1, 0, 0, 0, 0, 0, 0]
]

def generate_payload():
    return {
        "dataframe_split": {
            "columns": COLUMNS,
            "data": [random.choice(BASE_DATA)]
        }
    }

def run_simulation(interval=1, max_requests=500):
    print(f"Starting inference on {URL}\n")

    for i in range(max_requests):

        payload = generate_payload()
        start_time = time.time()

        try:
            response = requests.post(URL, json=payload)
            response_time = time.time() - start_time

            response.raise_for_status()
            prediction = response.json()

            logging.info(
                f"Request {i+1} | Payload: {payload} | "
                f"Response: {prediction} | Time: {response_time:.4f}s"
            )

            print(f"[{i+1}] Prediction: {prediction} | {response_time:.4f}s")

        except Exception as e:
            logging.error(f"Error: {str(e)}")
            print(f"Error: {str(e)}")

        time.sleep(interval)

if __name__ == "__main__":
    run_simulation()