import os
import time
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report, confusion_matrix

import mlflow
import mlflow.sklearn
import warnings

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

class DietModelPipeline:
    """
    Penjelasan: Pipeline untuk memproses data diet, melakukan hyperparameter tuning pada model 
                Random Forest dan SVC, serta mencatat hasil evaluasinya menggunakan MLflow.
    Input: data_path (string, default: 'healthy_diet_calorie_intake_preprocessing.csv')
    Output: Objek pipeline yang siap dijalankan
    """
    
    def __init__(self, data_path='healthy_diet_calorie_intake_preprocessing.csv'):
        """
        Penjelasan: Inisialisasi parameter dasar dan mematikan autolog MLflow.
        Input: data_path (string)
        Output: None
        """
        self.data_path = data_path
        self.target_names = ['Obese', 'Underweight', 'Overweight', 'Healthy']
        self.tuned_models = {}
        self.best_params = {}
        self.tuned_predictions = {}
        
        mlflow.sklearn.autolog(disable=True)

    def prepare_data(self, kolom_bocor=['BMI', 'Height_cm', 'Weight_kg', 'Health_Status']):
        """
        Penjelasan: Membaca dataset, menghapus kolom yang tidak diperlukan, dan membagi data training/testing.
        Input: kolom_bocor (list of strings)
        Output: None (menyimpan X_train, X_test, y_train, y_test ke dalam atribut class)
        """
        df_model = pd.read_csv(self.data_path)
        kolom_drop = [col for col in kolom_bocor if col in df_model.columns]
        
        X = df_model.drop(columns=kolom_drop)
        y = df_model['Health_Status']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    def setup_credentials(self, dagshub_username, dagshub_repo):
        """
        Penjelasan: Memuat token dari file .env dan menyiapkan URI untuk DagsHub.
        Input: dagshub_username (string), dagshub_repo (string)
        Output: None (menyimpan dagshub_uri ke dalam atribut class)
        """
        load_dotenv() 
        os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_username
        
        token = os.getenv('MLFLOW_TRACKING_PASSWORD')
        if not token:
            print("Error: Token tidak ditemukan di .env!")
            return
            
        os.environ['MLFLOW_TRACKING_PASSWORD'] = token
        self.dagshub_uri = f"https://dagshub.com/{dagshub_username}/{dagshub_repo}.mlflow"

    def tuning_models(self):
        """
        Penjelasan: Melakukan hyperparameter tuning untuk model Random Forest dan SVC, 
                    serta mencatat waktu proses masing-masing model.
        Input: None
        Output: None (menyimpan model terbaik dan parameter terbaik ke dalam atribut class)
        """
        # Random Forest Tuning
        print("mulai tunning Random Forest...")
        start_rf = time.time()
        
        rf_dist = {
            'n_estimators': [100, 200, 300, 400, 500], 
            'criterion': ['gini', 'entropy'],
            'max_depth': [10, 20, 30, 40, None], 
            'min_samples_split': [2, 5, 10], 
            'min_samples_leaf': [1, 2, 4], 
            'max_features': ['sqrt', 'log2', None], 
            'bootstrap': [True, False], 
            'class_weight': ['balanced', 'balanced_subsample'] 
        }
        
        rf_search = RandomizedSearchCV(
            RandomForestClassifier(random_state=42), 
            param_distributions=rf_dist, 
            n_iter=10,  
            cv=3, 
            scoring='f1_weighted', 
            n_jobs=-1, 
            random_state=42
        )
        rf_search.fit(self.X_train, self.y_train)
        
        self.tuned_models['Tuned Random Forest'] = rf_search.best_estimator_
        self.best_params['Tuned Random Forest'] = rf_search.best_params_
        
        end_rf = time.time()
        rf_mins, rf_secs = divmod(end_rf - start_rf, 60)
        print(f"selesai tunning Random Forest: waktu {int(rf_mins)} menit {int(rf_secs)} detik")

        # SVC Tuning
        print("mulai tunning SVC...")
        start_svc = time.time()
        
        svc_grid_params = [
            {
                'kernel': ['rbf'], 
                'C': [0.1, 1, 10, 50], 
                'gamma': ['scale', 'auto', 0.1, 0.01]
            },
            {
                'kernel': ['linear'], 
                'C': [0.1, 1, 10]
            },
            {
                'kernel': ['poly'], 
                'C': [0.1, 1, 10], 
                'degree': [2, 3], 
                'gamma': ['scale']
            }
        ]
        
        svc_search = GridSearchCV(
            SVC(probability=True, class_weight='balanced', random_state=42), 
            param_grid=svc_grid_params, 
            cv=3, 
            scoring='f1_weighted', 
            n_jobs=-1 
        )
        svc_search.fit(self.X_train, self.y_train)

        self.tuned_models['Tuned SVC'] = svc_search.best_estimator_
        self.best_params['Tuned SVC'] = svc_search.best_params_
        
        end_svc = time.time()
        svc_mins, svc_secs = divmod(end_svc - start_svc, 60)
        print(f"selesai tunning SVC: waktu {int(svc_mins)} menit {int(svc_secs)} detik")

    def evaluate_and_log_mlflow(self):
        """
        Penjelasan: Mengevaluasi model yang telah dituning, membuat artefak (CSV & PNG), 
                    dan mencatat parameter serta metrik ke MLflow (Lokal dan DagsHub).
        Input: None
        Output: None (menyimpan file artefak secara lokal dan mengirim data ke MLflow)
        """
        print("\n=== Evaluasi & MLflow Dual Logging (Lokal & DagsHub) ===\n")
        
        for name, model in self.tuned_models.items():
            print(f"\nModel: {name}")
            print(f"Parameter Terbaik: {self.best_params[name]}")

            # Prediksi & Hitung Metrik
            y_pred = model.predict(self.X_test)
            self.tuned_predictions[name] = y_pred
            
            accuracy = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(self.y_test, y_pred, average='weighted')
            
            metrics_dict = {
                "accuracy": accuracy,
                "f1_score_weighted": f1,
                "precision_weighted": precision,
                "recall_weighted": recall
            }

            print(f"F1-Score: {f1:.4f} | Accuracy: {accuracy:.4f}")
            
            # Buat Artefak
            safe_name = name.replace(" ", "_") 
            report_path = f"hasil_evaluasi_{safe_name}_tunning.csv"
            plot_path = f"hasil_evaluasi_{safe_name}_tunning.png"
            
            report_dict = classification_report(self.y_test, y_pred, target_names=self.target_names, output_dict=True)
            pd.DataFrame(report_dict).transpose().to_csv(report_path, index=True)
            
            cm = confusion_matrix(self.y_test, y_pred)
            plt.figure(figsize=(6,4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=self.target_names, yticklabels=self.target_names)
            plt.title(f"Confusion Matrix: {name}")
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300)
            plt.close()

            # Logging Lokal
            mlflow.set_tracking_uri("http://127.0.0.1:5000/") 
            mlflow.set_experiment("Diet_Health_Status_Skilled")
            
            with mlflow.start_run(run_name=f"Local_{name}"):
                mlflow.log_params(self.best_params[name])
                mlflow.log_metrics(metrics_dict)
                mlflow.log_artifact(report_path)
                mlflow.log_artifact(plot_path)
                mlflow.sklearn.log_model(model, "model")
            print(f"{name} tercatat di LOKAL.")

            # Logging DagsHub
            mlflow.set_tracking_uri(self.dagshub_uri) 
            mlflow.set_experiment("Diet_Health_Status_Advance")
            
            with mlflow.start_run(run_name=f"DagsHub_{name}"):
                mlflow.log_params(self.best_params[name])
                mlflow.log_metrics(metrics_dict)
                mlflow.log_artifact(report_path) 
                mlflow.log_artifact(plot_path)   
                mlflow.sklearn.log_model(model, "model")
            print(f"{name} tercatat di DAGSHUB.\n")

if __name__ == '__main__':
    pipeline = DietModelPipeline()
    
    DAGSHUB_USERNAME = "noviardhana" 
    DAGSHUB_REPO = "sml_noviardhana"
    
    pipeline.setup_credentials(DAGSHUB_USERNAME, DAGSHUB_REPO)
    pipeline.prepare_data() 
    pipeline.tuning_models()
    pipeline.evaluate_and_log_mlflow()