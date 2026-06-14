import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import warnings
import mlflow
import mlflow.sklearn

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

class DietModelPipeline:
    """
    Penjelasan: Pipeline dasar untuk melatih dan mengevaluasi model machine learning.
    Input: data_path (string)
    Output: Objek pipeline yang siap dieksekusi
    """
    def __init__(self, data_path='healthy_diet_calorie_intake_preprocessing.csv'):
        """
        Penjelasan: Inisialisasi variabel, model dasar, dan mengaktifkan autolog MLflow.
        Input: data_path (string)
        Output: None
        """
        self.data_path = data_path
        self.target_names = ['Obese', 'Underweight', 'Overweight', 'Healthy']
        self.predictions = {}
        
        self.models = {
            'Logistic Regression': LogisticRegression(penalty='l2', C=0.1, solver='lbfgs', max_iter=2000, tol=1e-4, class_weight='balanced', n_jobs=-1, random_state=42),
            'SVC': SVC(C=1.0, kernel='rbf', gamma='scale', probability=True, decision_function_shape='ovr', tol=1e-3, class_weight='balanced', random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=300, criterion='gini', max_depth=15, min_samples_split=5, min_samples_leaf=2, max_features='sqrt', bootstrap=True, oob_score=True, class_weight='balanced_subsample', n_jobs=-1, random_state=42)
        }

        mlflow.sklearn.autolog()
        mlflow.set_tracking_uri("http://127.0.0.1:5000/") 
        mlflow.set_experiment("Diet_Health_Status_Basic")

    def prepare_data(self, kolom_bocor=['BMI', 'Height_cm', 'Weight_kg', 'Health_Status']):
        """
        Penjelasan: Memuat dataset, membuang kolom yang tidak relevan, dan membagi data.
        Input: kolom_bocor (list of string)
        Output: None
        """
        df_model = pd.read_csv(self.data_path)
        kolom_drop = [col for col in kolom_bocor if col in df_model.columns]
        
        X = df_model.drop(columns=kolom_drop)
        y = df_model['Health_Status']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    def train_and_evaluate(self, report_path='laporan_klasifikasi.csv'):
        """
        Penjelasan: Melatih model, menghitung waktu eksekusi, dan menyimpan laporan performa.
        Input: report_path (string)
        Output: None (membuat file CSV lokal)
        """
        print("=== Hasil Training dan Akurasi Awal Model ===\n")
        
        all_reports = []
        
        for name, model in self.models.items():
            print(f"mulai training {name}...")
            start_time = time.time()

            with mlflow.start_run(run_name=f"Basic_{name.replace(' ', '_')}"):
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
                self.predictions[name] = y_pred
                
                f1 = f1_score(self.y_test, y_pred, average='weighted')
                print(f"F1-Score {name}: {f1 * 100:.2f}%\n")

                print(f"Laporan {name}:")
                print(classification_report(self.y_test, y_pred, target_names=self.target_names))
                print("-" * 55)
                
                report_dict = classification_report(self.y_test, y_pred, target_names=self.target_names, output_dict=True)
                df_report = pd.DataFrame(report_dict).transpose()
                
                df_report['Model'] = name
                df_report['Metric_Class'] = df_report.index
                all_reports.append(df_report)

            end_time = time.time()
            mins, secs = divmod(end_time - start_time, 60)
            print(f"selesai training {name}: waktu {int(mins)} menit {int(secs)} detik\n")

        df_final_report = pd.concat(all_reports, ignore_index=True)
        kolom_urutan = ['Model', 'Metric_Class', 'precision', 'recall', 'f1-score', 'support']
        df_final_report = df_final_report[kolom_urutan]
        
        df_final_report.to_csv(report_path, index=False)
        print(f"Laporan klasifikasi berhasil disimpan sebagai '{report_path}'")

    def plot_confusion_matrix(self, save_path='confusion_matrix.png'):
        """
        Penjelasan: Menghasilkan gambar Confusion Matrix untuk semua model yang diuji.
        Input: save_path (string)
        Output: None (membuat file PNG lokal)
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes = axes.flatten()
        
        for i, (name, y_pred) in enumerate(self.predictions.items()):
            cm = confusion_matrix(self.y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=self.target_names, yticklabels=self.target_names, ax=axes[i])
            axes[i].set_title(name, fontsize=12, fontweight='bold')
            axes[i].set_xlabel("Prediksi Model")
            axes[i].set_ylabel("Data Aktual")
            
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot berhasil disimpan sebagai '{save_path}'")
        plt.close()

if __name__ == '__main__':
    pipeline = DietModelPipeline(data_path='healthy_diet_calorie_intake_preprocessing.csv')
    pipeline.prepare_data()
    pipeline.train_and_evaluate(report_path='laporan_klasifikasi_awal.csv')
    pipeline.plot_confusion_matrix(save_path='hasil_evaluasi_model_awal.png')