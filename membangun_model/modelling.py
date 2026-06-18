from pathlib import Path
from dataclasses import dataclass

import time, warnings, logging
import mlflow
import mlflow.sklearn

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_theme(style="whitegrid")

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from xgboost import XGBClassifier

logging.getLogger(
    "mlflow"
).setLevel(
    logging.ERROR
)

warnings.filterwarnings(
    "ignore"
)

@dataclass
class EvaluationResult:
    """
    Description:
        Stores evaluation results
        and generated artifacts.

    Args:
        metrics (dict):
            Evaluation metrics.

        report_path (Path):
            Path to classification report CSV.

        plot_path (Path):
            Path to confusion matrix PNG.

        classification_report_text (str):
            Classification report text.

        confusion_matrix_df (pd.DataFrame):
            Confusion matrix dataframe.

    Input:
        Evaluation outputs.

    Output:
        Structured evaluation result.
    """

    metrics: dict
    report_path: Path
    plot_path: Path
    classification_report_text: str
    confusion_matrix_df: pd.DataFrame

class DietModelPipeline:
    """
    Description:
        Pipeline for training,
        evaluating, and logging
        machine learning models.

    Args:
        data_path (str):
            Dataset location.

    Input:
        Dataset path.

    Output:
        Ready-to-run pipeline.
    """

    def __init__(
        self,
        data_path=(
            "healthy_diet_calorie_intake_preprocessing.csv"
        ),
    ):
        """
        Description:
            Initializes pipeline variables,
            model configurations,
            output directories,
            and MLflow tracking.

        Args:
            data_path (str):
                Dataset path.

        Input:
            Dataset location.

        Output:
            Initialized pipeline object.
        """

        self.data_path = data_path

        self.target_names = [
            "Obese",
            "Underweight",
            "Overweight",
            "Healthy",
        ]

        self.predictions = {}

        self.evaluation_results = {}

        self.best_params = {

            "Random Forest": {
                "n_estimators": 200,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
                "max_depth": 20,
                "class_weight": (
                    "balanced_subsample"
                ),
            },

            "SVC": {
                "C": (
                    1.4330109455635685
                ),
                "kernel": "linear",
                "class_weight": None,
            },

            "KNN": {
                "metric": "manhattan",
                "n_neighbors": 6,
                "weights": "distance",
            },

            "Gradient Boosting": {
                "learning_rate": (
                    0.14447746112718687
                ),
                "n_estimators": 207,
                "subsample": (
                    0.8542703315240834
                ),
            },

            "XGBoost": {
                "colsample_bytree": (
                    0.9439761626945282
                ),
                "learning_rate": (
                    0.10113389297817889
                ),
                "max_depth": 3,
                "n_estimators": 266,
                "subsample": (
                    0.6053059844639466
                ),
            },
        }

        self.models = {

            "Random Forest":
            RandomForestClassifier(
                n_estimators=200,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features="sqrt",
                max_depth=20,
                class_weight=(
                    "balanced_subsample"
                ),
                bootstrap=True,
                random_state=42,
                n_jobs=-1,
            ),

            "SVC":
            Pipeline(
                [
                    (
                        "scaler",
                        StandardScaler(),
                    ),
                    (
                        "clf",
                        SVC(
                            C=(
                                1.4330109455635685
                            ),
                            kernel="linear",
                            class_weight=None,
                            probability=True,
                            random_state=42,
                        ),
                    ),
                ]
            ),

            "KNN":
            Pipeline(
                [
                    (
                        "scaler",
                        StandardScaler(),
                    ),
                    (
                        "clf",
                        KNeighborsClassifier(
                            metric=(
                                "manhattan"
                            ),
                            n_neighbors=6,
                            weights=(
                                "distance"
                            ),
                        ),
                    ),
                ]
            ),

            "Gradient Boosting":
            GradientBoostingClassifier(
                learning_rate=(
                    0.14447746112718687
                ),
                n_estimators=207,
                subsample=(
                    0.8542703315240834
                ),
                random_state=42,
            ),

            "XGBoost":
            XGBClassifier(
                colsample_bytree=(
                    0.9439761626945282
                ),
                learning_rate=(
                    0.10113389297817889
                ),
                max_depth=3,
                n_estimators=266,
                subsample=(
                    0.6053059844639466
                ),
                objective=(
                    "multi:softmax"
                ),
                eval_metric=(
                    "mlogloss"
                ),
                random_state=42,
                n_jobs=-1,
            ),
        }

        self.create_output_directories()

        mlflow.set_tracking_uri(
            "http://127.0.0.1:5001/"
        )

        mlflow.set_experiment(
            "Diet_Health_Status_Basic"
        )

        mlflow.sklearn.autolog()

    def create_output_directories(
        self,
    ) -> None:
        """
        Description:
            Creates output folders
            for reports and plots.

        Args:
            None

        Input:
            None

        Output:
            Creates:

            model_base/
            ├── reports/
            └── confusion_matrices/
        """

        self.output_dir = Path(
            "model_base"
        )

        self.report_dir = (
            self.output_dir
            / "reports"
        )

        self.confusion_dir = (
            self.output_dir
            / "confusion_matrices"
        )

        self.output_dir.mkdir(
            exist_ok=True
        )

        self.report_dir.mkdir(
            exist_ok=True
        )

        self.confusion_dir.mkdir(
            exist_ok=True
        )

    def prepare_data(
        self,
        leaked_columns=None,
    ) -> None:
        """
        Description:
            Loads dataset, removes
            leakage features,
            and performs train-test split.

        Args:
            leaked_columns (list[str], optional):
                Columns removed from
                the feature matrix.

        Input:
            CSV dataset.

        Output:
            Creates:

                - self.X_train
                - self.X_test
                - self.y_train
                - self.y_test
        """

        if leaked_columns is None:

            leaked_columns = [
                "BMI", "Height_cm", "Weight_kg", "Health_Status",
            ]

        df = pd.read_csv(
            self.data_path
        )

        X = df.drop(
            columns=[
                column
                for column
                in leaked_columns
                if column in df.columns
            ]
        )

        y = df[
            "Health_Status"
        ]

        (
            self.X_train, self.X_test,
            self.y_train, self.y_test,
        ) = train_test_split(
            X, y,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )

    def _save_classification_report(
        self,
        report_dict: dict,
        model_name: str,
    ) -> Path:
        """
        Description:
            Saves classification report
            as CSV.

        Args:
            report_dict (dict):
                Classification report.

            model_name (str):
                Model name.

        Input:
            Classification report dictionary.

        Output:
            CSV file path.
        """

        safe_name = (
            model_name
            .replace(" ", "_")
            .replace("/", "_")
        )

        report_path = (
            self.report_dir
            / (
                f"classification_report_"
                f"{safe_name}.csv"
            )
        )

        pd.DataFrame(
            report_dict
        ).transpose().to_csv(
            report_path,
            index=True,
        )

        return report_path

    def _save_confusion_matrix(
        self,
        confusion_matrix_df: pd.DataFrame,
        model_name: str,
    ) -> Path:
        """
        Description:
            Saves confusion matrix
            as PNG image.

        Args:
            confusion_matrix_df (pd.DataFrame):
                Confusion matrix dataframe.

            model_name (str):
                Model name.

        Input:
            Confusion matrix dataframe.

        Output:
            PNG file path.
        """

        safe_name = (
            model_name
            .replace(" ", "_")
            .replace("/", "_")
        )

        plot_path = (
            self.confusion_dir
            / (
                f"confusion_matrix_"
                f"{safe_name}.png"
            )
        )

        plt.figure(
            figsize=(8, 6)
        )

        sns.heatmap(
            confusion_matrix_df,
            annot=True,
            fmt="d",
            cmap="Blues",
        )

        plt.title(
            f"Confusion Matrix - "
            f"{model_name}"
        )

        plt.xlabel(
            "Predicted Class"
        )

        plt.ylabel(
            "Actual Class"
        )

        plt.tight_layout()

        plt.savefig(
            plot_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        return plot_path

    def _evaluate_model(
        self,
        model_name: str,
        model,
    ) -> EvaluationResult:
        """
        Description:
            Evaluates a trained model,
            calculates metrics,
            generates reports,
            and saves artifacts.

        Args:
            model_name (str):
                Model name.

            model:
                Trained estimator.

        Input:
            Test features and labels.

        Output:
            EvaluationResult object.
        """

        y_pred = model.predict(
            self.X_test
        )

        self.predictions[
            model_name
        ] = y_pred

        metrics = {
            "accuracy":
            accuracy_score(
                self.y_test,
                y_pred,
            ),

            "precision_weighted":
            precision_score(
                self.y_test,
                y_pred,
                average="weighted",
                zero_division=0,
            ),

            "recall_weighted":
            recall_score(
                self.y_test,
                y_pred,
                average="weighted",
                zero_division=0,
            ),

            "f1_score_weighted":
            f1_score(
                self.y_test,
                y_pred,
                average="weighted",
                zero_division=0,
            ),
        }

        report_dict = (
            classification_report(
                self.y_test,
                y_pred,
                target_names=(
                    self.target_names
                ),
                output_dict=True,
                zero_division=0,
            )
        )

        report_text = (
            classification_report(
                self.y_test,
                y_pred,
                target_names=(
                    self.target_names
                ),
                zero_division=0,
            )
        )

        confusion_matrix_df = (
            pd.DataFrame(
                confusion_matrix(
                    self.y_test,
                    y_pred,
                ),
                index=(
                    self.target_names
                ),
                columns=(
                    self.target_names
                ),
            )
        )

        report_path = (
            self._save_classification_report(
                report_dict=(
                    report_dict
                ),
                model_name=(
                    model_name
                ),
            )
        )

        plot_path = (
            self._save_confusion_matrix(
                confusion_matrix_df=(
                    confusion_matrix_df
                ),
                model_name=(
                    model_name
                ),
            )
        )

        return EvaluationResult(
            metrics=metrics,
            report_path=report_path,
            plot_path=plot_path,
            classification_report_text=(
                report_text
            ),
            confusion_matrix_df=(
                confusion_matrix_df
            ),
        )


    def display_results(
        self,
    ) -> None:
        """
        Description:
            Displays model metrics,
            classification report,
            confusion matrix,
            and best parameters.

        Args:
            None

        Input:
            Evaluation results.

        Output:
            Console output.
        """

        for (
            model_name,
            result
        ) in self.evaluation_results.items():

            print(
                "\n"
                + "=" * 60
            )

            print(
                model_name.upper()
            )

            print(
                "=" * 60
            )

            print(
                "\nBest Parameters"
            )

            print(
                "-" * 60
            )

            for (
                parameter,
                value
            ) in self.best_params[
                model_name
            ].items():

                print(
                    f"{parameter:<30}: {value}"
                )

            print(
                "\nEvaluation Metrics"
            )

            print(
                "-" * 60
            )

            for (
                metric,
                value
            ) in result.metrics.items():

                print(
                    f"{metric:<30}: "
                    f"{value:.4f}"
                )

            print(
                "\nClassification Report"
            )

            print(
                "-" * 60
            )

            print(
                result.classification_report_text
            )

            print(
                "\nConfusion Matrix"
            )

            print(
                "-" * 60
            )

            print(
                result.confusion_matrix_df
            )

    def save_model_ranking(
        self,
    ) -> None:
        """
        Description:
            Creates model ranking
            based on weighted F1-score.

        Args:
            None

        Input:
            Evaluation results.

        Output:
            ranking.csv file.
        """

        ranking = pd.DataFrame(
            {
                model_name:
                result.metrics

                for (
                    model_name,
                    result
                )

                in self.evaluation_results.items()
            }
        ).T

        ranking = ranking.sort_values(
            by="f1_score_weighted",
            ascending=False,
        )

        ranking_path = (
            self.output_dir
            / "ranking.csv"
        )

        ranking.to_csv(
            ranking_path,
            index=True,
        )

        print(
            "\n"
            + "=" * 60
        )

        print(
            "MODEL RANKING"
        )

        print(
            "=" * 60
        )

        print(
            ranking
        )

    def train_and_evaluate(
        self,
    ) -> None:
        """
        Description:
            Trains every model,
            evaluates performance,
            logs artifacts to MLflow,
            and generates ranking.

        Args:
            None

        Input:
            Training and testing datasets.

        Output:
            Evaluation artifacts.
        """

        for (
            model_name,
            model
        ) in self.models.items():

            print(
                "\n"
                + "=" * 60
            )

            print(
                f"Training {model_name}"
            )

            print(
                "=" * 60
            )

            start_time = (
                time.time()
            )

            
            with mlflow.start_run(
                run_name=f"Base_{model_name.replace(' ', '_')}"
            ):

                # Train model
                model.fit(self.X_train, self.y_train)

                # Evaluate
                evaluation_result = self._evaluate_model(model_name, model)

                self.evaluation_results[model_name] = evaluation_result

                # Log metrics (aman kalau dict kosong)
                if evaluation_result.metrics:
                    mlflow.log_metrics(evaluation_result.metrics)

                # Log artifacts (report + plot)
                mlflow.log_artifact(str(evaluation_result.report_path))
                mlflow.log_artifact(str(evaluation_result.plot_path))

                # 🔥 LOG MODEL (ini yang penting)
                if model_name in ["XGBoost", "KNN"]:
                    mlflow.sklearn.log_model(
                        model,
                        "model",
                        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
                    )
                else:
                    mlflow.sklearn.log_model(
                        model,
                        "model",
                    )

                # optional: tag biar gampang tracking di UI
                mlflow.set_tag("model_name", model_name)

            elapsed_time = (
                time.time()
                - start_time
            )

            minutes, seconds = (
                divmod(
                    elapsed_time,
                    60,
                )
            )

            print(
                f"\nCompleted in "
                f"{int(minutes)} minutes "
                f"{int(seconds)} seconds"
            )

        self.display_results()

        self.save_model_ranking()

    def plot_confusion_matrix(
        self,
        save_path=(
            "model_base/"
            "all_models_confusion_matrix.png"
        ),
    ) -> None:
        """
        Description:
            Creates a combined
            confusion matrix figure
            for all trained models.

        Args:
            save_path (str):
                Output image path.

        Input:
            Predictions from all models.

        Output:
            PNG image.
        """

        number_of_models = (
            len(
                self.predictions
            )
        )

        fig, axes = plt.subplots(
            1,
            number_of_models,
            figsize=(
                6 * number_of_models, 5,
            ),
        )

        if (
            number_of_models == 1
        ):
            axes = [axes]

        for (
            axis,
            (
                model_name,
                prediction,
            ),
        ) in zip(
            axes,
            self.predictions.items(),
        ):

            sns.heatmap(
                confusion_matrix(
                    self.y_test,
                    prediction,
                ),
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=(
                    self.target_names
                ),
                yticklabels=(
                    self.target_names
                ),
                ax=axis,
            )

            axis.set_title(
                model_name
            )

            axis.set_xlabel(
                "Predicted"
            )

            axis.set_ylabel(
                "Actual"
            )

        plt.tight_layout()

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

if __name__ == "__main__":
    pipeline = (
        DietModelPipeline(
            data_path=
            "healthy_diet_calorie_intake_preprocessing.csv"
        )
    )

    pipeline.prepare_data()
    pipeline.train_and_evaluate()
    pipeline.plot_confusion_matrix()