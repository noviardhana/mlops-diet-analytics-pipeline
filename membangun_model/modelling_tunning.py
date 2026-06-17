"""
Description:
    Hyperparameter tuning pipeline for diet health status classification.

    Models Included:
        - Random Forest
        - Support Vector Classifier (SVC)
        - K-Nearest Neighbors (KNN)
        - Gradient Boosting
        - XGBoost

    Features:
        - Hyperparameter tuning using RandomizedSearchCV
        - Evaluation metrics generation
        - Classification report generation
        - Confusion matrix visualization
        - MLflow logging
        - DagsHub integration
"""

from __future__ import annotations

import logging, os, sys, time, warnings

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import seaborn as sns

from dotenv import load_dotenv
from scipy.stats import loguniform, randint, uniform

from sklearn.base import ClassifierMixin
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
)
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier


warnings.filterwarnings("ignore")

logging.getLogger("mlflow").setLevel(logging.ERROR)

RANDOM_STATE = 42
CV_FOLDS = 5

TARGET_NAMES = [
    "Obese",
    "Underweight",
    "Overweight",
    "Healthy",
]

LOG_PATH = Path("tuning_errors.log")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """
    Description:
        Stores the configuration required to tune a machine learning model.

    Args:
        name (str):
            Human-readable model name.

        estimator (ClassifierMixin):
            Machine learning estimator or pipeline.

        param_distributions (Any):
            Hyperparameter search space for RandomizedSearchCV.

        n_iter (int):
            Number of random search iterations.

        needs_balanced_sample_weight (bool):
            Indicates whether balanced sample weights
            should be supplied during training.

    Input:
        Model metadata and hyperparameter search space.

    Output:
        Model configuration object.
    """

    name: str
    estimator: ClassifierMixin
    param_distributions: Any
    n_iter: int
    needs_balanced_sample_weight: bool = False


@dataclass
class EvaluationResult:
    """
    Description:
        Stores evaluation results and generated artifacts
        for a trained model.

    Args:
        metrics (dict[str, float]):
            Evaluation metric values.

        report_path (Path):
            Location of the exported classification report.

        plot_path (Path):
            Location of the exported confusion matrix image.

        classification_report_text (str):
            Text version of the classification report.

        confusion_matrix_df (pd.DataFrame):
            Confusion matrix represented as a DataFrame.

    Input:
        Model predictions and evaluation artifacts.

    Output:
        Structured evaluation result object.
    """

    metrics: dict[str, float]
    report_path: Path
    plot_path: Path
    classification_report_text: str
    confusion_matrix_df: pd.DataFrame

@contextmanager
def tee_stdout(log_path: Path):
    """
    Description:
        Duplicates terminal output to both the console and
        a log file simultaneously.

        This is useful because RandomizedSearchCV and joblib
        often write directly to stdout instead of using
        the logging module.

    Args:
        log_path (Path):
            Destination log file.

    Input:
        Standard output stream.

    Output:
        Context manager that mirrors console output
        into a file.
    """

    original_stdout = sys.stdout

    with log_path.open("a", encoding="utf-8") as log_file:

        class TeeWriter:
            """
            Description:
                Internal helper class used to duplicate
                stdout output.

            Args:
                None

            Input:
                Text written to stdout.

            Output:
                Writes text to terminal and log file.
            """

            def write(self, message: str) -> None:
                original_stdout.write(message)
                log_file.write(message)

            def flush(self) -> None:
                original_stdout.flush()
                log_file.flush()

        sys.stdout = TeeWriter()

        try:
            yield

        finally:
            sys.stdout = original_stdout


class XGBClassifierWithLabelEncoding(XGBClassifier):
    """
    Description:
        Extends XGBClassifier by automatically encoding
        string target labels during training and decoding
        them during prediction.

        This allows XGBoost to behave consistently with
        other classifiers that accept string labels directly.

    Args:
        Same arguments accepted by XGBClassifier.

    Input:
        Feature matrix and string target labels.

    Output:
        Trained XGBoost model capable of handling
        string targets transparently.
    """

    def fit(self, X, y, sample_weight=None, **kwargs,):
        """
        Description:
            Fits an internal LabelEncoder before training
            the underlying XGBoost model.

        Args:
            X:
                Training feature matrix.

            y:
                String target labels.

            sample_weight:
                Optional sample weights.

            **kwargs:
                Additional keyword arguments forwarded
                to XGBoost.

        Input:
            Training features and target labels.

        Output:
            Fitted estimator instance.
        """

        self._label_encoder = LabelEncoder().fit(y)
        encoded_labels = self._label_encoder.transform(y)

        return super().fit(
            X,
            encoded_labels,
            sample_weight=sample_weight,
            **kwargs,
        )

    def predict(self, X):
        """
        Description:
            Generates predictions and converts encoded
            class labels back to their original values.

        Args:
            X:
                Feature matrix used for prediction.

        Input:
            Feature matrix.

        Output:
            Predicted class labels in their original form.
        """

        encoded_predictions = super().predict(X)

        return self._label_encoder.inverse_transform(
            encoded_predictions
        )

class DietModelPipeline:
    """
    Description:
        Main machine learning pipeline responsible for:

        - Data preparation
        - Hyperparameter tuning
        - Model evaluation
        - Artifact generation
        - MLflow logging
        - DagsHub integration

    Args:
        data_path (str):
            Path to the preprocessed dataset.

    Input:
        Dataset path and pipeline configuration.

    Output:
        Fully configured machine learning pipeline.
    """

    def __init__(
        self,
        data_path: str = "healthy_diet_calorie_intake_preprocessing.csv",
    ) -> None:
        """
        Description:
            Initializes pipeline attributes and disables
            automatic MLflow logging.

        Args:
            data_path (str):
                Dataset location.

        Input:
            Dataset file path.

        Output:
            Initialized pipeline object.
        """

        self.data_path = data_path
        self.tuned_models: dict[str, ClassifierMixin] = {}
        self.best_params: dict[str, dict[str, Any]] = {}
        self.tuned_predictions: dict[str, Any] = {}
        self.evaluation_results: dict[str, EvaluationResult] = {}
        self.dagshub_uri: str | None = None

        mlflow.sklearn.autolog(disable=True)

    def prepare_data( 
        self, 
        leaked_columns: list[str] = ( 
            "BMI", "Height_cm", "Weight_kg", "Health_Status", 
        ), 
    ) -> None:
        """
        Description:
            Loads the dataset, removes data leakage columns,
            and creates stratified train-test splits.

        Args:
            leaked_columns (list[str]):
                Features that must be removed before training.

        Input:
            Dataset CSV file.

        Output:
            Updates:
                - self.X_train
                - self.X_test
                - self.y_train
                - self.y_test
        """

        logger.info("Loading dataset...")

        dataset = pd.read_csv(self.data_path)

        feature_columns = [
            column
            for column in dataset.columns
            if column not in leaked_columns
        ]

        X = dataset[feature_columns]

        y = dataset["Health_Status"]

        (
            self.X_train, self.X_test, self.y_train, self.y_test,
        ) = train_test_split(
            X, y,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        logger.info(
            "Dataset loaded successfully. "
            "Training samples: %d | Testing samples: %d",
            len(self.X_train),
            len(self.X_test),
        )

    def setup_credentials(
        self,
        dagshub_username: str,
        dagshub_repo: str,
    ) -> None:
        """
        Description:
            Loads DagsHub credentials from environment variables
            and configures MLflow authentication.

        Args:
            dagshub_username (str):
                DagsHub account username.

            dagshub_repo (str):
                DagsHub repository name.

        Input:
            .env file containing:
                MLFLOW_TRACKING_PASSWORD

        Output:
            Configured DagsHub tracking URI.
        """

        load_dotenv()

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_username

        token = os.getenv("MLFLOW_TRACKING_PASSWORD")

        if not token:
            logger.warning(
                "MLFLOW_TRACKING_PASSWORD was not found. "
                "DagsHub logging will be skipped."
            )
            return

        os.environ["MLFLOW_TRACKING_PASSWORD"] = token

        self.dagshub_uri = (
            f"https://dagshub.com/"
            f"{dagshub_username}/"
            f"{dagshub_repo}.mlflow"
        )

        logger.info(
            "DagsHub tracking URI configured successfully."
        )

    def _build_model_configs(self) -> list[ModelConfig]:
        """
        Description:
            Creates all machine learning model configurations
            and their corresponding hyperparameter search spaces.

        Args:
            None

        Input:
            Internal model definitions and search spaces.

        Output:
            List of ModelConfig objects.
        """

        svc_pipeline = Pipeline(
            [
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "clf",
                    SVC(
                        probability=True, random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )

        knn_pipeline = Pipeline(
            [
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "clf",
                    KNeighborsClassifier(),
                ),
            ]
        )

        return [
            ModelConfig(
                name="Tuned Random Forest",
                estimator=RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    bootstrap=True,
                ),
                param_distributions={
                    "n_estimators": [100, 200, 300],
                    "max_depth": [10, 20, None],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2],
                    "max_features": ["sqrt", "log2", ],
                    "class_weight": ["balanced", "balanced_subsample", ],
                },
                n_iter=15,
            ),
            ModelConfig(
                name="Tuned SVC",
                estimator=svc_pipeline,
                param_distributions=[
                    {
                        "clf__kernel": ["rbf"],
                        "clf__C": loguniform(
                            0.1, 100,
                        ),
                        "clf__gamma": loguniform(
                            0.001, 10,
                        ),
                        "clf__class_weight": [
                            "balanced", None,
                        ],
                    },
                    {
                        "clf__kernel": ["linear"],
                        "clf__C": loguniform(
                            0.1, 100,
                        ),
                        "clf__class_weight": [
                            "balanced", None,
                        ],
                    },
                ],
                n_iter=15,
            ),
            ModelConfig(
                name="Tuned KNN",
                estimator=knn_pipeline,
                param_distributions={
                    "clf__n_neighbors": randint(
                        3, 21,
                    ),
                    "clf__weights": [
                        "uniform", "distance",
                    ],
                    "clf__metric": [
                        "euclidean", "manhattan",
                    ],
                },
                n_iter=12,
            ),
            ModelConfig(
                name="Tuned Gradient Boosting",
                estimator=GradientBoostingClassifier(
                    random_state=RANDOM_STATE,
                ),
                param_distributions={
                    "n_estimators": randint(
                        100, 300,
                    ),
                    "learning_rate": loguniform(
                        0.01, 0.30,
                    ),
                    "max_depth": randint(
                        2, 6,
                    ),
                    "subsample": uniform(
                        0.70, 0.30,
                    ),
                },
                n_iter=15,
                needs_balanced_sample_weight=True,
            ),
            ModelConfig(
                name="Tuned XGBoost",
                estimator=XGBClassifierWithLabelEncoding(
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
                param_distributions={
                    "n_estimators": randint(
                        100, 300,
                    ),
                    "learning_rate": loguniform(
                        0.01, 0.30,
                    ),
                    "max_depth": randint(
                        3, 8,
                    ),
                    "subsample": uniform(
                        0.60, 0.40,
                    ),
                    "colsample_bytree": uniform(
                        0.60, 0.40,
                    ),
                },
                n_iter=15,
                needs_balanced_sample_weight=True,
            ),
        ]

    def _run_search(
        self,
        config: ModelConfig,
    ) -> bool:
        """
        Description:
            Executes RandomizedSearchCV for a single model
            configuration.

            If tuning fails, the error is logged and
            the pipeline continues with the next model.

        Args:
            config (ModelConfig):
                Model configuration object.

        Input:
            Training features and training labels.

        Output:
            bool:
                True if tuning succeeds,
                False otherwise.
        """

        search = RandomizedSearchCV(
            estimator=config.estimator,
            param_distributions=config.param_distributions,
            n_iter=config.n_iter,
            cv=CV_FOLDS,
            scoring="f1_weighted",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=1,
        )

        fit_kwargs = {}

        if config.needs_balanced_sample_weight:
            fit_kwargs["sample_weight"] = compute_sample_weight(
                class_weight="balanced",
                y=self.y_train,
            )

        logger.info(
            "Started tuning %s...",
            config.name,
        )

        start_time = time.time()

        try:
            with tee_stdout(LOG_PATH):
                search.fit(
                    self.X_train,
                    self.y_train,
                    **fit_kwargs,
                )

        except Exception:
            logger.exception(
                "Tuning failed for %s.",
                config.name,
            )
            return False

        elapsed_time = time.time() - start_time

        minutes, seconds = divmod(
            elapsed_time,
            60,
        )

        self.tuned_models[
            config.name
        ] = search.best_estimator_

        self.best_params[
            config.name
        ] = search.best_params_

        logger.info(
            "Finished tuning %s in %d minutes %d seconds.",
            config.name,
            int(minutes),
            int(seconds),
        )

        return True

    def tuning_models(self) -> None:
        """
        Description:
            Iterates through every model configuration
            and performs hyperparameter tuning.

        Args:
            None

        Input:
            Model configurations.

        Output:
            Updates:
                - self.tuned_models
                - self.best_params
        """

        logger.info(
            "Starting hyperparameter tuning process..."
        )

        for config in self._build_model_configs():
            self._run_search(config)

        if not self.tuned_models:
            logger.error(
                "All tuning jobs failed. "
                "Check %s for details.",
                LOG_PATH,
            )

        logger.info(
            "Hyperparameter tuning completed."
        )


    def _create_output_directories(self) -> tuple[Path, Path]:
        """
        Description:
            Creates output directories for evaluation artifacts.

        Args:
            None

        Input:
            None

        Output:
            Tuple containing:
                - report directory
                - confusion matrix directory
        """

        output_dir = Path("model_tuning")

        report_dir = output_dir / "reports"

        confusion_dir = (
            output_dir
            / "confusion_matrices"
        )

        report_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        confusion_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        return report_dir, confusion_dir


    def _save_classification_report(
        self,
        report_dict: dict,
        model_name: str,
        report_dir: Path,
    ) -> Path:
        """
        Description:
            Saves classification report as CSV.

        Args:
            report_dict (dict)
            model_name (str)
            report_dir (Path)

        Input:
            Classification report dictionary.

        Output:
            Path to saved CSV file.
        """

        safe_name = (
            model_name
            .replace(" ", "_")
            .replace("/", "_")
        )

        report_path = (
            report_dir
            / f"evaluation_report_{safe_name}.csv"
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
        confusion_dir: Path,
    ) -> Path:
        """
        Description:
            Saves confusion matrix figure.

        Args:
            confusion_matrix_df (pd.DataFrame)
            model_name (str)
            confusion_dir (Path)

        Input:
            Confusion matrix dataframe.

        Output:
            Path to saved PNG file.
        """

        safe_name = (
            model_name
            .replace(" ", "_")
            .replace("/", "_")
        )

        plot_path = (
            confusion_dir
            / f"confusion_matrix_{safe_name}.png"
        )

        plt.figure(figsize=(8, 6))

        sns.heatmap(
            confusion_matrix_df,
            annot=True,
            fmt="d",
            cmap="Greens",
        )

        plt.title(
            f"Confusion Matrix - {model_name}"
        )

        plt.ylabel("Actual Class")
        plt.xlabel("Predicted Class")

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
        model: ClassifierMixin,
    ) -> EvaluationResult:

        y_pred = model.predict(
            self.X_test
        )

        self.tuned_predictions[
            model_name
        ] = y_pred

        metrics = {
            "accuracy": accuracy_score(
                self.y_test,
                y_pred,
            ),
            "precision_weighted": precision_score(
                self.y_test,
                y_pred,
                average="weighted",
                zero_division=0,
            ),
            "recall_weighted": recall_score(
                self.y_test,
                y_pred,
                average="weighted",
                zero_division=0,
            ),
            "f1_score_weighted": f1_score(
                self.y_test,
                y_pred,
                average="weighted",
                zero_division=0,
            ),
        }

        report_dict = classification_report(
            self.y_test,
            y_pred,
            target_names=TARGET_NAMES,
            output_dict=True,
            zero_division=0,
        )

        report_text = classification_report(
            self.y_test,
            y_pred,
            target_names=TARGET_NAMES,
            zero_division=0,
        )

        cm_df = pd.DataFrame(
            confusion_matrix(
                self.y_test,
                y_pred,
            ),
            index=TARGET_NAMES,
            columns=TARGET_NAMES,
        )

        report_dir, confusion_dir = (
            self._create_output_directories()
        )

        report_path = (
            self._save_classification_report(
                report_dict,
                model_name,
                report_dir,
            )
        )

        plot_path = (
            self._save_confusion_matrix(
                cm_df,
                model_name,
                confusion_dir,
            )
        )

        return EvaluationResult(
            metrics=metrics,
            report_path=report_path,
            plot_path=plot_path,
            classification_report_text=report_text,
            confusion_matrix_df=cm_df,
        )
    
    def save_model_ranking(self) -> None:
        """
        Description:
            Creates and saves model ranking
            based on weighted F1-score.

        Args:
            None

        Input:
            Evaluation results.

        Output:
            ranking.csv
        """

        output_dir = Path(
            "model_tuning"
        )

        ranking = pd.DataFrame(
            {
                model_name: result.metrics
                for (
                    model_name,
                    result
                ) in self.evaluation_results.items()
            }
        ).T

        ranking = ranking.sort_values(
            by="f1_score_weighted",
            ascending=False,
        )

        ranking_path = (
            output_dir
            / "ranking.csv"
        )

        ranking.to_csv(
            ranking_path,
            index=True,
        )

        print("\nMODEL RANKING")
        print("=" * 60)
        print(ranking)


    def _build_evaluation_artifacts(
        self,
    ) -> dict[str, EvaluationResult]:
        """
        Description:
            Evaluates every successfully tuned model
            and stores the resulting artifacts.

        Args:
            None

        Input:
            Tuned models.

        Output:
            Dictionary containing all
            EvaluationResult objects.
        """

        logger.info(
            "Generating evaluation artifacts..."
        )

        evaluation_results = {}

        for (
            model_name,
            model,
        ) in self.tuned_models.items():

            result = self._evaluate_model(
                model_name=model_name,
                model=model,
            )

            evaluation_results[
                model_name
            ] = result

        self.evaluation_results = (
            evaluation_results
        )

        logger.info(
            "Evaluation artifact generation completed."
        )

        return evaluation_results

    def display_results(self) -> None:
        """
        Description:
            Displays tuning results, evaluation metrics,
            classification reports, and confusion matrices
            for every successfully tuned model.

        Args:
            None

        Input:
            Stored tuning and evaluation results.

        Output:
            Formatted console output.
        """

        print("\n")
        print("=" * 60)
        print("MODEL EVALUATION RESULTS")
        print("=" * 60)

        for model_name, result in self.evaluation_results.items():

            print("\n")
            print("=" * 60)
            print(model_name.upper())
            print("=" * 60)

            print("\nBest Parameters")
            print("-" * 60)

            for parameter, value in self.best_params[
                model_name
            ].items():
                print(
                    f"{parameter:<30}: {value}"
                )

            print("\nEvaluation Metrics")
            print("-" * 60)

            for metric_name, metric_value in (
                result.metrics.items()
            ):
                print(
                    f"{metric_name:<30}: "
                    f"{metric_value:.4f}"
                )

            print("\nClassification Report")
            print("-" * 60)

            print(
                result.classification_report_text
            )

            print("\nConfusion Matrix")
            print("-" * 60)

            print(
                result.confusion_matrix_df
            )

            print("\nSaved Artifacts")
            print("-" * 60)

            print(
                f"Report CSV : "
                f"{result.report_path}"
            )

            print(
                f"Confusion Matrix PNG : "
                f"{result.plot_path}"
            )

    def _log_all_models_to(
        self,
        tracking_uri: str,
        experiment_name: str,
        run_prefix: str,
        evaluation_results: dict[
            str,
            EvaluationResult,
        ],
    ) -> None:
        """
        Description:
            Logs all trained models, metrics,
            parameters, and artifacts to a specified
            MLflow tracking server.

        Args:
            tracking_uri (str):
                MLflow tracking URI.

            experiment_name (str):
                MLflow experiment name.

            run_prefix (str):
                Prefix used for MLflow run names.

            evaluation_results
            (dict[str, EvaluationResult]):
                Evaluation results dictionary.

        Input:
            Trained models and generated artifacts.

        Output:
            Logged MLflow runs.
        """

        mlflow.set_tracking_uri(
            tracking_uri
        )

        mlflow.set_experiment(
            experiment_name
        )

        for (
            model_name,
            model,
        ) in self.tuned_models.items():

            result = evaluation_results[
                model_name
            ]

            try:

                with mlflow.start_run(
                    run_name=(
                        f"{run_prefix}_"
                        f"{model_name}"
                    )
                ):

                    mlflow.log_params(
                        self.best_params[model_name]
                    )

                    mlflow.log_metrics(
                        result.metrics
                    )

                    mlflow.log_artifact(
                        str(
                            result.report_path
                        )
                    )

                    mlflow.log_artifact(
                        str(
                            result.plot_path
                        )
                    )

                    mlflow.sklearn.log_model(
                        sk_model=model,
                        artifact_path="model",
                        serialization_format=(
                            mlflow.sklearn
                            .SERIALIZATION_FORMAT_CLOUDPICKLE
                        ),
                    )

                logger.info(
                    "%s successfully logged to %s.",
                    model_name,
                    run_prefix,
                )

            except Exception:

                logger.exception(
                    "Failed to log %s to %s.",
                    model_name,
                    run_prefix,
                )

    def evaluate_and_log_mlflow(
        self,
    ) -> None:
        """
        Description:
            Executes model evaluation,
            generates artifacts,
            displays results,
            and logs everything to
            MLflow and DagsHub.

        Args:
            None

        Input:
            Tuned models.

        Output:
            Evaluation artifacts and
            MLflow tracking records.
        """

        logger.info(
            "Starting evaluation process..."
        )

        evaluation_results = (
            self._build_evaluation_artifacts()
        )

        self.display_results()

        self.save_model_ranking()

        logger.info(
            "Starting local MLflow logging..."
        )

        self._log_all_models_to(
            tracking_uri=(
                "http://127.0.0.1:5001/"
            ),
            experiment_name=(
                "Diet_Health_Status_Skilled"
            ),
            run_prefix="Local",
            evaluation_results=(
                evaluation_results
            ),
        )

        if self.dagshub_uri:

            logger.info(
                "Starting DagsHub logging..."
            )

            self._log_all_models_to(
                tracking_uri=self.dagshub_uri,
                experiment_name=(
                    "Diet_Health_Status_Advance"
                ),
                run_prefix="DagsHub",
                evaluation_results=(evaluation_results
                ),
            )

        else:

            logger.warning(
                "DagsHub URI not available. "
                "Skipping DagsHub logging."
            )

if __name__ == "__main__":
    """
    Description:
        Main application entry point.

    Workflow:
        1. Configure credentials.
        2. Load dataset.
        3. Tune models.
        4. Evaluate models.
        5. Generate artifacts.
        6. Log everything to MLflow.
    """

    pipeline = DietModelPipeline()

    DAGSHUB_USERNAME = ("noviardhana")
    DAGSHUB_REPO = ("sml_noviardhana")
    pipeline.setup_credentials(
        dagshub_username=(
            DAGSHUB_USERNAME
        ),
        dagshub_repo=(
            DAGSHUB_REPO
        ),
    )

    pipeline.prepare_data()
    pipeline.tuning_models()
    pipeline.evaluate_and_log_mlflow()