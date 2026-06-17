"""
Description: Hyperparameter tuning pipeline for diet health status classification.
             Models included: Random Forest, SVC, KNN, Gradient Boosting, XGBoost.
             Features dual MLflow logging (Local + DagsHub) with individual model error handling.
"""
from __future__ import annotations

import os
import sys
import time
import logging
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from scipy.stats import loguniform, uniform, randint

import mlflow
import mlflow.sklearn
from sklearn.base import ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
logging.getLogger("mlflow").setLevel(logging.ERROR)

RANDOM_STATE = 42
CV_FOLDS = 5
TARGET_NAMES = ["Obese", "Underweight", "Overweight", "Healthy"]
LOG_PATH = Path("tuning_errors.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@contextmanager
def tee_stdout(log_path: Path):
    """
    Description: Copies stdout to a log file while the block is running, ensuring that verbose output 
                 from RandomizedSearchCV or joblib (which writes directly to stdout bypassing the logging module) 
                 is saved to the file without removing the display from the terminal.
    Args: 
        log_path (Path): Pathlib object representing the log file destination.
    Input: System standard output stream.
    Output: Yields control back to the context while capturing and duplicating standard output.
    """
    old_stdout = sys.stdout
    with log_path.open("a", encoding="utf-8") as log_file:

        class _Tee:
            def write(self, data: str) -> None:
                old_stdout.write(data)
                log_file.write(data)

            def flush(self) -> None:
                old_stdout.flush()
                log_file.flush()

        sys.stdout = _Tee()
        try:
            yield
        finally:
            sys.stdout = old_stdout


class XGBClassifierWithLabelEncoding(XGBClassifier):
    """
    Description: A custom wrapper for XGBClassifier that automatically encodes and decodes 
                 string labels internally. This allows it to act as a drop-in replacement 
                 for other models (like RandomForest, SVC) without requiring external code modifications.
    Args: None
    Input: Inherits arguments from xgboost.XGBClassifier.
    Output: An instantiated XGBClassifier model capable of handling string target variables directly.
    """

    def fit(self, X, y, sample_weight=None, **kwargs):
        """
        Description: Fits the label encoder on the target variable before passing it to the standard XGBoost fit method.
        Args:
            X: Array-like feature matrix.
            y: Array-like target vector with string labels.
            sample_weight: Array-like sample weights (optional).
            **kwargs: Additional keyword arguments for the fit method.
        Input: Unencoded target array `y` and feature matrix `X`.
        Output: The fitted estimator instance.
        """
        self._label_encoder = LabelEncoder().fit(y)
        y_encoded = self._label_encoder.transform(y)
        return super().fit(X, y_encoded, sample_weight=sample_weight, **kwargs)

    def predict(self, X):
        """
        Description: Predicts class labels and decodes them back to their original string format.
        Args:
            X: Array-like feature matrix to predict on.
        Input: Feature matrix `X`.
        Output: Array of predicted original string labels.
        """
        return self._label_encoder.inverse_transform(super().predict(X))


@dataclass
class ModelConfig:
    """
    Description: A data class representing the configuration of a single model for hyperparameter tuning.
    Args:
        name (str): The display name of the model.
        estimator (ClassifierMixin): The uninstantiated or instantiated scikit-learn compatible classifier/pipeline.
        param_distributions (Any): Dictionary or list of dictionaries specifying the hyperparameter search space.
        n_iter (int): Number of iterations for RandomizedSearchCV.
        needs_balanced_sample_weight (bool): Flag indicating if the model needs computed sample weights during fitting.
    Input: Initialization parameters.
    Output: An instantiated ModelConfig data object.
    """
    name: str
    estimator: ClassifierMixin
    param_distributions: Any
    n_iter: int
    needs_balanced_sample_weight: bool = False


class DietModelPipeline:
    """
    Description: Main pipeline class to process diet data, tune multiple classification models, 
                 and log evaluation results and artifacts to MLflow (both locally and on DagsHub).
    Args:
        data_path (str): The string path to the CSV dataset.
    Input: Dataset path string (defaults to 'healthy_diet_calorie_intake_preprocessing.csv').
    Output: An instantiated DietModelPipeline object ready for execution.
    """

    def __init__(self, data_path: str = "healthy_diet_calorie_intake_preprocessing.csv"):
        self.data_path = data_path
        self.tuned_models: dict[str, ClassifierMixin] = {}
        self.best_params: dict[str, dict] = {}
        self.tuned_predictions: dict[str, Any] = {}
        self.dagshub_uri: str | None = None

        mlflow.sklearn.autolog(disable=True)

    def prepare_data(
        self, leaked_columns: list[str] = ("BMI", "Height_cm", "Weight_kg", "Health_Status")
    ) -> None:
        """
        Description: Reads the CSV dataset, removes features that could cause data leakage, 
                     and splits the data into stratified training and testing sets.
        Args:
            leaked_columns (list[str]): List of column names to be removed from the features.
        Input: The raw CSV file specified by self.data_path.
        Output: Sets self.X_train, self.X_test, self.y_train, and self.y_test attributes.
        """
        df = pd.read_csv(self.data_path)
        X = df.drop(columns=[col for col in leaked_columns if col in df.columns])
        y = df["Health_Status"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )

    def setup_credentials(self, dagshub_username: str, dagshub_repo: str) -> None:
        """
        Description: Loads environment variables and configures MLflow credentials to enable logging to DagsHub.
        Args:
            dagshub_username (str): The exact username of the DagsHub account.
            dagshub_repo (str): The repository name on DagsHub.
        Input: A '.env' file containing the 'MLFLOW_TRACKING_PASSWORD' token.
        Output: Configures OS environment variables and sets self.dagshub_uri.
        """
        load_dotenv()
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_username

        token = os.getenv("MLFLOW_TRACKING_PASSWORD")
        if not token:
            logger.warning("Token not found in .env! DagsHub logging will be skipped.")
            return

        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
        self.dagshub_uri = f"https://dagshub.com/{dagshub_username}/{dagshub_repo}.mlflow"

    def _build_model_configs(self) -> list[ModelConfig]:
        """
        Description: Defines all models and their respective hyperparameter search spaces in one centralized method.
        Args: None
        Input: Hardcoded hyperparameter dictionaries and classifier pipeline initializations.
        Output: A list of instantiated ModelConfig objects to be iterated over during tuning.
        """
        svc_pipeline = Pipeline([("scaler", StandardScaler()), ("clf", SVC(probability=True, random_state=RANDOM_STATE))])
        knn_pipeline = Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier())])

        return [
            ModelConfig(
                name="Tuned Random Forest",
                estimator=RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, bootstrap=True),
                param_distributions={
                    "n_estimators": [100, 200, 300],
                    "max_depth": [10, 20, None],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2],
                    "max_features": ["sqrt", "log2"],
                    "class_weight": ["balanced", "balanced_subsample"],
                },
                n_iter=15,
            ),
            ModelConfig(
                name="Tuned SVC",
                estimator=svc_pipeline,
                param_distributions=[
                    {
                        "clf__kernel": ["rbf"],
                        "clf__C": loguniform(0.1, 100),
                        "clf__gamma": loguniform(0.001, 10),
                        "clf__class_weight": ["balanced", None],
                    },
                    {
                        "clf__kernel": ["linear"],
                        "clf__C": loguniform(0.1, 100),
                        "clf__class_weight": ["balanced", None],
                    },
                ],
                n_iter=15,
            ),
            ModelConfig(
                name="Tuned KNN",
                estimator=knn_pipeline,
                param_distributions={
                    "clf__n_neighbors": randint(3, 21),
                    "clf__weights": ["uniform", "distance"],
                    "clf__metric": ["euclidean", "manhattan"],
                },
                n_iter=12,
            ),
            ModelConfig(
                name="Tuned Gradient Boosting",
                estimator=GradientBoostingClassifier(random_state=RANDOM_STATE),
                param_distributions={
                    "n_estimators": randint(100, 300),
                    "learning_rate": loguniform(0.01, 0.3),
                    "max_depth": randint(2, 6),
                    "subsample": uniform(0.7, 0.3),
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
                    "n_estimators": randint(100, 300),
                    "learning_rate": loguniform(0.01, 0.3),
                    "max_depth": randint(3, 8),
                    "subsample": uniform(0.6, 0.4),
                    "colsample_bytree": uniform(0.6, 0.4),
                },
                n_iter=15,
                needs_balanced_sample_weight=True,
            ),
        ]

    def _run_search(self, config: ModelConfig) -> bool:
        """
        Description: Executes a RandomizedSearchCV block with integrated error handling. If a model fails to tune, 
                     the pipeline will safely skip it and continue to the next model without crashing globally.
        Args:
            config (ModelConfig): The configuration object containing the estimator and search space.
        Input: The configured model search objects and the training dataset.
        Output: Returns True if tuning was successful (updating internal state dictionaries), or False if it failed.
        """
        search = RandomizedSearchCV(
            config.estimator,
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
            fit_kwargs["sample_weight"] = compute_sample_weight("balanced", self.y_train)

        start = time.time()
        logger.info("Started tuning %s...", config.name)

        try:
            with tee_stdout(LOG_PATH):
                search.fit(self.X_train, self.y_train, **fit_kwargs)
        except Exception:
            logger.exception("Tuning %s FAILED, skipping this model.", config.name)
            return False

        self.tuned_models[config.name] = search.best_estimator_
        self.best_params[config.name] = search.best_params_

        mins, secs = divmod(time.time() - start, 60)
        logger.info("Finished tuning %s: time %d minutes %d seconds", config.name, int(mins), int(secs))
        return True

    def tuning_models(self) -> None:
        """
        Description: Iterates over all configured models generated by _build_model_configs and attempts 
                     to run hyperparameter tuning for each.
        Args: None
        Input: Model configuration list.
        Output: Populates self.tuned_models and self.best_params dictionaries. Triggers error log if all fail.
        """
        for config in self._build_model_configs():
            self._run_search(config)

        if not self.tuned_models:
            logger.error("All models failed tuning! Check %s for details.", LOG_PATH)

    def _build_evaluation_artifacts(self) -> dict[str, dict]:
        """
        Description: Calculates evaluation metrics and generates physical artifacts (CSV classification reports 
                     and PNG confusion matrices) for all models that were successfully tuned.
        Args: None
        Input: self.X_test, self.y_test, and all fitted models in self.tuned_models.
        Output: A nested dictionary mapping model names to their respective metrics and saved artifact paths.
        """
        artifacts: dict[str, dict] = {}

        for name, model in self.tuned_models.items():
            y_pred = model.predict(self.X_test)
            self.tuned_predictions[name] = y_pred

            metrics = {
                "accuracy": accuracy_score(self.y_test, y_pred),
                "f1_score_weighted": f1_score(self.y_test, y_pred, average="weighted"),
                "precision_weighted": precision_score(self.y_test, y_pred, average="weighted", zero_division=0),
                "recall_weighted": recall_score(self.y_test, y_pred, average="weighted", zero_division=0),
            }
            logger.info(
                "%s -> F1-Score: %.4f | Accuracy: %.4f", name, metrics["f1_score_weighted"], metrics["accuracy"]
            )

            safe_name = name.replace(" ", "_")
            report_path = Path(f"evaluation_report_{safe_name}_tuning.csv")
            plot_path = Path(f"evaluation_plot_{safe_name}_tuning.png")

            report_dict = classification_report(
                self.y_test, y_pred, target_names=TARGET_NAMES, output_dict=True
            )
            pd.DataFrame(report_dict).transpose().to_csv(report_path, index=True)

            cm = confusion_matrix(self.y_test, y_pred)
            plt.figure(figsize=(6, 4))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Greens", xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES
            )
            plt.title(f"Confusion Matrix: {name}")
            plt.ylabel("Actual")
            plt.xlabel("Predicted")
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300)
            plt.close()

            artifacts[name] = {"metrics": metrics, "report_path": report_path, "plot_path": plot_path}

        return artifacts

    def _display_evaluation_results(self) -> None:
        """
        Display best parameters, evaluation metrics, classification report,
        and confusion matrix for each tuned model.
        """
        print("\n" + "=" * 100)
        print("HASIL EVALUASI MODEL")
        print("=" * 100)

        for name, model in self.tuned_models.items():
            y_pred = self.tuned_predictions[name]

            print(f"\n{'=' * 100}")
            print(f"{name}")
            print(f"{'=' * 100}")

            print("\nBest Parameters:")
            for param, value in self.best_params[name].items():
                print(f"  {param}: {value}")

            metrics = {
                "Accuracy": accuracy_score(self.y_test, y_pred),
                "Precision (Weighted)": precision_score(
                    self.y_test,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                ),
                "Recall (Weighted)": recall_score(
                    self.y_test,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                ),
                "F1 Score (Weighted)": f1_score(
                    self.y_test,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                ),
            }

            print("\nEvaluation Metrics:")
            for metric, value in metrics.items():
                print(f"  {metric:<22}: {value:.4f}")

            print("\nClassification Report:")
            print(
                classification_report(
                    self.y_test,
                    y_pred,
                    target_names=TARGET_NAMES,
                    zero_division=0,
                )
            )

            print("Confusion Matrix:")
            print(
                pd.DataFrame(
                    confusion_matrix(self.y_test, y_pred),
                    index=TARGET_NAMES,
                    columns=TARGET_NAMES,
                )
            )

    def _log_all_models_to(self, tracking_uri: str, experiment_name: str, run_prefix: str, artifacts: dict) -> None:
        """
        Description: Connects to a specified MLflow tracking server and pushes all logged metrics, generated 
                     artifacts, and serialized models under distinct run names. Uses cloudpickle to bypass 
                     strict serialization security audits for underlying C-based structures (like KDTree in KNN).
        Args:
            tracking_uri (str): The HTTP or local file URI for the target MLflow server.
            experiment_name (str): The overarching experiment tag within the MLflow interface.
            run_prefix (str): Prefix added to the model name to denote where it was logged (e.g., 'Local' or 'DagsHub').
            artifacts (dict): The dictionary compiled by _build_evaluation_artifacts containing metrics and paths.
        Input: Evaluated models, metric dictionaries, and generated physical files.
        Output: Network/local I/O commits to the MLflow tracking environment.
        """
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        for name, model in self.tuned_models.items():
            artifact = artifacts[name]
            try:
                with mlflow.start_run(run_name=f"{run_prefix}_{name}"):
                    mlflow.log_params(self.best_params[name])
                    mlflow.log_metrics(artifact["metrics"])
                    mlflow.log_artifact(str(artifact["report_path"]))
                    mlflow.log_artifact(str(artifact["plot_path"]))
                    mlflow.sklearn.log_model(
                        model, "model", serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
                    )
                logger.info("%s logged to %s.", name, run_prefix.upper())
            except Exception:
                logger.exception("%s logging for %s failed.", run_prefix.upper(), name)

    def evaluate_and_log_mlflow(self) -> None:
        """
        Description: Orchestrates the generation of evaluation artifacts and subsequently triggers 
                     the dual logging processes for both the Local instance and the DagsHub cloud.
        Args: None
        Input: Fully tuned models (self.tuned_models).
        Output: Finalizes script execution by exporting reports and uploading payload arrays to MLflow endpoints.
        """
        logger.info("=== Evaluation & MLflow Dual Logging (Local & DagsHub) ===")

        artifacts = self._build_evaluation_artifacts()

        self._log_all_models_to("http://127.0.0.1:5001/", "Diet_Health_Status_Skilled", "Local", artifacts)

        if self.dagshub_uri:
            self._log_all_models_to(self.dagshub_uri, "Diet_Health_Status_Advance", "DagsHub", artifacts)
        else:
            logger.warning("DagsHub URI not available — skipping DagsHub logging.")


if __name__ == "__main__":
    pipeline = DietModelPipeline()

    DAGSHUB_USERNAME = "noviardhana"
    DAGSHUB_REPO = "sml_noviardhana"

    pipeline.setup_credentials(DAGSHUB_USERNAME, DAGSHUB_REPO)
    pipeline.prepare_data()
    pipeline.tuning_models()
    pipeline.evaluate_and_log_mlflow()