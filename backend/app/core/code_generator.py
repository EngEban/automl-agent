"""Generate portable, reproducible Jupyter notebooks for trained models."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path, PurePosixPath

from app.config import settings

logger = logging.getLogger(__name__)


def _source_lines(source: str) -> list[str]:
    """Return notebook source in Jupyter's preferred line-oriented format."""
    return source.splitlines(keepends=True)


def _quoted(value: str) -> str:
    """Serialize a value as a safe Python string literal."""
    return json.dumps(str(value), ensure_ascii=False)


def _portable_filename(dataset_filename: str) -> str:
    """Return a safe basename that preserves the uploaded filename where possible."""
    raw_filename = str(dataset_filename or "").replace("\x00", "")
    filename = PurePosixPath(raw_filename.replace("\\", "/")).name
    filename = re.sub(r'[\x00-\x1f\x7f<>:"|?*]', "_", filename)
    filename = filename.rstrip(" .")
    return filename if filename not in {"", ".", ".."} else "data.csv"


def _model_code(task_type: str, best_model_name: str, tuning_params: dict | None = None) -> str:
    """Generate the Python code for the winning model with tuned params."""
    if task_type == "classification":
        models = {
            "GradientBoosting": "GradientBoostingClassifier",
            "RandomForest": "RandomForestClassifier",
            "DecisionTree": "DecisionTreeClassifier",
            "HistGradientBoosting": "HistGradientBoostingClassifier",
            "LogisticRegression": "LogisticRegression",
        }
        base_model = models.get(best_model_name, "LogisticRegression")
    else:
        models = {
            "GradientBoosting": "GradientBoostingRegressor",
            "RandomForest": "RandomForestRegressor",
            "DecisionTree": "DecisionTreeRegressor",
            "HistGradientBoosting": "HistGradientBoostingRegressor",
            "Ridge": "Ridge",
            "LinearRegression": "LinearRegression",
        }
        base_model = models.get(best_model_name, "LinearRegression")

    # Default parameters based on model type
    params = {}
    if "RandomForest" in base_model:
        params["n_estimators"] = 100
        params["n_jobs"] = -1
    elif "Logistic" in base_model:
        params["max_iter"] = 1000
    
    if "Logistic" not in base_model and "Linear" not in base_model and "Ridge" not in base_model:
        params["random_state"] = 42

    if tuning_params:
        params.update(tuning_params)

    param_str = ", ".join(f"{k}={v!r}" for k, v in params.items())
    return f"{base_model}({param_str})"


def _evaluation_code(task_type: str) -> str:
    if task_type == "classification":
        return """# Classification metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 score: {f1:.4f}")
"""
    return """# Regression metrics
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"R2 score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
"""


def _load_dataset_code(expected_filename: str) -> str:
    """Build the Colab-aware data-loading cell."""
    expected_literal = _quoted(expected_filename)
    return f"""# 1. Load the dataset
expected_filename = {expected_literal}

try:
    from google.colab import files
    print(f"Please upload {{expected_filename!r}}:")
    uploaded = files.upload()
    if not uploaded:
        raise FileNotFoundError("No file was uploaded.")
    if expected_filename not in uploaded:
        uploaded_names = ", ".join(repr(name) for name in uploaded)
        raise FileNotFoundError(
            f"The uploaded filename does not match {{expected_filename!r}}. "
            f"Uploaded file(s): {{uploaded_names}}."
        )
    file_path = expected_filename
except ImportError:
    file_path = expected_filename

if not os.path.isfile(file_path):
    raise FileNotFoundError(
        f"Dataset file {{expected_filename!r}} was not found in the notebook's "
        f"current directory ({{os.getcwd()!r}}). "
        "If you are running this locally, make sure to download the dataset "
        "from the web app and place it in the EXACT SAME folder as this notebook."
    )

try:
    df = pd.read_csv(file_path)
except pd.errors.EmptyDataError as exc:
    raise ValueError("The dataset file is empty.") from exc

df.columns = [str(column).strip().replace(" ", "_") for column in df.columns]
df = df.dropna(axis=1, how="all")

if df.empty:
    raise ValueError("The dataset contains no rows.")
if df.shape[1] == 0:
    raise ValueError("The dataset has no usable columns.")

print(f"Dataset shape: {{df.shape}}")
display(df.head())
"""


def generate_notebook(
    session_id: str,
    dataset_filename: str,
    target_column: str,
    task_type: str,
    best_model_name: str,
    tuning_params: dict | None = None,
) -> Path:
    """Generate a portable .ipynb file and return its path."""
    expected_filename = _portable_filename(dataset_filename)
    task_type = task_type if task_type in {"classification", "regression"} else "regression"
    
    model_code = _model_code(task_type, best_model_name, tuning_params)

    imports = """import os
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

print("Libraries imported successfully.")
"""

    dataset_instructions = f"""## Dataset file

If you are running this notebook in Google Colab, run the loading cell below and upload `{expected_filename}` when prompted.

If you are running this locally, make sure to download the dataset from the web app and place it in the EXACT SAME folder as this notebook.
The filename must be exactly `{expected_filename}`.
"""

    load_dataset = _load_dataset_code(expected_filename)

    split_code = (
        """class_count = int(y.nunique())
class_sizes = y.value_counts()
if class_count < 2:
    raise ValueError("Classification requires at least two target classes.")
if int(class_sizes.min()) < 3:
    raise ValueError("Each target class needs at least three rows for a stratified train/test split.")

test_count = max(2, int(np.ceil(len(X) * 0.2)), class_count)
if len(X) - test_count < class_count:
    raise ValueError("Too few rows to place every class in both train and test sets.")
stratify = y
"""
        if task_type == "classification"
        else """test_count = max(2, int(np.ceil(len(X) * 0.2)))
stratify = None
"""
    )

    prepare_data = f"""# 2. Prepare features and target
target_col = {_quoted(target_column)}
if target_col not in df.columns:
    raise KeyError(f"Target column {{target_col!r}} not found.")

df = df.dropna(subset=[target_col]).drop_duplicates().reset_index(drop=True)
if len(df) < 2:
    raise ValueError("At least two rows with non-missing target values are required.")

X = df.drop(columns=[target_col])
numeric_columns = X.select_dtypes(include="number").columns
if len(numeric_columns):
    X.loc[:, numeric_columns] = X.loc[:, numeric_columns].replace([np.inf, -np.inf], np.nan)
if X.shape[1] == 0:
    raise ValueError("No usable feature columns remain.")

numeric_features = X.select_dtypes(include="number").columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

y = df[target_col]

{split_code}
X_train, X_test, y_train_raw, y_test_raw = train_test_split(
    X, y, test_size=test_count, random_state=42, stratify=stratify,
)

# Fit target-label mapping on the training labels after the split.
target_encoder = None
if y_train_raw.dtype == "object" or y_train_raw.dtype.name in {{"category", "string"}}:
    target_encoder = LabelEncoder()
    y_train = pd.Series(
        target_encoder.fit_transform(y_train_raw), name=target_col, index=y_train_raw.index
    )
    y_test = pd.Series(
        target_encoder.transform(y_test_raw), name=target_col, index=y_test_raw.index
    )
else:
    y_train, y_test = y_train_raw, y_test_raw

print(f"Train set: {{X_train.shape}}, Test set: {{X_test.shape}}")
"""

    preprocessing = """# 3. Build the preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=0.05, max_categories=50)),
])

transformers = []
if numeric_features: transformers.append(("num", numeric_transformer, numeric_features))
if categorical_features: transformers.append(("cat", categorical_transformer, categorical_features))
if not transformers: raise ValueError("No supported feature columns remain.")

preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
print("Preprocessor built successfully.")
"""

    training = f"""# 4. Define and train the winning model
model = {model_code}
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
pipeline.fit(X_train, y_train)
print("Model trained successfully.")
"""

    evaluation = """# 5. Evaluate the model
y_pred = pipeline.predict(X_test)
""" + _evaluation_code(task_type)

    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": _source_lines(f"# AutoML Agent: Reproducible Pipeline\n\nTarget: `{target_column}`\n\nTask: `{task_type}`\n\nModel: `{best_model_name}`"),
            },
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": _source_lines(imports)},
            {"cell_type": "markdown", "metadata": {}, "source": _source_lines(dataset_instructions)},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": _source_lines(load_dataset)},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": _source_lines(prepare_data)},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": _source_lines(preprocessing)},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": _source_lines(training)},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": _source_lines(evaluation)},
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py", "mimetype": "text/x-python", "name": "python", "nbconvert_exporter": "python", "pygments_lexer": "ipython3", "version": "3.11.9"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    output_path = settings.STORAGE_DIR / "sessions" / session_id / "analysis_pipeline.ipynb"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open("w", encoding="utf-8") as notebook_file:
        json.dump(notebook, notebook_file, ensure_ascii=False, indent=2)
        notebook_file.write("\n")

    logger.info("Generated Jupyter Notebook: %s", output_path)
    return output_path
