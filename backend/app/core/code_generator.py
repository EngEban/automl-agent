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
    """Return a safe basename that preserves the uploaded filename where possible.

    The notebook is distributed independently from the server, so it must never
    embed an absolute path or a path containing parent-directory components.
    """
    raw_filename = str(dataset_filename or "").replace("\x00", "")
    filename = PurePosixPath(raw_filename.replace("\\", "/")).name

    # Remove characters that are invalid on common local filesystems, as well as
    # control characters that could make the generated notebook ambiguous.
    filename = re.sub(r'[\x00-\x1f\x7f<>:"|?*]', "_", filename)
    filename = filename.rstrip(" .")

    return filename if filename not in {"", ".", ".."} else "data.csv"


def _model_code(task_type: str, best_model_name: str) -> str:
    if task_type == "classification":
        models = {
            "GradientBoosting": "GradientBoostingClassifier(random_state=42)",
            "RandomForest": (
                "RandomForestClassifier(n_estimators=100, "
                "random_state=42, n_jobs=-1)"
            ),
            "DecisionTree": "DecisionTreeClassifier(random_state=42)",
            "LogisticRegression": (
                "LogisticRegression(max_iter=1000, random_state=42)"
            ),
        }
        return models.get(
            best_model_name,
            "LogisticRegression(max_iter=1000, random_state=42)",
        )

    models = {
        "GradientBoosting": "GradientBoostingRegressor(random_state=42)",
        "RandomForest": (
            "RandomForestRegressor(n_estimators=100, "
            "random_state=42, n_jobs=-1)"
        ),
        "DecisionTree": "DecisionTreeRegressor(random_state=42)",
        "Ridge": "Ridge(random_state=42)",
        "LinearRegression": "LinearRegression()",
    }
    return models.get(best_model_name, "LinearRegression()")


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
        raise FileNotFoundError(
            "No file was uploaded. Please upload "
            + repr(expected_filename)
            + " and run this cell again."
        )

    if expected_filename not in uploaded:
        uploaded_names = ", ".join(repr(name) for name in uploaded)
        raise FileNotFoundError(
            "The uploaded filename does not match the expected dataset filename "
            + repr(expected_filename)
            + ". Uploaded file(s): "
            + uploaded_names
            + ". Please upload the original file with the expected name."
        )

    file_path = expected_filename
except ImportError:
    # In a local Jupyter environment, the dataset must be beside this notebook.
    file_path = expected_filename

if not os.path.isfile(file_path):
    raise FileNotFoundError(
        f"Dataset file {{expected_filename!r}} was not found in the notebook's "
        f"current directory ({{os.getcwd()!r}}). "
        "If you are running this locally, make sure to download the dataset "
        "from the web app and place it in the EXACT SAME folder as this notebook. "
        "Place a file named "
        + repr(expected_filename)
        + " there, then run this cell again."
    )

try:
    df = pd.read_csv(file_path)
except pd.errors.EmptyDataError as exc:
    raise ValueError(
        "The dataset file is empty or has no readable column headers. "
        "Please download a valid dataset from the web app."
    ) from exc
df.columns = [str(column).strip().replace(" ", "_") for column in df.columns]
df = df.dropna(axis=1, how="all")
if df.empty:
    raise ValueError(
        "The dataset contains no rows. Please download and upload a non-empty dataset."
    )
if df.shape[1] == 0:
    raise ValueError(
        "The dataset has no usable columns after removing empty columns."
    )

print(f"Dataset shape: {{df.shape}}")
display(df.head())
"""


def generate_notebook(
    session_id: str,
    dataset_filename: str,
    target_column: str,
    task_type: str,
    best_model_name: str,
) -> Path:
    """Generate a portable .ipynb file and return its path."""
    expected_filename = _portable_filename(dataset_filename)
    task_type = (
        task_type
        if task_type in {"classification", "regression"}
        else "regression"
    )
    model_code = _model_code(task_type, best_model_name)

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
    raise ValueError(
        "Classification requires at least two target classes after removing "
        "rows with missing targets."
    )
if int(class_sizes.min()) < 2:
    raise ValueError(
        "Each target class needs at least two rows to create a reliable "
        "train/test split. Add more rows for the least frequent class."
    )

test_count = max(1, int(np.ceil(len(X) * 0.2)), class_count)
if len(X) - test_count < class_count:
    raise ValueError(
        "There are too few rows to place every target class in both the "
        "training and test sets. Add more rows per class."
    )
stratify = y
"""
        if task_type == "classification"
        else """test_count = max(1, int(np.ceil(len(X) * 0.2)))
stratify = None
"""
    )

    prepare_data = f"""# 2. Prepare features and target
target_col = {_quoted(target_column)}
if target_col not in df.columns:
    raise KeyError(
        f"Target column {{target_col!r}} was not found after normalizing CSV "
        f"column names. Available columns: {{df.columns.tolist()!r}}."
    )

df = df.dropna(subset=[target_col]).drop_duplicates().reset_index(drop=True)
if len(df) < 2:
    raise ValueError(
        "At least two rows with non-missing target values are required."
    )

X = df.drop(columns=[target_col])
numeric_columns = X.select_dtypes(include="number").columns
if len(numeric_columns):
    X.loc[:, numeric_columns] = X.loc[:, numeric_columns].replace(
        [np.inf, -np.inf], np.nan
    )
X = X.dropna(axis=1, how="all")
if X.shape[1] == 0:
    raise ValueError(
        "No usable feature columns remain after removing all-null columns."
    )

numeric_features = X.select_dtypes(include="number").columns.tolist()
categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()
unsupported_features = [
    column for column in X.columns
    if column not in numeric_features and column not in categorical_features
]
for column in unsupported_features:
    non_missing = X[column].notna()
    X[column] = X[column].astype("string").astype("object")
    X.loc[~non_missing, column] = None
categorical_features.extend(unsupported_features)

y = df[target_col]

if y.dtype == "object" or y.dtype.name in {{"category", "string"}}:
    target_encoder = LabelEncoder()
    y = pd.Series(
        target_encoder.fit_transform(y),
        name=target_col,
        index=y.index,
    )

{split_code}
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_count,
    random_state=42,
    stratify=stratify,
)

print(f"Train set: {{X_train.shape}}, Test set: {{X_test.shape}}")
"""

    preprocessing = """# 3. Build the preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )),
])

transformers = []
if numeric_features:
    transformers.append(("num", numeric_transformer, numeric_features))
if categorical_features:
    transformers.append(("cat", categorical_transformer, categorical_features))
if not transformers:
    raise ValueError("No supported feature columns remain for model training.")

preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
print("Preprocessor built successfully.")
"""

    training = f"""# 4. Define and train the winning model
# Model selected by AutoML Agent: {best_model_name}
model = {model_code}

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model),
])
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
                "source": _source_lines(
                    """# AutoML Agent: Reproducible Analysis Pipeline

This notebook was automatically generated by AutoML Agent.

- Target column: {target}
- Task type: {task}
- Winning model: {model}

Run the cells sequentially to reproduce the preprocessing, training, and evaluation steps.
""".format(
                        target=target_column,
                        task=task_type,
                        model=best_model_name,
                    )
                ),
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": _source_lines(imports),
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": _source_lines(dataset_instructions),
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": _source_lines(load_dataset),
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": _source_lines(prepare_data),
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": _source_lines(preprocessing),
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": _source_lines(training),
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": _source_lines(evaluation),
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3,
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    output_path = (
        settings.STORAGE_DIR
        / "sessions"
        / session_id
        / "analysis_pipeline.ipynb"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as notebook_file:
        json.dump(
            notebook,
            notebook_file,
            ensure_ascii=False,
            indent=2,
        )
        notebook_file.write("\n")

    logger.info("Generated Jupyter Notebook: %s", output_path)
    return output_path
