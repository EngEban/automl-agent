"""
Tool definitions for the agent (OpenAI function-calling schema).
The schema is what the LLM sees; the implementation lives in executor.py.
"""
from __future__ import annotations

from typing import Any


def get_tool_schemas() -> list[dict[str, Any]]:
    """Return the list of tool schemas advertised to the LLM."""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_session_status",
                "description": (
                    "Return the current state of the session: whether EDA, "
                    "preprocessing, training, and report generation have run, "
                    "the detected task type, target column, and available artifacts."
                ),
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "inspect_dataset",
                "description": (
                    "Return a structural view of the dataset: shape, columns, "
                    "dtypes, missing counts, and the first 5 rows."
                ),
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_eda",
                "description": (
                    "Run exploratory data analysis on the dataset. Returns a "
                    "summary dict; figures are attached to the session and shown "
                    "in the final report."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_column": {
                            "type": "string",
                            "description": "Optional target column to focus on.",
                        },
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_preprocessing",
                "description": (
                    "Build and apply a preprocessing pipeline (missing-value "
                    "imputation, encoding, scaling). Returns the list of "
                    "operations applied."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_column": {
                            "type": "string",
                            "description": "Column to predict.",
                        },
                        "drop_duplicates": {
                            "type": "boolean",
                            "description": "Whether to drop duplicate rows (default true).",
                        },
                    },
                    "required": ["target_column"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "train_models",
                "description": (
                    "Train and evaluate candidate models on the preprocessed data. "
                    "Returns per-model metrics and the winning model name."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_type": {
                            "type": "string",
                            "enum": ["classification", "regression", "auto"],
                            "description": "Override task detection (default 'auto').",
                        },
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "generate_report",
                "description": (
                    "Generate the English PDF report AND the reproducible Jupyter Notebook (.ipynb) "
                    "containing EDA figures, operations log, model comparison, and the source code. "
                    "Requires all previous steps to have run."
                ),
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_operations_log",
                "description": (
                    "Return every transformation applied to the dataset so far "
                    "(missing-value handling, encoding, scaling, dropped rows, ...)."
                ),
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "set_target_column",
                "description": (
                    "Change the target column for this session. Requires rerunning "
                    "the pipeline afterwards."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "column": {
                            "type": "string",
                            "description": "Name of the target column.",
                        },
                    },
                    "required": ["column"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "rerun_pipeline",
                "description": (
                    "Run the full pipeline end-to-end (EDA -> preprocessing -> "
                    "training -> report -> notebook) with the session's current settings."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_column": {
                            "type": "string",
                            "description": "Optional new target column.",
                        },
                    },
                    "required": [],
                },
            },
        },
    ]