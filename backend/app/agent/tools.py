"""Allowlisted Pipeline Agent tools and argument validation."""
from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping
from typing import Any


class ToolValidationError(ValueError):
    """A tool call does not match the registered tool contract."""


class UnknownToolError(ToolValidationError):
    """The model requested a tool outside the allowlisted registry."""


_NO_ARGS = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}

TOOL_SCHEMAS: tuple[dict[str, Any], ...] = (
    {
        "type": "function",
        "function": {
            "name": "get_session_status",
            "description": (
                "Read this session's current pipeline state, target, task type, "
                "latest model metrics, and available artifact types. Does not "
                "change data or start work."
            ),
            "parameters": copy.deepcopy(_NO_ARGS),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_dataset",
            "description": (
                "Profile the selected session's dataset: shape, columns, data "
                "types, missing values, cardinality, identifiers, dates, "
                "quality warnings, and centralized target suggestions. Sample "
                "values are excluded unless explicitly requested."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "include_sample_rows": {
                        "type": "boolean",
                        "description": "Return at most five sample rows. Use only when the user asks to inspect example values.",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_eda",
            "description": (
                "Run descriptive exploratory analysis and save its summary and "
                "figures. Does not fit training preprocessing or train models. "
                "Use only when the user requests profiling or exploratory analysis."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target_column": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Optional existing target column to focus the descriptive analysis on.",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_preprocessing",
            "description": (
                "Prepare and save the existing AutoML preprocessing blueprint "
                "for a validated target. Learned imputation and encoding values "
                "are fit only within training folds and the final training fit."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target_column": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Existing column to predict.",
                    },
                },
                "required": ["target_column"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "train_models",
            "description": (
                "Train the supported candidate models for this session and "
                "return cross-validation selection metrics plus an independent "
                "holdout evaluation. Requires a validated target and trusted "
                "AnalysisRun context. Task type may be classification, "
                "regression, or auto (training-only inference)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_type": {
                        "type": "string",
                        "enum": ["classification", "regression", "auto"],
                        "description": "Optional task type; auto uses Task 07 inference inside the training workflow.",
                    },
                    "target_column": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Optional exact target selected by the user or trusted Advanced Mode request.",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_report",
            "description": (
                "Generate a report from an already completed analysis. Does not "
                "train a model. The result keeps report status separate from "
                "analysis status."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "report_type": {
                        "type": "string",
                        "enum": [
                            "machine_learning_analysis", "data_analysis",
                            "model_comparison", "technical_report",
                            "executive_summary",
                        ],
                    },
                    "language": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Report language; defaults to English.",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_operations_log",
            "description": "Read the transformations recorded for this session without changing the dataset.",
            "parameters": copy.deepcopy(_NO_ARGS),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_target_column",
            "description": (
                "Set an existing, validated target column explicitly. This "
                "changes session settings but does not start training."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Exact existing column name chosen by the user.",
                    },
                },
                "required": ["column"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "propose_plan",
            "description": (
                "Save a target and task type plan for user approval. Does not "
                "train. Use only when the user asks to prepare or approve a plan."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target_column": {"type": "string", "minLength": 1},
                    "task_type": {
                        "type": "string",
                        "enum": ["classification", "regression", "auto"],
                        "description": "Use auto to infer the task through Task 07 on training-only rows.",
                    },
                    "reasoning": {"type": "string", "minLength": 1},
                },
                "required": ["target_column", "task_type", "reasoning"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rerun_pipeline",
            "description": (
                "Run the established end-to-end flow in order: EDA, preprocessing, "
                "training with Task 06 leakage protections, then report and "
                "notebook generation. Use only after an explicit user request "
                "to run the full analysis. Requires trusted AnalysisRun context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target_column": {"type": "string", "minLength": 1},
                    "task_type": {
                        "type": "string",
                        "enum": ["classification", "regression", "auto"],
                    },
                    "report_type": {
                        "type": "string",
                        "enum": [
                            "machine_learning_analysis", "data_analysis",
                            "model_comparison", "technical_report",
                            "executive_summary",
                        ],
                    },
                    "language": {"type": "string", "minLength": 1},
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_from_kaggle",
            "description": (
                "Download the user-requested Kaggle dataset into this session. "
                "Requires Kaggle credentials and replaces the session's current "
                "dataset source; call only after the user explicitly requests it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "kaggle_ref": {
                        "type": "string",
                        "minLength": 3,
                        "description": "Kaggle dataset reference in owner/dataset-name format.",
                    },
                },
                "required": ["kaggle_ref"],
                "additionalProperties": False,
            },
        },
    },
)

_TOOLS_BY_NAME = {
    item["function"]["name"]: item
    for item in TOOL_SCHEMAS
}
_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def get_tool_schemas(allowed_names: set[str] | None = None) -> list[dict[str, Any]]:
    """Return copies of the registered schemas, optionally restricted by intent."""
    return [
        copy.deepcopy(tool)
        for tool in TOOL_SCHEMAS
        if allowed_names is None or tool["function"]["name"] in allowed_names
    ]


def validate_tool_arguments(tool_name: str, arguments: Any) -> dict[str, Any]:
    """Validate a decoded argument object against the allowlisted JSON schema."""
    tool = _TOOLS_BY_NAME.get(tool_name)
    if tool is None:
        raise UnknownToolError(f"Unknown tool '{tool_name}'.")
    if not isinstance(arguments, Mapping):
        raise ToolValidationError("Tool arguments must be a JSON object.")

    schema = tool["function"]["parameters"]
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    extra = set(arguments) - set(properties)
    if extra:
        names = ", ".join(sorted(str(key) for key in extra))
        raise ToolValidationError(f"Unsupported argument(s) for '{tool_name}': {names}.")
    missing = [name for name in required if name not in arguments]
    if missing:
        raise ToolValidationError(
            f"Missing required argument(s) for '{tool_name}': {', '.join(missing)}."
        )

    for name, value in arguments.items():
        if not isinstance(name, str) or name not in properties:
            raise ToolValidationError("Tool argument names must match the registered schema.")
        definition = properties[name]
        expected = definition.get("type")
        valid_type = {
            "string": lambda item: isinstance(item, str),
            "boolean": lambda item: isinstance(item, bool),
            "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
            "number": lambda item: (
                isinstance(item, (int, float))
                and not isinstance(item, bool)
                and math.isfinite(float(item))
            ),
            "object": lambda item: isinstance(item, Mapping),
            "array": lambda item: isinstance(item, list),
        }.get(expected)
        if valid_type is None or not valid_type(value):
            raise ToolValidationError(
                f"Argument '{name}' for '{tool_name}' must be {expected or 'valid'}."
            )
        if expected == "string" and len(value) < int(definition.get("minLength", 0)):
            raise ToolValidationError(f"Argument '{name}' for '{tool_name}' cannot be empty.")
        if "enum" in definition and value not in definition["enum"]:
            choices = ", ".join(definition["enum"])
            raise ToolValidationError(
                f"Argument '{name}' for '{tool_name}' must be one of: {choices}."
            )

    return dict(arguments)


def validate_tool_registry() -> None:
    """Raise when a published tool definition is malformed or duplicated."""
    names: set[str] = set()
    for tool in TOOL_SCHEMAS:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise ValueError("Every Agent tool must use a function schema.")
        function = tool.get("function")
        if not isinstance(function, dict):
            raise ValueError("Every Agent tool must define a function object.")
        name = function.get("name")
        if not isinstance(name, str) or not _NAME_PATTERN.fullmatch(name) or name in names:
            raise ValueError("Agent tool names must be unique and match the supported identifier format.")
        names.add(name)
        parameters = function.get("parameters")
        if (
            not isinstance(parameters, dict)
            or parameters.get("type") != "object"
            or not isinstance(parameters.get("properties"), dict)
            or not isinstance(parameters.get("required"), list)
            or parameters.get("additionalProperties") is not False
        ):
            raise ValueError(f"Tool '{name}' must have a closed object argument schema.")
        if not set(parameters["required"]) <= set(parameters["properties"]):
            raise ValueError(f"Tool '{name}' requires an argument absent from its schema.")
        if not isinstance(function.get("description"), str) or not function["description"].strip():
            raise ValueError(f"Tool '{name}' needs a useful description.")


validate_tool_registry()
