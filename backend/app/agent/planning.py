"""Trusted request intent and execution plan for the Pipeline Agent."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


_FULL_RUN = re.compile(
    r"\b(run|start|execute|perform|complete)\b.{0,70}\b(full\s+pipeline|pipeline|analysis)\b"
    r"|\b(analyze|analyse)\b.{0,50}\b(dataset|data|file|table)\b"
    r"|\bautomate\b.{0,50}\bdataset\b",
    re.IGNORECASE,
)
_TRAIN = re.compile(
    r"\b(train|fit|build)\b.{0,60}\b(model|models|classifier|regressor|training)\b"
    r"|\b(train|fit|start|run|execute)\b.{0,40}\b(model\s+)?training\b"
    r"|\b(train|fit)\s+(a\s+)?(classification|regression)\b",
    re.IGNORECASE,
)
_EXPLANATION = re.compile(
    r"^\s*(what|why|how|explain|describe|tell me|can you explain|could you explain|"
    r"what does|how does|should i)\b",
    re.IGNORECASE,
)
_TARGET_CHANGE = re.compile(
    r"^\s*(set|change|select|choose|switch|use)\b.{0,45}\b(target|label|outcome)\b"
    r"|\bchange\b.{0,30}\b(target|label|outcome)\b",
    re.IGNORECASE,
)
_TARGET_VALUE_PATTERNS = (
    re.compile(
        r"\b(?:target|label|outcome)(?:\s+column)?\s*(?:is|=|to|as|:)\s*"
        r"[\"']?([A-Za-z_][A-Za-z0-9 .-]{0,99})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:use|choose|select)\s+[\"']?([A-Za-z_][A-Za-z0-9 .-]{0,99})"
        r"[\"']?\s+as\s+(?:the\s+)?(?:target|label|outcome)\b",
        re.IGNORECASE,
    ),
)
_PLAN = re.compile(r"\b(plan|planning|recommend a workflow|prepare a workflow)\b", re.IGNORECASE)
_PROFILE = re.compile(r"\b(profile|profiling|eda|explore|exploratory|inspect|summari[sz]e)\b", re.IGNORECASE)
_REPORT = re.compile(r"\b(report|pdf|notebook)\b", re.IGNORECASE)
_KAGGLE = re.compile(r"\b(download|load|import|use)\b.{0,50}\bkaggle\b", re.IGNORECASE)
_PREDICTION = re.compile(r"\b(predict|prediction|score this row|classify these records)\b", re.IGNORECASE)
_COMPARE_RESULTS = re.compile(
    r"\b(compare|show|summari[sz]e)\b.{0,50}\b(model\s+)?(results|metrics|scores)\b",
    re.IGNORECASE,
)

BASE_TOOLS = frozenset({"get_session_status", "inspect_dataset", "get_operations_log"})
TRAIN_TOOLS = frozenset({"train_models"})
FULL_PIPELINE_TOOLS = frozenset({"rerun_pipeline"})


@dataclass(frozen=True)
class RequestPlan:
    """A small in-memory plan; persisted AnalysisPlan stays the approval record."""

    intent: str
    mode: str
    session_id: str
    target_column: str | None
    task_type: str | None
    allowed_tools: frozenset[str]
    execution_order: tuple[str, ...]
    missing_information: tuple[str, ...]
    requires_expensive_execution: bool

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "mode": self.mode,
            "selected_session": self.session_id,
            "target_column": self.target_column,
            "task_type": self.task_type or "automatic Task 07 inference at training time",
            "allowed_tools": sorted(self.allowed_tools),
            "execution_order": list(self.execution_order),
            "missing_information": list(self.missing_information),
            "requires_expensive_execution": self.requires_expensive_execution,
        }


def prepare_request_plan(
    *,
    session_id: str,
    user_message: str,
    mode: str = "simple",
    analysis_options: dict[str, Any] | None = None,
    current_target: str | None = None,
) -> RequestPlan:
    """Restrict model tool access using the user's actual request and trusted options."""
    text = user_message.strip()
    options = analysis_options or {}
    asks_for_explanation = bool(_EXPLANATION.search(text))
    explicit_target_change = bool(_TARGET_CHANGE.search(text))
    explicit_profile = bool(_PROFILE.search(text))
    explicit_report = bool(_REPORT.search(text))
    explicit_kaggle = bool(_KAGGLE.search(text))
    explicit_plan = bool(_PLAN.search(text))
    explicit_target = None
    for pattern in _TARGET_VALUE_PATTERNS:
        match = pattern.search(text)
        if match:
            explicit_target = match.group(1).strip().strip(" .,:;'\"")
            break
    if explicit_target:
        for separator in (" for training", " for the model", " so the model", " before training"):
            explicit_target = re.split(separator, explicit_target, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    explicit_target = options.get("target_column") or explicit_target

    run_requested = not asks_for_explanation and bool(_FULL_RUN.search(text))
    train_requested = not asks_for_explanation and bool(_TRAIN.search(text))
    if explicit_target_change:
        intent = "change_target"
        if explicit_target:
            allowed = set(BASE_TOOLS | {"set_target_column"})
            order = ("inspect_dataset", "set_target_column")
        else:
            allowed = set(BASE_TOOLS)
            order = ("inspect_dataset",)
        expensive = False
    elif explicit_kaggle:
        intent = "load_kaggle_dataset"
        allowed = set(BASE_TOOLS | {"load_from_kaggle"})
        order = ("load_from_kaggle", "inspect_dataset")
        expensive = True
    elif run_requested and explicit_report:
        intent = "run_full_analysis_and_report"
        allowed = set(BASE_TOOLS | {"rerun_pipeline"})
        order = ("inspect_dataset", "rerun_pipeline")
        expensive = True
    elif run_requested:
        intent = "run_full_analysis"
        allowed = set(BASE_TOOLS | {"rerun_pipeline"})
        order = ("inspect_dataset", "rerun_pipeline")
        expensive = True
    elif train_requested:
        intent = "train_models"
        allowed = set(BASE_TOOLS | TRAIN_TOOLS)
        order = ("inspect_dataset", "train_models")
        expensive = True
    elif explicit_plan:
        intent = "prepare_plan"
        allowed = set(BASE_TOOLS | {"propose_plan"})
        order = ("inspect_dataset", "propose_plan")
        expensive = False
    elif explicit_profile:
        intent = "profile_dataset"
        allowed = set(BASE_TOOLS | {"run_eda"})
        order = ("inspect_dataset", "run_eda")
        expensive = False
    elif explicit_report:
        intent = "generate_report"
        allowed = set(BASE_TOOLS | {"generate_report"})
        order = ("get_session_status", "generate_report")
        expensive = True
    elif _COMPARE_RESULTS.search(text) and not asks_for_explanation:
        intent = "compare_existing_results"
        allowed = set(BASE_TOOLS)
        order = ("get_session_status",)
        expensive = False
    elif _PREDICTION.search(text) and not asks_for_explanation:
        intent = "request_prediction"
        allowed = set(BASE_TOOLS)
        order = ("get_session_status",)
        expensive = False
    else:
        intent = "answer_or_retrieve"
        allowed = set(BASE_TOOLS)
        order = ("get_session_status", "inspect_dataset")
        expensive = False

    requested_target = explicit_target or None
    requested_task = options.get("task_type") or None
    target = requested_target or current_target
    missing: list[str] = []
    if intent == "prepare_plan" and not target:
        allowed.discard("propose_plan")
        order = ("inspect_dataset",)
    if intent == "change_target" and not requested_target:
        missing.append("exact target column name")
    if intent in {"run_full_analysis", "run_full_analysis_and_report", "train_models", "prepare_plan"} and not target:
        missing.append("target column must be selected from Task 07 suggestions before training")
    if intent == "request_prediction":
        missing.append("prediction execution is not available as a Pipeline Agent tool")

    return RequestPlan(
        intent=intent,
        mode=mode,
        session_id=session_id,
        target_column=target,
        task_type=requested_task,
        allowed_tools=frozenset(allowed),
        execution_order=order,
        missing_information=tuple(missing),
        requires_expensive_execution=expensive,
    )
