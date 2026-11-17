# Agent workflows

## Pipeline Agent

The Pipeline Agent lives in `app.agents.pipeline_agent` and uses LiteLLM with the independent `LLM_*` settings. Backend imports use the canonical `app.*` namespace. It is invoked asynchronously through the chat and run APIs: the request creates an AnalysisRun, Celery executes it through `app.worker.worker:celery_app`, and persisted events are available through SSE and event history.

Simple mode accepts a user message without analysis options. Advanced mode also accepts a target column and task type. Tool schemas are allowlisted, arguments are validated before execution, and per-run iteration and tool-call limits bound the loop. Approval-required plans are not run until the owner approves them.

The registered tools are:

- Read state: get_session_status, inspect_dataset, get_operations_log.
- Analyze and prepare: run_eda, run_preprocessing, set_target_column.
- Train and retrieve results: train_models.
- Orchestrate: propose_plan, rerun_pipeline, generate_report, load_from_kaggle.

The executor calls the shared ML services and Model Registry. Dataset sampling, training-only task inference, train-fold preprocessing, independent holdout evaluation, memory limits, target encoding, prediction artifact compatibility, and ownership checks remain enforced in their respective services.

Tool failures are returned as structured results. A completed tool result is preserved if a later model response or operation fails; run status and events distinguish success, partial completion, failure, cancellation, and approval/input waits.

## Report Agent

The Report Agent lives in `app.agents.report_agent` and remains independent from the Pipeline Agent. It builds a compact fact payload from completed results, runs a bounded LangGraph generation and validation workflow, and renders HTML/PDF artifacts. It reads `REPORT_LLM_*` settings; changing Pipeline Agent settings does not select its provider or model.

When `REPORT_AGENT_ENABLED` is false, `app.agents.report_agent.service` retains the legacy `app.core.report_generator` fallback. Keep this fallback and compatible artifact loading paths until downstream consumers have been checked.

## Verification

Automated Report Agent and Pipeline Agent tests mock provider calls. The scripts scripts/smoke_test_ollama_pipeline.py and scripts/smoke_test_report_agent.py make live requests when intentionally run and must not be called from pytest. The Report Agent script can contact Groq; the Pipeline script uses the configured Ollama endpoint. Do not infer live provider success from mocked tests.
