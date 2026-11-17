from __future__ import annotations

import importlib
import sys

from app.worker.worker import celery_app


def test_production_tasks_register_with_the_production_worker() -> None:
    worker_module = importlib.import_module("app.worker.worker")
    assert worker_module.celery_app is celery_app
    assert "app.tasks.analyze" in celery_app.conf.include
    assert "app.tasks.agent_tasks" in celery_app.conf.include

    celery_app.loader.import_default_modules()

    analyze_module = importlib.import_module("app.tasks.analyze")
    agent_module = importlib.import_module("app.tasks.agent_tasks")

    assert analyze_module.analyze_dataset.app is celery_app
    assert agent_module.run_agent_task.app is celery_app
    assert analyze_module.analyze_dataset.name == "analyze_dataset"
    assert agent_module.run_agent_task.name == "run_agent_task"

    for task_name in ("analyze_dataset", "run_agent_task"):
        registered_task = celery_app.tasks[task_name]
        assert registered_task.app is celery_app
        assert registered_task.name == task_name

    first_registry = {
        name: celery_app.tasks[name]
        for name in ("analyze_dataset", "run_agent_task")
    }
    for module_name in (
        "app.worker.worker",
        "app.tasks.analyze",
        "app.tasks.agent_tasks",
    ):
        assert importlib.import_module(module_name) is sys.modules[module_name]

    celery_app.loader.import_default_modules()

    assert celery_app.tasks["analyze_dataset"] is first_registry["analyze_dataset"]
    assert celery_app.tasks["run_agent_task"] is first_registry["run_agent_task"]
    assert not any(
        module_name == "backend.app" or module_name.startswith("backend.app.")
        for module_name in sys.modules
    )

    assert "run_agent_task" in celery_app.tasks
