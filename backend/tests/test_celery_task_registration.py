from app.worker import celery_app


def test_run_agent_task_is_registered_with_worker() -> None:
    celery_app.loader.import_default_modules()

    assert "run_agent_task" in celery_app.tasks
