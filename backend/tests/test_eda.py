from __future__ import annotations

import pandas as pd

from app.core.eda import run_eda


def test_eda_uses_the_resolved_task_type_for_numeric_classification_target():
    frame = pd.DataFrame({
        "measurement": [float(index) for index in range(30)],
        "class_code": [0, 1] * 15,
    })

    result = run_eda(frame, target="class_code", task_type="classification")

    assert result["summary"]["task_type"] == "classification"
    target = result["summary"]["profile"]["target_info"]
    assert target["class_count"] == 2
    assert target["task_type"] == "classification"
    assert target["is_imbalanced"] is False
    assert result["figures"].get("target")
