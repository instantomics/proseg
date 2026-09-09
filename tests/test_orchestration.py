from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1]


def _entrypoint():
    path = ROOT / "run_reference.py"
    spec = importlib.util.spec_from_file_location("proseg_run_reference", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeTools:
    def __init__(self, wait_statuses: list[str], final_status: str) -> None:
        self.wait_statuses = iter(wait_statuses)
        self.final_status = final_status
        self.calls: list[tuple[str, dict[str, object]]] = []

    def call(self, name: str, arguments: dict[str, object]) -> dict[str, object]:
        self.calls.append((name, arguments))
        if name == "freeze_candidate":
            return {"ok": True}
        if name == "validate_model":
            return {"valid": True}
        if name == "start_evaluation":
            return {"job_id": "job-1", "status": "queued"}
        if name == "wait_job":
            return {"job_id": "job-1", "status": next(self.wait_statuses)}
        if name == "inspect_job":
            return {"job_id": "job-1", "status": self.final_status}
        raise AssertionError(f"unexpected tool: {name}")


def test_entrypoint_uses_the_ordinary_candidate_lifecycle() -> None:
    tools = FakeTools(["running", "succeeded"], "succeeded")

    result = _entrypoint().run(tools, SimpleNamespace(reference_id="proseg"))

    assert result["status"] == "succeeded"
    assert [name for name, _ in tools.calls] == [
        "freeze_candidate",
        "validate_model",
        "start_evaluation",
        "wait_job",
        "wait_job",
        "inspect_job",
    ]
    assert tools.calls[2][1] == {
        "candidate_label": "proseg",
        "profile_id": "validation",
    }


def test_entrypoint_propagates_evaluation_failure() -> None:
    tools = FakeTools(["failed"], "failed")

    with pytest.raises(RuntimeError, match="did not succeed"):
        _entrypoint().run(tools, SimpleNamespace(reference_id="proseg"))
