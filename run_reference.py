from __future__ import annotations

_TERMINAL_STATUSES = {"succeeded", "failed", "timed_out", "cancelled", "orphaned"}


def run(tools, context):
    candidate_label = context.reference_id
    tools.call("freeze_candidate", {"candidate_label": candidate_label})

    validation = tools.call("validate_model", {"candidate_label": candidate_label})
    if validation.get("valid") is not True:
        raise RuntimeError(f"candidate validation failed: {validation!r}")

    evaluation = tools.call(
        "start_evaluation",
        {"candidate_label": candidate_label, "profile_id": "validation"},
    )
    job_id = evaluation["job_id"]
    for _ in range(5):
        current = tools.call("wait_job", {"job_id": job_id, "seconds": 300})
        if current["status"] in _TERMINAL_STATUSES:
            break

    completed = tools.call("inspect_job", {"job_id": job_id, "view": "summary", "detail": "full"})
    if completed["status"] != "succeeded":
        raise RuntimeError(f"reference evaluation did not succeed: {completed!r}")
    return completed
