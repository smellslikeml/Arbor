"""Tests for the event-sourced experiment memory and its wiring.

Covers the deterministic memory layer (adapted from PROJECTMEM,
arXiv:2606.12329) and that it is actually invoked by the existing executor
artifact sink, ``_save_experiment_artifacts`` in
``arbor.coordinator.tools.executor_run``.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

# Import the NON-NEW call-site module to exercise the wiring edit, plus the
# new capability functions it now calls.
from arbor.coordinator.tools.executor_run import _save_experiment_artifacts
from arbor.coordinator.tools.experiment_memory import (
    MEMORY_FILENAME,
    record_outcome,
    recall_prior_failures,
)


def _config(workspace, cwd):
    return SimpleNamespace(
        workspace_dir=str(workspace),
        cwd=str(cwd),
        trunk_branch="trunk",
    )


def test_record_and_recall_similar_failure(tmp_path):
    record_outcome(
        str(tmp_path),
        node_id="1.2",
        hypothesis="Add dropout 0.5 to the encoder to regularize training",
        score=22.0,
        insight="Dropout hurt convergence; score fell below baseline.",
        status="done",
    )

    # A hypothesis describing the same approach is gated when it did not beat
    # the baseline.
    warning = recall_prior_failures(
        str(tmp_path),
        "Apply dropout regularization to the encoder layers",
        baseline_score=30.0,
    )
    assert warning
    assert "1.2" in warning
    assert "do not repeat" in warning.lower()
    assert "dropout" in warning.lower()


def test_recall_ignores_dissimilar_and_successful(tmp_path):
    record_outcome(
        str(tmp_path),
        node_id="1.1",
        hypothesis="Add dropout 0.5 to the encoder",
        score=80.0,  # beat baseline -> success, not a failure
        status="done",
    )
    record_outcome(
        str(tmp_path),
        node_id="2.1",
        hypothesis="Switch the optimizer to Adam with cosine schedule",
        score=10.0,  # failure, but a totally different approach
        status="done",
    )

    warning = recall_prior_failures(
        str(tmp_path),
        "Apply dropout regularization to the encoder",
        baseline_score=30.0,
    )
    # Successful similar attempt is not warned about; dissimilar failure is
    # below the similarity threshold.
    assert warning == ""


def test_save_artifacts_records_outcome_event(tmp_path):
    """The existing artifact sink must append to the memory log (wiring)."""
    workspace = tmp_path / "ws"
    config = _config(workspace, tmp_path)

    asyncio.run(
        _save_experiment_artifacts(
            config=config,
            node_id="3.1",
            hypothesis="Increase beam width to 8 during decoding",
            raw_report="[Error: boom]",
            parsed={"score": None, "insight": "crashed", "result": ""},
            actual_branch="research/exp",
            agent_turns=3,
        )
    )

    log_path = workspace / MEMORY_FILENAME
    assert log_path.exists(), "experiment-memory log was not written by the sink"

    events = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    assert len(events) == 1
    event = events[0]
    assert event["node_id"] == "3.1"
    assert event["status"] == "failed"  # derived from the "[Error" report
    assert event["score"] is None

    # And the recorded failure is recallable for a similar future hypothesis.
    warning = recall_prior_failures(str(workspace), "Use a wider decoding beam of size 8")
    assert "3.1" in warning
