"""Event-sourced experiment memory for Executors.

Adapted from *PROJECTMEM: A Local-First, Event-Sourced Memory and Judgment
Layer for AI Coding Agents* (arXiv:2606.12329). The paper observes that
coding agents are largely stateless: each run re-derives context and, most
costly, may repeat debugging attempts that already failed. Its remedy is an
append-only event log paired with a *deterministic pre-action gate* that
warns the agent before it repeats a previously failed fix ("Memory-as-
Governance": memory that acts on the agent's next action, not just answers
it).

Arbor's Executors have the same gap — they "have no memory of the parent"
(``executor/prompts.py``) and each runs in a fresh worktree. This module
gives the Coordinator a tiny slice of that idea, scoped to what is reachable
here:

* a write side (:func:`record_outcome`) that appends a typed outcome event to
  a plain-text JSONL log in the workspace, emitted at the existing experiment
  artifact sink; and
* a read side (:func:`recall_prior_failures`) — the deterministic gate — that
  projects the log into a compact warning surfaced into the next Executor's
  prompt, flagging approaches similar to its hypothesis that already failed to
  improve.

The log is local-first, append-only, plain text, and contains no randomness
or model calls — the projection is fully deterministic, mirroring the paper's
provenance/auditability goals. It complements (does not replace) the
research-level :func:`_gather_ancestor_insights`, which only carries forward
*successful* ancestor insights.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

MEMORY_FILENAME = "experiment_memory.jsonl"

# Hypotheses sharing at least this fraction of their salient vocabulary are
# treated as "the same approach" for the purpose of the failure gate. The
# gate is advisory (it only adds a warning), so this leans toward recall —
# catching near-repeats — over precision.
_DEFAULT_SIMILARITY = 0.3
_MAX_WARNINGS = 3

# Words too generic to signal that two hypotheses describe the same approach.
_STOPWORDS = frozenset(
    """
    a an the of to for and or with without via using use uses on in into at by
    from as is are be that this it its our we will would should can could may
    try apply add improve increase reduce better baseline approach method idea
    experiment model score result change make based more less than then so
    """.split()
)


def _memory_path(workspace_dir: str | None) -> Path | None:
    """Return the event-log path for ``workspace_dir`` (or ``None``)."""
    if not workspace_dir:
        return None
    return Path(workspace_dir) / MEMORY_FILENAME


def _tokens(text: str) -> set[str]:
    """Lowercase salient word tokens, dropping stopwords and short tokens."""
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {w for w in words if len(w) > 2 and w not in _STOPWORDS}


def _similarity(a: str, b: str) -> float:
    """Jaccard overlap of the salient vocabulary of two hypotheses."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _is_failure(event: dict[str, Any], baseline_score: float | None) -> bool:
    """Deterministic verdict: did this recorded attempt fail to improve?

    An attempt is a failure worth warning about when it errored/timed out,
    when evaluation produced no score, or when its score did not beat the
    known baseline.
    """
    if event.get("status") in ("failed", "timeout"):
        return True
    score = event.get("score")
    if score is None:
        return True
    if baseline_score is not None:
        try:
            return float(score) <= float(baseline_score)
        except (TypeError, ValueError):
            return False
    return False


def record_outcome(
    workspace_dir: str | None,
    *,
    node_id: str,
    hypothesis: str,
    score: float | None,
    insight: str = "",
    result: str = "",
    branch: str = "",
    status: str = "done",
) -> None:
    """Append one outcome event to the append-only experiment-memory log.

    Best-effort and side-effect-only: any I/O error is logged and swallowed so
    that recording memory never breaks an Executor run. No-ops when the
    workspace has no on-disk location.
    """
    path = _memory_path(workspace_dir)
    if path is None:
        return

    event = {
        "type": "outcome",
        "node_id": node_id,
        "hypothesis": hypothesis,
        "score": score,
        "status": status,
        "insight": (insight or "").strip(),
        "result": (result or "").strip()[:500],
        "branch": branch,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")
    except OSError as exc:  # pragma: no cover - defensive
        log.warning("Could not append to experiment memory %s: %s", path, exc)


def _read_events(path: Path) -> list[dict[str, Any]]:
    """Read all well-formed events from the JSONL log (skipping bad lines)."""
    events: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return events
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def recall_prior_failures(
    workspace_dir: str | None,
    hypothesis: str,
    *,
    baseline_score: float | None = None,
    similarity_threshold: float = _DEFAULT_SIMILARITY,
    max_warnings: int = _MAX_WARNINGS,
) -> str:
    """Project the event log into a pre-action warning for the next Executor.

    Returns a Markdown block listing prior experiments whose hypothesis
    resembles ``hypothesis`` and which already failed to improve, or an empty
    string when the gate finds nothing. This is the deterministic "judgment"
    side of the memory layer: it acts on the Executor's next action by telling
    it which nearby approaches are already known not to work.
    """
    path = _memory_path(workspace_dir)
    if path is None or not path.exists():
        return ""

    scored: list[tuple[float, dict[str, Any]]] = []
    for event in _read_events(path):
        if not _is_failure(event, baseline_score):
            continue
        sim = _similarity(hypothesis, event.get("hypothesis", ""))
        if sim >= similarity_threshold:
            scored.append((sim, event))

    if not scored:
        return ""

    scored.sort(key=lambda item: item[0], reverse=True)

    lines = [
        "## ⚠️ Prior Failed Attempts (do not repeat)",
        "",
        "Earlier experiments tried approaches similar to this hypothesis and "
        "did **not** improve. Review what already failed and pursue a "
        "genuinely different angle rather than repeating these:",
        "",
    ]
    for _, event in scored[:max_warnings]:
        score = event.get("score")
        score_str = "no score" if score is None else f"score {score}"
        detail = event.get("insight") or event.get("result") or "no details recorded"
        lines.append(
            f"- **{event.get('node_id', '?')}** ({score_str}): "
            f"{event.get('hypothesis', '').strip()} — {detail}"
        )
    return "\n".join(lines)
