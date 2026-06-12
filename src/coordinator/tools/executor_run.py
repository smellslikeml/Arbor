"""RunExecutor tools — dispatch Research Agents in isolated git worktrees.

Each executor gets its own worktree (branched from current trunk HEAD),
so multiple executors can run in parallel without interfering with each
other or the main working directory.
"""

# pylint: disable=broad-exception-caught,protected-access

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shlex
import shutil
from pathlib import Path
from typing import Any, TYPE_CHECKING

from ...core.tools.base import Tool
from ...core.git_artifacts import filter_commit_paths
from ..hitl import await_user_decision
from .tree_ops import propagate_insights
from .git_ops import _run_git
from .experiment_memory import record_outcome, recall_prior_failures

if TYPE_CHECKING:
    from ..config import CoordinatorConfig
    from ..idea_tree import IdeaTree
    from ...core.llm.base import LLMProvider

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Experiment artifact persistence
# ---------------------------------------------------------------------------


_CYCLE_STATUSES = {"done", "merged", "pruned", "failed"}


def _completed_cycles(tree: "IdeaTree") -> int:
    """Count nodes that consume a cycle budget.

    A cycle is consumed once a executor finishes (regardless of outcome) or
    once a branch is pruned/merged. Failed runs are counted on purpose — they
    spent compute, so they spend budget.
    """
    return sum(
        1 for n in tree.get_all_nodes()
        if n.id != tree.root_id and n.status in _CYCLE_STATUSES
    )


async def _save_experiment_artifacts(
    *,
    config: "CoordinatorConfig",
    node_id: str,
    hypothesis: str,
    raw_report: str,
    parsed: dict[str, Any],
    actual_branch: str,
    agent_turns: int,
) -> None:
    """Save per-experiment artifacts to the workspace experiments/ directory."""
    workspace = config.workspace_dir
    if not workspace:
        return

    exp_dir = Path(workspace) / "experiments" / node_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    report_md = (
        f"# Experiment {node_id}\n\n"
        f"**Hypothesis**: {hypothesis}\n"
        f"**Branch**: `{actual_branch}`\n"
        f"**Turns**: {agent_turns}\n\n"
        f"---\n\n"
        f"{raw_report}\n"
    )
    (exp_dir / "report.md").write_text(report_md, encoding="utf-8")

    metrics = {
        "node_id": node_id,
        "hypothesis": hypothesis,
        "score": parsed.get("score"),
        "insight": parsed.get("insight", ""),
        "result": parsed.get("result", ""),
        "branch": actual_branch,
        "turns": agent_turns,
    }
    (exp_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    trunk = config.trunk_branch or "HEAD"
    diff_out, rc = await _run_git(
        f"git diff {shlex.quote(trunk)}...{shlex.quote(actual_branch)} --stat",
        config.cwd,
    )
    if rc == 0 and diff_out.strip():
        full_diff, _ = await _run_git(
            f"git diff {shlex.quote(trunk)}...{shlex.quote(actual_branch)}",
            config.cwd,
        )
        (exp_dir / "diff.patch").write_text(full_diff, encoding="utf-8")

    # Emit an outcome event into the append-only experiment-memory log so a
    # later executor can be warned against repeating this approach if it fails.
    status = "failed" if raw_report.startswith("[Error") else (
        "timeout" if raw_report.startswith("[Timed out") else "done"
    )
    record_outcome(
        workspace,
        node_id=node_id,
        hypothesis=hypothesis,
        score=parsed.get("score"),
        insight=parsed.get("insight", ""),
        result=parsed.get("result", ""),
        branch=actual_branch,
        status=status,
    )


# ---------------------------------------------------------------------------
# Git / worktree helpers
# ---------------------------------------------------------------------------


def _compute_branch_name(config: "CoordinatorConfig", node_id: str, hypothesis: str) -> str:
    """Derive the git branch name for a executor experiment."""
    from ...core.experiment import _slugify

    hypothesis_text = hypothesis or "unnamed"
    node_slug = _slugify(str(node_id), max_len=16)
    hypothesis_slug = _slugify(hypothesis_text, max_len=32)
    digest = hashlib.sha1(hypothesis_text.encode("utf-8")).hexdigest()[:8]
    return f"{config.git_branch_prefix}/n{node_slug}-{hypothesis_slug}-{digest}"


def _worktree_dir_name(branch_name: str) -> str:
    """Build a filesystem-safe, bounded worktree directory name."""
    safe = branch_name.replace("/", "__").replace(".", "_")
    if len(safe) <= 180:
        return safe
    digest = hashlib.sha1(branch_name.encode("utf-8")).hexdigest()[:12]
    return f"{safe[:160]}__{digest}"


async def _create_worktree(cwd: str, branch_name: str, start_point: str | None = None) -> tuple[Path, str]:
    """Create an isolated git worktree for a executor.

    Returns (worktree_path, actual_branch_name).
    The worktree is placed in a temp directory so it doesn't pollute the repo.

    If start_point is provided, the worktree branches from that ref
    (e.g. a working trunk branch) instead of HEAD.
    """
    import tempfile

    worktree_base = Path(tempfile.gettempdir()) / f"coordinator-worktrees-{os.getuid()}"
    worktree_base.mkdir(parents=True, exist_ok=True)

    dir_name = _worktree_dir_name(branch_name)
    worktree_path = worktree_base / dir_name

    # Clean up if exists from a previous failed run
    if worktree_path.exists():
        await _run_git(
            f"git worktree remove --force {shlex.quote(str(worktree_path))}", cwd
        )
        if worktree_path.exists():
            shutil.rmtree(worktree_path, ignore_errors=True)

    # Create worktree with new branch from start_point (or HEAD if not specified)
    start_ref = shlex.quote(start_point) if start_point else ""
    out, rc = await _run_git(
        f"git worktree add -b {shlex.quote(branch_name)} {shlex.quote(str(worktree_path))} {start_ref}".strip(),
        cwd,
    )
    if rc != 0:
        # Branch might already exist from a previous run — add timestamp suffix
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).strftime("%m%d-%H%M%S")
        branch_name = f"{branch_name}-{ts}"
        dir_name = _worktree_dir_name(branch_name)
        worktree_path = worktree_base / dir_name

        out, rc = await _run_git(
            f"git worktree add -b {shlex.quote(branch_name)} {shlex.quote(str(worktree_path))} {start_ref}".strip(),
            cwd,
        )
        if rc != 0:
            if worktree_path.exists():
                shutil.rmtree(worktree_path, ignore_errors=True)
            raise RuntimeError(f"Failed to create git worktree: {out}")

    return worktree_path, branch_name


async def _finalize_worktree(worktree_path: Path, node_id: str) -> None:
    """Commit useful code changes in the worktree before removal."""
    wt = str(worktree_path)
    await _run_git("git reset --", wt)
    diff, _ = await _run_git("git diff --name-only", wt)
    untracked, _ = await _run_git("git ls-files --others --exclude-standard", wt)
    changed_paths = [line.strip() for line in (diff + "\n" + untracked).splitlines() if line.strip()]
    commit_paths, artifact_paths = filter_commit_paths(changed_paths)

    if artifact_paths:
        log.info("Skipping generated artifacts for %s: %s", node_id, ", ".join(artifact_paths[:20]))

    if not commit_paths:
        return

    quoted_paths = " ".join(shlex.quote(path) for path in commit_paths)
    await _run_git(f"git add -- {quoted_paths}", wt)
    await _run_git(
        f"git commit -m {shlex.quote(f'coordinator: finalize {node_id}')}",
        wt,
    )


async def _remove_worktree(cwd: str, worktree_path: Path) -> None:
    """Remove a git worktree (the branch is preserved for later merging)."""
    try:
        await _run_git(
            f"git worktree remove --force {shlex.quote(str(worktree_path))}", cwd
        )
    except Exception as e:
        log.warning("Failed to remove worktree %s: %s", worktree_path, e)
    if worktree_path.exists():
        shutil.rmtree(worktree_path, ignore_errors=True)


# ---------------------------------------------------------------------------
# Context helpers
# ---------------------------------------------------------------------------

def _gather_ancestor_insights(tree: "IdeaTree", node_id: str) -> str:
    """Collect insights from ancestor nodes (root -> parent order)."""
    path = tree.get_path_to_root(node_id)
    parts: list[str] = []
    for ancestor in reversed(path[1:]):  # skip the node itself; root first
        if ancestor.insight:
            parts.append(f"- {ancestor.id}: {ancestor.insight}")
    if not parts:
        return ""
    return "## Insights from Prior Experiments\n\n" + "\n".join(parts)


def _substitute_eval_templates(cmd: str, cwd: str, node_id: str) -> str:
    """Replace template variables in eval_cmd.

    Supported variables:
      {cwd}     - working directory (worktree path for executors)
      {node_id} - tree node ID (e.g. "1.2.1") for unique result naming
    """
    return cmd.replace("{cwd}", cwd).replace("{node_id}", node_id)


def _get_eval_info(tree: "IdeaTree", *, worktree_cwd: str | None = None, node_id: str | None = None) -> str:
    """Read evaluation info from tree.meta and format for executor injection.

    If worktree_cwd and node_id are provided, template variables in eval_cmd
    are substituted so the executor gets ready-to-run commands.
    """
    meta = tree.meta
    parts: list[str] = []

    def _sub(cmd: str) -> str:
        """Apply template substitution if context is available."""
        if worktree_cwd and node_id:
            return _substitute_eval_templates(cmd, worktree_cwd, node_id)
        return cmd

    if meta.get("eval_cmd"):
        parts.append(f"- **Evaluation command (B_dev)**: `{_sub(meta['eval_cmd'])}`")
    if meta.get("eval_cmd_test"):
        parts.append(
            f"- **Evaluation command (B_test)** "
            f"(DO NOT use for routine experiments): `{_sub(meta['eval_cmd_test'])}`"
        )
    if meta.get("dataset_info"):
        parts.append(f"- **Dataset info**: {meta['dataset_info']}")
    if meta.get("baseline_score") is not None:
        parts.append(f"- **Baseline score**: {meta['baseline_score']}")
    if meta.get("trunk_score") is not None:
        parts.append(f"- **Current trunk score**: {meta['trunk_score']}")

    if not parts:
        return ""
    return (
        "## Evaluation Info\n\n"
        + "\n".join(parts)
        + "\n\nUse the B_dev evaluation command above to measure your final results. "
        "Do NOT use B_test.\n\n"
        "You may test on a subset of examples or individual cases first for "
        "quick validation before committing to a full evaluation run. Use your "
        "judgment — if a quick check shows the approach clearly isn't working, "
        "iterate on the implementation instead of running the full eval.\n\n"
        "**CRITICAL**: Run ALL commands from your working directory. "
        "Do NOT `cd` to any other directory, especially not the main repository. "
        "Your working directory is an isolated worktree with the correct code. "
        "Running eval elsewhere will evaluate the wrong code and corrupt results."
    )


def _build_executor_prompt(
    *,
    worktree_path: Path,
    node: Any,
    ancestor_insights: str,
    eval_info: str,
    additional_context: str | None,
) -> str:
    """Build the user message for a executor."""
    parts = [
        f"## Codebase\n\nWorking directory: {worktree_path}",
        (
            "## Git Isolation\n\n"
            "You are running in an isolated git worktree created from the current "
            "coordinator trunk. Do not checkout a different branch and do not `cd` "
            "back to the main workspace. If you need to sanity-check the setup, run "
            "`pwd` and `git branch --show-current`; both should point to this "
            "worktree/experiment branch."
        ),
        (
            f"## Research Idea\n\n"
            f"**ID**: {node.id}\n"
            f"**Hypothesis**: {node.hypothesis}"
        ),
    ]

    if eval_info:
        parts.append(eval_info)
    if ancestor_insights:
        parts.append(ancestor_insights)
    if additional_context:
        parts.append(f"## Additional Context\n\n{additional_context}")

    parts.append(
        "## Instructions\n\n"
        "Implement the research idea faithfully and thoroughly:\n\n"
        "1. **UNDERSTAND**: Read the codebase to understand the current "
        "implementation before making any changes.\n"
        "2. **IMPLEMENT**: Make the code changes to realize the idea.\n"
        "3. **VALIDATE IMPLEMENTATION**: Run on 2-3 representative examples "
        "to verify your code actually works — no crashes, the new logic is "
        "actually being reached, outputs look sane. If there are bugs or the "
        "idea isn't taking effect, **go back to step 2 and fix it**. Do NOT "
        "run full eval on a broken implementation.\n"
        "4. **ITERATE UNTIL SOLID**: Repeat steps 2-3 until you are confident "
        "the idea is implemented correctly and completely — the code runs "
        "without errors, the new logic is active, and the approach has been "
        "given a fair shot. The goal is to separate 'idea quality' from "
        "'implementation quality': a bad score should mean the idea itself "
        "didn't help, not that the code was buggy.\n"
        "5. **EVALUATE**: Once the implementation is solid, run the full "
        "evaluation command to get the definitive score. You may skip full "
        "eval ONLY if, after a thorough and correct implementation, quick "
        "checks already make the outcome unambiguous.\n"
        "6. **REPORT**: Provide:\n"
        "   - **Changes**: Files modified\n"
        "   - **Baseline vs Result**: Metrics comparison\n"
        "   - **Score**: The primary metric value achieved "
        "(report as absolute value, not delta; null only if eval was skipped)\n"
        "   - **Insight**: Why it worked or didn't\n\n"
        "**Key principle**: Give every idea a fair shot. A negative result is "
        "valuable — but only if the implementation was correct. Iterate on "
        "implementation bugs before concluding an idea doesn't work.\n\n"
        "**Do not stop after describing what you will do. If you say you will "
        "write, edit, train, or evaluate something, call the appropriate tool "
        "in that same turn or the next turn before reporting completion.**\n\n"
        f"**Result directory**: Save results to `results/{node.id}-<brief-description>/` "
        f"to avoid overwriting other experiments."
    )

    return "\n\n".join(parts)


async def _parse_executor_report(
    provider: "LLMProvider",
    report: str,
    hypothesis: str,
    *,
    bus: Any | None = None,
    cwd: str | None = None,
) -> dict[str, Any]:
    """Use the LLM to extract structured fields from a executor report."""
    max_chars = 12000
    if len(report) <= max_chars:
        report_excerpt = report
    else:
        head = report[: max_chars // 2]
        tail = report[-(max_chars // 2) :]
        report_excerpt = f"{head}\n\n[... middle truncated ...]\n\n{tail}"

    response = await provider.create(
        system=(
            "You extract structured data from experiment reports. "
            "Return ONLY valid JSON with these fields:\n"
            '- "score": number or null (the absolute metric value achieved, '
            "e.g. 45.2 means 45.2%. This is NOT a delta.)\n"
            '- "insight": string (key learning, 1-3 sentences)\n'
            '- "result": string (factual description, 1-2 sentences)\n'
            '- "code_ref": string or null (git branch name if mentioned)\n'
            "No markdown fencing. Just raw JSON."
        ),
        messages=[
            {
                "role": "user",
                "content": (
                    f"Hypothesis: {hypothesis}\n\n"
                    f"Experiment report:\n{report_excerpt}\n\n"
                    "Extract the structured fields as JSON."
                ),
            }
        ],
        max_tokens=1024,
    )
    try:
        from ...core.agent import record_llm_usage
        record_llm_usage(
            response,
            bus=bus,
            model=getattr(provider, "model", None),
            source="parse_executor_report",
            agent_cwd=cwd,
        )
    except Exception:  # pylint: disable=broad-exception-caught
        pass
    text = response.get_text().strip()

    # Strip markdown code fences if the LLM added them
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        log.warning("Failed to parse JSON from LLM response: %s", text[:200])
        return {"score": None, "insight": "", "result": text[:500], "code_ref": None}


async def _run_after_executor_hook(
    config: "CoordinatorConfig",
    worktree_path: Path,
    node_id: str,
) -> None:
    """Run the after_executor lifecycle hook — snapshot submission if it exists."""
    plugin = config.plugin
    if not plugin or not plugin.lifecycle_hooks:
        return

    hook = plugin.lifecycle_hooks.get("after_executor")
    if not hook:
        return

    submission_rel = plugin.eval_contract.get("submission_path", "submission.csv")
    submission = worktree_path / submission_rel
    if submission.exists():
        workspace_root = Path(config.workspace_dir) if config.workspace_dir else Path(config.cwd)
        snapshot_dir = workspace_root / "submissions"
        snapshot_dir.mkdir(exist_ok=True)
        ext = Path(submission_rel).suffix or ".csv"
        snapshot_name = f"{node_id}{ext}"
        shutil.copy2(submission, snapshot_dir / snapshot_name)
        log.info("Snapshot submission for %s -> submissions/%s", node_id, snapshot_name)


# ---------------------------------------------------------------------------
# Core: run a single executor in an isolated worktree
# ---------------------------------------------------------------------------

async def _run_single_executor(
    *,
    tree: "IdeaTree",
    config: "CoordinatorConfig",
    provider: "LLMProvider",
    node_id: str,
    additional_context: str | None = None,
) -> str:
    """Run one executor in an isolated git worktree.

    Handles the full lifecycle: validate → worktree → run → parse → update tree.
    """
    # ── Early stop: gold already achieved ──────────────────────────
    if tree.meta.get("achieved_medal") == "gold":
        return (
            f"Early stop: Gold medal already achieved on trunk. "
            f"No further experiments needed. Node {node_id} was NOT dispatched."
        )

    from ...core.agent import Agent
    from ...core.tools import get_all_tools
    from ...core.tools.executor_tool import ExecutorTool
    from ...events import types as ev
    from ...executor.prompts import build_system_prompt

    # ── 1. Validate node ────────────────────────────────────────────────
    node = tree.get_node(node_id)
    if node is None:
        return f"Error: Node {node_id!r} not found in the idea tree."
    if node.status not in ("pending", "running"):
        return (
            f"Error: Node {node_id} has status={node.status!r}. "
            f"Only 'pending' nodes can be dispatched."
        )

    # ── 1b. Enforce leaf-only dispatch when max_depth is set ───────────
    if tree.max_depth is not None and node.depth < tree.max_depth:
        return (
            f"Error: Node {node_id} is at depth {node.depth}, but max_depth "
            f"is {tree.max_depth}. Only leaf nodes (depth={tree.max_depth}) "
            f"can be dispatched for experiments. Please refine this idea into "
            f"more specific sub-ideas using TreeAddNode before dispatching."
        )

    # ── 2. Mark as running ──────────────────────────────────────────────
    await tree.async_update_node(node_id, status="running")
    cycle_num = _completed_cycles(tree) + 1
    tree.bus.emit(ev.CYCLE_START, {
        "cycle_num": cycle_num,
        "total_cycles": config.max_cycles,
        "node_id": node_id,
    })

    # ── 3. Create worktree ──────────────────────────────────────────────
    branch_name = _compute_branch_name(config, node_id, node.hypothesis)
    worktree_path: Path | None = None
    actual_branch = branch_name

    try:
        worktree_path, actual_branch = await _create_worktree(
            config.cwd, branch_name, start_point=config.trunk_branch,
        )
    except RuntimeError as e:
        await tree.async_update_node(node_id, status="pending", result=f"Worktree creation failed: {e}")
        return f"Error creating worktree for {node_id}: {e}"
    tree.bus.emit(ev.EXECUTOR_START, {
        "node_id": node_id,
        "idea": node.hypothesis,
        "branch": actual_branch,
        "cycle_num": cycle_num,
    })

    # ── 4. Build executor ───────────────────────────────────────────────
    raw_report = ""
    agent_turns = 0
    agent: Agent | None = None
    executor_t0 = asyncio.get_running_loop().time()

    try:
        executor_config = config.to_executor_config(node_id, node.hypothesis)
        executor_config.cwd = str(worktree_path)
        executor_config.event_bus = tree.bus

        system_prompt = build_system_prompt(executor_config, plugin=config.plugin)
        tools = get_all_tools(
            cwd=str(worktree_path),
            workspace_dir=executor_config.workspace_dir,
            config=executor_config,
        )

        agent = Agent(
            provider=provider,
            tools=tools,
            system_prompt=system_prompt,
            config=executor_config,
        )

        # Pre-initialize git manager — worktree already has the correct branch
        agent.git_manager._initialized = True
        agent.git_manager.branch_name = actual_branch
        agent.git_manager.cwd = str(worktree_path)

        # Add Executor tool for nested delegation
        executor_tool = ExecutorTool(cwd=str(worktree_path), parent_agent=agent, workspace_dir=executor_config.workspace_dir)
        agent.tools[executor_tool.name] = executor_tool

        # ── 5. Build prompt with auto-injected eval info ────────────────
        ancestor_insights = _gather_ancestor_insights(tree, node_id)
        eval_info = _get_eval_info(
            tree,
            worktree_cwd=str(worktree_path),
            node_id=node_id,
        )
        # Pre-action gate: warn against repeating approaches that already
        # failed (recorded in the experiment-memory log by prior cycles).
        prior_failures = recall_prior_failures(
            config.workspace_dir,
            node.hypothesis,
            baseline_score=tree.meta.get("baseline_score"),
        )
        executor_context = "\n\n".join(
            part for part in (additional_context, prior_failures) if part
        ) or None
        prompt = _build_executor_prompt(
            worktree_path=worktree_path,
            node=node,
            ancestor_insights=ancestor_insights,
            eval_info=eval_info,
            additional_context=executor_context,
        )

        log.info(
            "Dispatching executor for %s in worktree %s (branch=%s, timeout=%ds)",
            node_id, worktree_path, actual_branch, config.executor_timeout,
        )

        # ── 6. Run executor ─────────────────────────────────────────────
        result = await asyncio.wait_for(
            agent.run(prompt),
            timeout=config.executor_timeout,
        )
        raw_report = result
        agent_turns = agent.total_turns

    except asyncio.TimeoutError:
        agent_turns = agent.total_turns if agent is not None else 0
        raw_report = f"[Timed out after {config.executor_timeout}s]"
        log.warning("Executor for %s timed out after %ds", node_id, config.executor_timeout)

    except Exception as e:
        agent_turns = agent.total_turns if agent is not None else 0
        raw_report = f"[Error: {e}]"
        log.error("Executor for %s failed: %s", node_id, e)

    # ── 7. Finalize & clean up worktree ─────────────────────────────────
    if worktree_path is not None:
        try:
            await _finalize_worktree(worktree_path, node_id)
        except Exception as e:
            log.warning("Failed to finalize worktree for %s: %s", node_id, e)

        # ── 7b. after_executor lifecycle hook ─────────────────────────
        try:
            await _run_after_executor_hook(config, worktree_path, node_id)
        except Exception as e:
            log.warning("after_executor hook failed for %s: %s", node_id, e)

        await _remove_worktree(config.cwd, worktree_path)

    # ── 8. Parse report ─────────────────────────────────────────────────
    try:
        parsed = await _parse_executor_report(
            provider,
            raw_report,
            node.hypothesis,
            bus=tree.bus,
            cwd=str(worktree_path) if worktree_path is not None else config.cwd,
        )
    except Exception as e:
        log.warning("Failed to parse report for %s: %s", node_id, e)
        parsed = {}

    score = parsed.get("score")
    insight = parsed.get("insight", "")
    result_text = parsed.get("result", "")
    code_ref = parsed.get("code_ref") or actual_branch

    # ── 9. Update tree node ─────────────────────────────────────────────
    await tree.async_update_node(
        node_id,
        status="done",
        score=score,
        insight=insight or ("Timed out" if raw_report.startswith("[Timed out") else ""),
        result=result_text or raw_report[:300],
        code_ref=code_ref,
    )
    duration = max(0.0, asyncio.get_running_loop().time() - executor_t0)
    tree.bus.emit(ev.EXECUTOR_END, {
        "node_id": node_id,
        "score": score,
        "duration": duration,
        "tokens": (
            (agent.total_input_tokens + agent.total_output_tokens)
            if agent is not None else None
        ),
        "turns": agent_turns,
        "branch": code_ref,
    })
    tree.bus.emit(ev.CYCLE_END, {
        "cycle_num": cycle_num,
        "total_cycles": config.max_cycles,
        "node_id": node_id,
        "duration": duration,
    })

    # ── 9b. Save experiment artifacts to workspace ─────────────────────
    try:
        await _save_experiment_artifacts(
            config=config,
            node_id=node_id,
            hypothesis=node.hypothesis,
            raw_report=raw_report,
            parsed=parsed,
            actual_branch=actual_branch,
            agent_turns=agent_turns,
        )
    except Exception as e:
        log.warning("Failed to save experiment artifacts for %s: %s", node_id, e)

    # ── 10. Propagate insights upward ───────────────────────────────────
    propagation_result = ""
    try:
        propagation_result = await propagate_insights(tree, provider, node_id)
    except Exception as e:
        log.warning("Propagation failed for %s: %s", node_id, e)
        propagation_result = f"Propagation failed: {e}"

    # ── 11. Format summary ──────────────────────────────────────────────
    score_str = f"{score:.1f}%" if score is not None else "N/A"

    # Include a reasonable excerpt of the raw report
    report_excerpt = raw_report
    if len(raw_report) > 8000:
        report_excerpt = (
            raw_report[:4000]
            + f"\n\n[... middle truncated, full report was {len(raw_report)} chars ...]\n\n"
            + raw_report[-4000:]
        )

    return (
        f"## Executor Result for {node_id}\n\n"
        f"**Hypothesis**: {node.hypothesis}\n"
        f"**Status**: done (auto-updated)\n"
        f"**Score**: {score_str}\n"
        f"**Insight**: {insight}\n"
        f"**Branch**: `{code_ref}`\n"
        f"**Turns**: {agent_turns}\n\n"
        f"### Propagation\n{propagation_result}\n\n"
        f"### Report Excerpt\n\n{report_excerpt}"
    )


# ---------------------------------------------------------------------------
# HITL review gate (#2)
# ---------------------------------------------------------------------------

async def _review_gate(
    tree: "IdeaTree", config: "CoordinatorConfig", node_id: str, hypothesis: str,
) -> tuple[str, str | None]:
    """In ``review`` mode, ask the human before exploring a node.

    Returns ``(action, note)`` where ``action`` is ``"approve"`` or ``"skip"``
    and ``note`` is an optional free-text edit/comment to fold into the
    executor's context. No-op (auto-approve) in ``auto`` mode or on timeout.
    """
    interaction_mode = (getattr(config.ui, "interaction_mode", "auto") or "auto").lower()
    if interaction_mode not in ("review", "collaborative"):
        return ("approve", None)
    reply = await await_user_decision(
        tree.bus,
        kind="idea_review",
        prompt=f"Explore idea {node_id}: {hypothesis}",
        node_id=node_id,
        options=["approve", "skip", "edit <note>"],
        timeout=max(1, int(config.ui.review_timeout)),
    )
    if reply is None:
        log.info("review gate: node %s auto-approved (no review in window)", node_id)
        return ("approve", None)
    text = reply.strip()
    low = text.lower()
    if low in ("", "approve", "approved", "yes", "y", "ok", "go"):
        return ("approve", None)
    if low in ("skip", "no", "n", "reject"):
        return ("skip", None)
    if low.startswith("edit "):
        text = text[5:].strip()
    return ("approve", text or None)


# ---------------------------------------------------------------------------
# Tool: RunExecutor (single dispatch)
# ---------------------------------------------------------------------------

class RunExecutorTool(Tool):
    """Dispatch a single executor to implement and test a specific idea."""

    name = "RunExecutor"
    description = (
        "Dispatch a executor to implement and test a specific idea from the tree.\n\n"
        "The executor runs in an isolated git worktree branched from current trunk.\n"
        "It will:\n"
        "1. Implement the idea on an isolated branch\n"
        "2. Run evaluation (eval_cmd is auto-injected from tree metadata)\n"
        "3. Report results and insights\n\n"
        "The tree node is auto-updated with results and insights are propagated.\n\n"
        "For running multiple ideas at once, use RunExecutorParallel instead."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "node_id": {
                "type": "string",
                "description": "The idea node ID to implement (must be 'pending').",
            },
            "additional_context": {
                "type": "string",
                "description": (
                    "Extra context: relevant file paths, implementation hints, "
                    "insights from the tree. (Eval info is auto-injected — "
                    "no need to repeat it here.)"
                ),
            },
        },
        "required": ["node_id"],
    }
    is_read_only = False
    max_result_chars = 100_000

    def __init__(
        self,
        *,
        cwd: str,
        tree: "IdeaTree",
        config: "CoordinatorConfig",
        provider: "LLMProvider",
        convergence_detector: Any | None = None,
        **kwargs: Any,
    ):
        super().__init__(cwd=cwd, **kwargs)
        self._tree = tree
        self._config = config
        self._provider = provider
        self._convergence_detector = convergence_detector

    async def execute(self, **kwargs: Any) -> str:
        done = _completed_cycles(self._tree)
        cap = self._config.max_cycles
        if done >= cap:
            return (
                f"HARD LIMIT REACHED: {done}/{cap} cycles already consumed "
                f"(counting done/merged/pruned/failed). RunExecutor is disabled. "
                f"Finalize now: merge the best branch if it beats the threshold, "
                f"otherwise stop and report."
            )
        # ── HITL review gate (#2): approve / skip / edit before spending compute ──
        node = self._tree.get_node(kwargs["node_id"])
        if node is not None:
            action, note = await _review_gate(
                self._tree, self._config, kwargs["node_id"], node.hypothesis)
            if action == "skip":
                self._tree.prune_node(kwargs["node_id"], reason="skipped by user in review")
                return (f"Idea {kwargs['node_id']} was skipped by the user (review mode); "
                        f"pruned, not explored. Propose or dispatch another idea.")
            if note:
                ctx = kwargs.get("additional_context")
                kwargs["additional_context"] = (
                    (f"{ctx}\n\n" if ctx else "") + f"User review note: {note}")
        result = await _run_single_executor(
            tree=self._tree,
            config=self._config,
            provider=self._provider,
            node_id=kwargs["node_id"],
            additional_context=kwargs.get("additional_context"),
        )
        # Check convergence after experiment completion
        if self._convergence_detector:
            signal = self._convergence_detector.on_experiment_complete(kwargs["node_id"])
            if signal:
                intervention = self._convergence_detector.format_intervention(signal)
                result += f"\n\n---\n{intervention}\n---"
                if signal.level == "stop":
                    self._convergence_detector.write_stop_signal(self._config.workspace_dir)
        return result


# ---------------------------------------------------------------------------
# Tool: RunExecutorParallel (batch dispatch)
# ---------------------------------------------------------------------------

class RunExecutorParallelTool(Tool):
    """Dispatch multiple executors in parallel, each in its own git worktree."""

    name = "RunExecutorParallel"
    description = (
        "Dispatch 2-4 executors in parallel, each in its own isolated git worktree.\n\n"
        "Use this to explore multiple ideas simultaneously for faster iteration.\n"
        "Each executor gets its own copy of the trunk codebase and cannot interfere\n"
        "with others.\n\n"
        "All tree nodes are auto-updated with results and insights are propagated.\n"
        "Returns combined results for all dispatched executors."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "node_id": {
                            "type": "string",
                            "description": "Idea node ID to implement.",
                        },
                        "additional_context": {
                            "type": "string",
                            "description": "Extra context for this specific executor.",
                        },
                    },
                    "required": ["node_id"],
                },
                "minItems": 2,
                "maxItems": 4,
                "description": "List of ideas to explore in parallel (2-4 items).",
            },
        },
        "required": ["tasks"],
    }
    is_read_only = False
    max_result_chars = 200_000

    def __init__(
        self,
        *,
        cwd: str,
        tree: "IdeaTree",
        config: "CoordinatorConfig",
        provider: "LLMProvider",
        convergence_detector: Any | None = None,
        **kwargs: Any,
    ):
        super().__init__(cwd=cwd, **kwargs)
        self._tree = tree
        self._config = config
        self._provider = provider
        self._convergence_detector = convergence_detector

    async def execute(self, **kwargs: Any) -> str:
        tasks = kwargs["tasks"]

        max_parallel = self._config.budget_policy.max_parallel_executors
        if max_parallel is not None:
            if max_parallel <= 1:
                return (
                    "RunExecutorParallel is disabled by "
                    "budget_policy.max_parallel_executors=1. "
                    "Use RunExecutor(node_id=..., additional_context=...) and "
                    "wait for that single experiment to finish before starting another."
                )
            if len(tasks) > max_parallel:
                tasks = tasks[:max_parallel]
                kwargs["tasks"] = tasks
                log.warning(
                    "RunExecutorParallel: truncated to %d task(s) to respect "
                    "budget_policy.max_parallel_executors=%d",
                    len(tasks),
                    max_parallel,
                )

        # ── Early stop: gold already achieved ──────────────────────
        if self._tree.meta.get("achieved_medal") == "gold":
            return (
                "Early stop: Gold medal already achieved on trunk. "
                "No further experiments needed."
            )

        # ── Hard cycle cap ─────────────────────────────────────────
        done = _completed_cycles(self._tree)
        cap = self._config.max_cycles
        remaining = cap - done
        if remaining <= 0:
            return (
                f"HARD LIMIT REACHED: {done}/{cap} cycles already consumed "
                f"(counting done/merged/pruned/failed). RunExecutorParallel is "
                f"disabled. Finalize now: merge the best branch if it beats the "
                f"threshold, otherwise stop and report."
            )
        if len(tasks) > remaining:
            tasks = tasks[:remaining]
            kwargs["tasks"] = tasks
            log.warning(
                "RunExecutorParallel: truncated to %d task(s) to respect "
                "max_cycles=%d (already done=%d)", len(tasks), cap, done,
            )

        # ── Validate all nodes upfront ──────────────────────────────────
        errors: list[str] = []
        for task in tasks:
            node = self._tree.get_node(task["node_id"])
            if node is None:
                errors.append(f"Node {task['node_id']!r} not found.")
            elif node.status != "pending":
                errors.append(
                    f"Node {task['node_id']} has status={node.status!r}, expected 'pending'."
                )
            elif (
                self._tree.max_depth is not None
                and node is not None
                and node.depth < self._tree.max_depth
            ):
                errors.append(
                    f"Node {task['node_id']} is at depth {node.depth}, but max_depth "
                    f"is {self._tree.max_depth}. Only leaf nodes (depth={self._tree.max_depth}) "
                    f"can be dispatched. Refine into sub-ideas first."
                )
        if errors:
            return "Validation errors:\n" + "\n".join(f"- {e}" for e in errors)

        # ── HITL review gate (#2): review each idea before compute; drop skips ──
        interaction_mode = (getattr(self._config.ui, "interaction_mode", "auto") or "auto").lower()
        if interaction_mode in ("review", "collaborative"):
            kept: list[dict] = []
            skipped: list[str] = []
            for task in tasks:
                node = self._tree.get_node(task["node_id"])
                action, note = await _review_gate(
                    self._tree, self._config, task["node_id"],
                    node.hypothesis if node else "")
                if action == "skip":
                    self._tree.prune_node(task["node_id"], reason="skipped by user in review")
                    skipped.append(task["node_id"])
                    continue
                if note:
                    ctx = task.get("additional_context")
                    task["additional_context"] = (
                        (f"{ctx}\n\n" if ctx else "") + f"User review note: {note}")
                kept.append(task)
            tasks = kept
            kwargs["tasks"] = tasks
            if not tasks:
                return (f"All proposed ideas ({', '.join(skipped)}) were skipped by the "
                        f"user (review mode). Nothing dispatched — propose different ideas.")

        # ── Dispatch all executors concurrently ─────────────────────────
        log.info("Dispatching %d executors in parallel", len(tasks))

        coroutines = [
            _run_single_executor(
                tree=self._tree,
                config=self._config,
                provider=self._provider,
                node_id=task["node_id"],
                additional_context=task.get("additional_context"),
            )
            for task in tasks
        ]

        results = await asyncio.gather(*coroutines, return_exceptions=True)

        # ── Format combined report ──────────────────────────────────────
        parts: list[str] = [f"# Parallel Executor Results ({len(tasks)} dispatched)\n"]

        completed = 0
        for i, (task, result) in enumerate(zip(tasks, results)):
            parts.append(f"---\n## Task {i + 1}: {task['node_id']}\n")
            if isinstance(result, Exception):
                parts.append(f"**Error**: {result}\n")
            else:
                completed += 1
                parts.append(result)

        parts.append(
            f"\n---\n**Summary**: {completed}/{len(tasks)} completed successfully."
        )

        # Check convergence after all parallel experiments complete
        combined = "\n\n".join(parts)
        if self._convergence_detector:
            signal = None
            for task in tasks:
                signal = self._convergence_detector.on_experiment_complete(task["node_id"])
            if signal:
                intervention = self._convergence_detector.format_intervention(signal)
                combined += f"\n\n---\n{intervention}\n---"
                if signal.level == "stop":
                    self._convergence_detector.write_stop_signal(self._config.workspace_dir)

        return combined
