"""Fixed-budget branch selection for the coordinator research loop.

The convergence detector tells the coordinator *when* a plateau has been hit
and *which* parents are exhausted, but it never says *which surviving branch
deserves the remaining compute*. This module fills that gap.

It treats the top-level research branches (the immediate children of ROOT) as
the *arms* of a best-arm-identification problem, the executor scores beneath
each branch as that arm's reward samples, and the unused portion of
``max_cycles`` as a fixed budget. It then ranks the arms by a Bayesian
upper-confidence bound whose exploration weight scales with the remaining
budget — with lots of budget left it favours under-sampled but promising
branches; as the budget runs out it collapses toward exploiting the branch
with the best posterior mean.

Adapted from "UCB Exploration for Fixed-Budget Bayesian Best Arm
Identification" (arXiv:2408.04869). We implement the allocation *rule* the
paper argues for — a budget-aware UCB index over a Gaussian posterior — rather
than its regret analysis, which is what is useful at this call site.

The output is advisory text injected into the coordinator's context alongside
the existing ConvergenceSignal intervention; see
``coordinator.convergence.ConvergenceDetector``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import CoordinatorConfig
    from .idea_tree import IdeaTree

log = logging.getLogger(__name__)

_DONE_STATUSES = ("done", "merged")


@dataclass
class ArmStat:
    """Per-branch (arm) statistics under the BAI view of the idea tree."""

    arm_id: str
    hypothesis: str
    n_pulls: int
    mean_reward: float
    posterior_mean: float
    posterior_std: float
    ucb_index: float


@dataclass
class BudgetRecommendation:
    """Advisory output: where to spend the remaining cycle budget."""

    remaining_budget: int
    best_arm: str
    best_arm_hypothesis: str
    ranking: list[ArmStat] = field(default_factory=list)
    deprioritize: list[str] = field(default_factory=list)


class BranchBudgetRecommender:
    """Recommends which surviving branch to invest the remaining budget in.

    Usage::

        rec = BranchBudgetRecommender(tree, config).recommend(remaining_budget)
        if rec is not None:
            text = BranchBudgetRecommender.format_recommendation(rec)
    """

    # Prior pseudo-count: how much the pooled (cross-arm) mean pulls a
    # sparsely sampled arm toward the global average.
    PRIOR_STRENGTH: float = 1.0

    def __init__(self, tree: "IdeaTree", config: "CoordinatorConfig"):
        self._tree = tree
        self._config = config

    # ── public API ───────────────────────────────────────────────────

    def recommend(self, remaining_budget: int) -> BudgetRecommendation | None:
        """Rank surviving branches by budget-aware UCB.

        Returns None when allocation is moot: no budget left, or fewer than
        two scored branches to choose between.
        """
        if remaining_budget <= 0:
            return None

        arms = self._collect_arm_rewards()
        scored = {aid: r for aid, r in arms.items() if r}
        if len(scored) < 2:
            return None

        # Pooled spread across every observed reward → uncertainty floor for
        # arms with one (or identical) sample(s).
        all_rewards = [v for rewards in scored.values() for v in rewards]
        pooled_std = _stddev(all_rewards)
        global_mean = sum(all_rewards) / len(all_rewards)
        if pooled_std <= 0.0:
            pooled_std = abs(global_mean) * 0.01 or 1.0

        # Exploration weight grows with the (log of the) remaining budget:
        # more runway → more willing to back uncertain arms.
        beta = math.sqrt(2.0 * (math.log(remaining_budget + 1.0) + 1.0))

        stats: list[ArmStat] = []
        for arm_id, rewards in scored.items():
            n = len(rewards)
            mean = sum(rewards) / n
            post_n = n + self.PRIOR_STRENGTH
            post_mean = (n * mean + self.PRIOR_STRENGTH * global_mean) / post_n
            post_std = pooled_std / math.sqrt(post_n)
            ucb = post_mean + beta * post_std
            stats.append(ArmStat(
                arm_id=arm_id,
                hypothesis=self._hypothesis_of(arm_id),
                n_pulls=n,
                mean_reward=mean,
                posterior_mean=post_mean,
                posterior_std=post_std,
                ucb_index=ucb,
            ))

        stats.sort(key=lambda s: s.ucb_index, reverse=True)
        best = stats[0]

        # An arm is dominated when even its optimistic UCB cannot reach the
        # leader's *expected* reward — no budget should chase it.
        deprioritize = [
            s.arm_id for s in stats[1:]
            if s.ucb_index < best.posterior_mean
        ]

        log.info(
            "Budget recommendation: best_arm=%s (ucb=%.5f, n=%d), "
            "remaining_budget=%d, arms=%d",
            best.arm_id, best.ucb_index, best.n_pulls,
            remaining_budget, len(stats),
        )

        return BudgetRecommendation(
            remaining_budget=remaining_budget,
            best_arm=best.arm_id,
            best_arm_hypothesis=best.hypothesis,
            ranking=stats,
            deprioritize=deprioritize,
        )

    @staticmethod
    def format_recommendation(rec: BudgetRecommendation) -> str:
        """Render the recommendation for injection into coordinator context."""
        lines = [
            "## [Budget] FIXED-BUDGET BRANCH ALLOCATION",
            "",
            (
                f"{rec.remaining_budget} cycle(s) of budget remain. Treating each "
                f"top-level branch as an arm (scores = reward samples), the "
                f"UCB best-arm rule recommends investing remaining budget in:"
            ),
            "",
            f"**Back branch `{rec.best_arm}`** — {rec.best_arm_hypothesis or '(no hypothesis)'}",
            "",
            "**Branch ranking (UCB index = posterior mean + budget-scaled uncertainty):**",
        ]
        for i, s in enumerate(rec.ranking, 1):
            lines.append(
                f"{i}. `{s.arm_id}` — ucb={s.ucb_index:.4f} "
                f"(mean={s.mean_reward:.4f}, n={s.n_pulls}, "
                f"±{s.posterior_std:.4f})"
            )
        if rec.deprioritize:
            lines.extend([
                "",
                f"**Deprioritize** (dominated — even optimistically below the "
                f"leader's expected reward): {rec.deprioritize}",
            ])
        return "\n".join(lines)

    # ── internals ────────────────────────────────────────────────────

    def _collect_arm_rewards(self) -> dict[str, list[float]]:
        """Map each top-level branch id to the oriented reward samples in its
        subtree (the branch node plus all descendants)."""
        root_id = self._tree.root_id
        minimize = self._tree.meta.get("metric_direction", "maximize") == "minimize"
        arms: dict[str, list[float]] = {}
        for arm in self._tree.get_children(root_id):
            rewards: list[float] = []
            for node in self._iter_subtree(arm.id):
                if node.status in _DONE_STATUSES and node.score is not None:
                    rewards.append(-node.score if minimize else node.score)
            arms[arm.id] = rewards
        return arms

    def _iter_subtree(self, node_id: str) -> list[Any]:
        """All nodes in the subtree rooted at ``node_id`` (inclusive)."""
        out: list[Any] = []
        stack = [node_id]
        seen: set[str] = set()
        while stack:
            nid = stack.pop()
            if nid in seen:
                continue
            seen.add(nid)
            node = self._tree.get_node(nid)
            if node is None:
                continue
            out.append(node)
            stack.extend(node.children_ids)
        return out

    def _hypothesis_of(self, node_id: str) -> str:
        node = self._tree.get_node(node_id)
        return node.hypothesis if node is not None else ""


def _stddev(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(var)
