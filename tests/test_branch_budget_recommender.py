"""Tests for fixed-budget UCB branch selection and its wiring into the
RunExecutor convergence hook.

The recommender is exercised through the existing ``IdeaTree`` model and the
wiring is exercised through the existing ``executor_run`` call site, so these
tests prove the integration rather than just the new module in isolation.
"""

from __future__ import annotations

from arbor.coordinator.config import CoordinatorConfig
from arbor.coordinator.idea_tree import IdeaTree, Node
from arbor.coordinator.branch_budget_recommender import BranchBudgetRecommender
from arbor.coordinator.tools.executor_run import _with_budget_recommendation


def _tree_with_two_branches(direction: str = "maximize") -> IdeaTree:
    """ROOT with two competing top-level branches.

    Branch ``1`` is well-sampled and strong; branch ``2`` is barely sampled
    and weak.
    """
    tree = IdeaTree(Node(id="ROOT", parent_id=None, hypothesis="root"))
    tree.meta["trunk_score"] = 0.50
    tree.meta["metric_direction"] = direction

    def add(node_id: str, parent: str, score: float | None, status: str = "done"):
        tree.add_node(Node(
            id=node_id, parent_id=parent, hypothesis=f"hyp {node_id}",
            status=status, score=score,
        ))

    # Strong, well-sampled branch.
    add("1", "ROOT", 0.80)
    add("1.1", "1", 0.82)
    add("1.2", "1", 0.81)
    # Weak, sparsely-sampled branch.
    add("2", "ROOT", 0.40)
    return tree


def _minimal_config(**overrides) -> CoordinatorConfig:
    base = dict(max_cycles=10)
    base.update(overrides)
    return CoordinatorConfig(**base)


def test_recommends_branch_with_better_reward():
    tree = _tree_with_two_branches()
    rec = BranchBudgetRecommender(tree, _minimal_config()).recommend(remaining_budget=5)

    assert rec is not None
    assert rec.best_arm == "1"
    # Ranking is ordered by UCB index, best first.
    assert [s.arm_id for s in rec.ranking][0] == "1"
    assert {s.arm_id for s in rec.ranking} == {"1", "2"}


def test_minimize_direction_flips_preference():
    tree = _tree_with_two_branches(direction="minimize")
    rec = BranchBudgetRecommender(tree, _minimal_config()).recommend(remaining_budget=5)

    assert rec is not None
    # Lower score is better under minimize → the weak-by-maximize branch wins.
    assert rec.best_arm == "2"


def test_remaining_budget_controls_exploration_weight():
    tree = _tree_with_two_branches()
    rec_kw = BranchBudgetRecommender(tree, _minimal_config())

    small = rec_kw.recommend(remaining_budget=1)
    large = rec_kw.recommend(remaining_budget=40)
    assert small is not None and large is not None

    def gap(rec):
        by_id = {s.arm_id: s for s in rec.ranking}
        return by_id["1"].ucb_index - by_id["2"].ucb_index

    # More remaining budget => stronger exploration bonus for the uncertain,
    # under-sampled branch => the leader's UCB advantage shrinks.
    assert gap(large) < gap(small)


def test_recommend_noops_without_choice():
    cfg = _minimal_config()
    # No budget left.
    tree = _tree_with_two_branches()
    assert BranchBudgetRecommender(tree, cfg).recommend(remaining_budget=0) is None

    # Only one scored branch => nothing to choose between.
    single = IdeaTree(Node(id="ROOT", parent_id=None))
    single.add_node(Node(id="1", parent_id="ROOT", status="done", score=0.7))
    assert BranchBudgetRecommender(single, cfg).recommend(remaining_budget=5) is None


def test_executor_hook_appends_recommendation():
    """The call-site helper augments a convergence intervention in place."""
    tree = _tree_with_two_branches()
    cfg = _minimal_config(max_cycles=10)

    out = _with_budget_recommendation("CONVERGENCE WARNING", tree, cfg)

    assert out.startswith("CONVERGENCE WARNING")
    assert "FIXED-BUDGET BRANCH ALLOCATION" in out
    assert "Back branch `1`" in out


def test_executor_hook_noop_returns_input_unchanged():
    single = IdeaTree(Node(id="ROOT", parent_id=None))
    single.add_node(Node(id="1", parent_id="ROOT", status="done", score=0.7))
    cfg = _minimal_config()

    out = _with_budget_recommendation("CONVERGENCE WARNING", single, cfg)
    assert out == "CONVERGENCE WARNING"
