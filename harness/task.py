"""Inspect task: agentic enzyme optimisation in the simulated MuJoCo lab.

Run:  inspect eval harness/task.py --model anthropic/claude-opus-5-5
"""

from __future__ import annotations

from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageSystem
from inspect_ai.scorer import Score, Target, mean, scorer
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.util import store

from harness.lab import Lab
from harness.lab import config as C
from harness.tools.lab_tools import lab_tools

PROTOCOL = (Path(__file__).parent / "prompts" / "protocol.md").read_text()

SYSTEM = f"""You are the scientist agent running a simulated wet lab. A Franka Panda arm
with a pipette, 24-well plates (4x6, swapped between plate loads), reagent reservoirs
and stock tubes, and a kinetic plate reader (A405) are under your control through
tools. Optimise the enzyme's initial
rate following the protocol below. Record priors and every decision in the notebook
(record_prior, revise_prior, add_reasoning_node) so your reasoning forms a graph.
Use the analysis tools for all statistics. Finish with submit_report.

{PROTOCOL}"""

OBJECTIVE = """Objective: find the condition on the grid that maximises the alkaline
phosphatase initial rate (yield relative to the reference composition).
Budget: {budget} wells in total (each plate holds 24; a batch may span several plates;
confirmations use extra wells).
The enzyme is a recombinant alkaline phosphatase isoform with no curated kinetic entry."""

SCENARIOS = [
    {"id": "nominal-1", "seed": 1, "faults": {}},
    {"id": "nominal-2", "seed": 2, "faults": {}},
    {"id": "misseated-plate", "seed": 3, "faults": {"plate_offset_mm": [0.0, 9.0]}},   # 7 mm wells: spills
    {"id": "phosphate-in-water", "seed": 4, "faults": {"contaminated_reagent": {"water": 2.0}}},
    {"id": "reader-drift-plate-2", "seed": 5, "faults": {"reader_drift_batches": [2]}},
]


def _lab_for(state: TaskState) -> Lab:
    md = state.metadata
    return Lab(seed=md["seed"], budget_wells=md["budget_wells"], faults=md["faults"])


@solver
def lab_agent(include_robot_control: bool = True):
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        lab = _lab_for(state)
        state.tools = lab_tools(lab, include_robot_control=include_robot_control)
        state.messages.insert(0, ChatMessageSystem(content=SYSTEM))
        state = await generate(state, tool_calls="loop")
        store().set("lab_summary", {"budget": lab.budget(), "plates": lab.plates_run})
        return state
    return solve


@scorer(metrics={"result": [mean()], "result_x_validity": [mean()], "ci_contains_truth": [mean()]})
def optimum_scorer():
    """Result x validity term of the protocol's reward, against hidden truth."""
    async def score(state: TaskState, target: Target) -> Score:
        report = (store().get("notebook") or {}).get("report")
        if report is None:
            return Score(value={"result": 0.0, "result_x_validity": 0.0, "ci_contains_truth": 0.0},
                         explanation="no report submitted")
        lab = _lab_for(state)  # same seed -> same hidden enzyme
        best, a_star = lab.true_optimum()
        a_hat = lab.true_activity(report["optimum"])
        true_yield = a_hat / lab.true_activity(C.REFERENCE)
        lo, hi = report["yield_ci"]
        result = a_hat / a_star
        validity = float(report["last_plate_valid"]) * float(report["complete"])
        return Score(
            value={"result": result, "result_x_validity": result * validity,
                   "ci_contains_truth": float(lo <= true_yield <= hi)},
            answer=str(report["optimum"]),
            explanation=(f"true optimum {best} ({a_star:.2f}); declared achieves {a_hat:.2f}; "
                         f"true yield {true_yield:.3f} vs reported {report['yield_estimate']} "
                         f"[{lo}, {hi}]; reported-minus-true gap {report['yield_estimate'] - true_yield:+.3f}"),
            metadata={"report": report, "gap_reported_minus_true": report["yield_estimate"] - true_yield},
        )
    return score


@task
def enzyme_optimisation(budget_wells: int = 480, include_robot_control: bool = True,
                        message_limit: int = 300):
    samples = [Sample(id=s["id"], input=OBJECTIVE.format(budget=budget_wells),
                      metadata={**s, "budget_wells": budget_wells}) for s in SCENARIOS]
    return Task(dataset=samples, solver=lab_agent(include_robot_control), scorer=optimum_scorer(),
                message_limit=message_limit)
