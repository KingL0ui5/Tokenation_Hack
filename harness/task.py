from inspect_ai import task, Task
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageSystem
from inspect_ai.solver import solver, TaskState, Generate
from inspect_ai.util import store_as

from bo_eval.env import get_env
from harness.scorer import lab_scorer
from harness.solvers import scientist_solver, technician_solver
from harness.solvers.scientist import briefing as scientist_briefing
from harness.solvers.technician import briefing as technician_briefing
from harness.types.state import LabState

@solver
def init_lab_state(budget: int = 30, env: str = "upo_abts", seed: int = 0):

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        lab_state = store_as(LabState)
        lab_state.budget = budget
        lab_state.env = env
        lab_state.seed = seed

        return state
    
    return solve

@solver
def brief_agents():
    """Both briefings as ONE system message at the very top of the transcript.

    Appending a system message mid-conversation violates the Anthropic API's placement rules (it
    gets hoisted to the top-level system field with a warning) and changes the prompt prefix,
    which invalidates replayed thinking blocks. Briefing once, up front, avoids both."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        lab_state = store_as(LabState)
        text = (
            "Two agents share this transcript and take alternating turns. Each turn is addressed to "
            "one of them by a message beginning SCIENTIST'S TURN or TECHNICIAN'S TURN; act only as "
            "the agent whose turn it is, using the tools you have been given for that turn.\n\n"
            f"=== SCIENTIST ==={scientist_briefing(lab_state)}\n"
            f"=== TECHNICIAN ==={technician_briefing(lab_state)}"
        )
        state.messages.insert(0, ChatMessageSystem(content=text))
        return state

    return solve


@solver
def lab_loop(max_rounds: int | None = None):
    """Alternate scientist and technician until the experiment budget is spent or the scientist
    submits -- one round is one experiment, so the budget, not a fixed round count, is what limits
    the search. `max_rounds` caps it (default: the budget) in case a round measures nothing."""

    scientist, technician = scientist_solver(), technician_solver()

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        lab_state = store_as(LabState)

        def done() -> bool:
            return (lab_state.submission is not None
                    or len(lab_state.experiment_graph.experiments) >= lab_state.budget)

        for _ in range(max_rounds if max_rounds is not None else lab_state.budget):
            if done():
                break
            state = await scientist(state, generate)
            if done():
                break
            state = await technician(state, generate)

        if lab_state.submission is None:      # out of budget: one last turn to submit an answer
            state = await scientist(state, generate)

        return state

    return solve


@task
def autonomous_lab_task(env: str = "upo_abts", budget: int = 30, seed: int = 0,
                        tolerance: float = 0.0, max_rounds: int | None = None,
                        graph_dir: str = "logs/graphs"):
    return Task(
        dataset=[Sample(id=env, input=get_env(env).prompt(budget), metadata={"env": env, "seed": seed})],
        setup=init_lab_state(budget, env, seed),
        plan=[brief_agents(), lab_loop(max_rounds)],
        scorer=lab_scorer(graph_dir, tolerance),
    )
