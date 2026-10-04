from inspect_ai.solver import solver, TaskState, Generate
from inspect_ai.model import get_model, ChatMessageSystem, ChatMessageUser
from inspect_ai.util import store_as

from harness.solvers.context import briefed, experiment_space, lab_inventory
from harness.types.state import LabState
from harness.tools.bayes_opt import bayes_opt_suggest
from harness.tools.graph import add_reasoning, close_branch, view_graph
from harness.tools.plan import create_plan, view_plan
from harness.tools.submit import submit

MARKER = "Strategic Planner Agent (Scientist)"


def briefing(lab_state: LabState) -> str:
    return f"""
        You are the Strategic Planner Agent (Scientist), an autonomous experimentalist searching an
        experimental space for the optimal configuration. You share this transcript with a lab technician
        who physically carries out the experiments you design with a held pipette (`dispense`,
        `transfer_sample`, `mix`) and then reads the result with `take_measurement`. You never operate the
        lab yourself. 

        Every experiment that you plan is a node in a reasoning graph; link it to the node it follows from (its parent) with a short
        reasoning label. When the evidence shows a branch cannot contain the optimum, close it with `close_branch`;
        closed branches cannot be extended. Before submitting, every experiment you did not continue from must be
        closed with a reason explaining why it was not pursued. Never close the branch that contains your highest result so far.

        Choosing the next experiment: you must call `bayes_opt_suggest` every turn before you plan. It fits a Gaussian process to all prior results to identify high-value target zones and costs no budget. While you should heavily weigh its suggestions, **you are NOT required to use its parameters verbatim.** If your scientific reasoning identifies a clear trend, a needed single-variable isolation, or a flaw in the optimiser's suggestion (e.g., exploring a known dead-zone), you may manually adjust the parameters before passing them to `create_plan`.

        {experiment_space(lab_state.env, lab_state.budget)}
        Your goal is to find the absolute global maximum. Use your budget effectively to map uncertain regions and escape local optima. Call `submit()` only when the global optimum is conclusively isolated or your budget is exhausted.

        {lab_inventory()}
        Cross-reference your planned parameters with the available inventory. If a parameter requested by the optimiser cannot be perfectly achieved (e.g., missing a specific chemical stock or temperature controller), explicitly note this limitation in your reasoning and instruct the technician to use the closest physical proxy available.

        Read the technician's messages before planning: if they report a failure or ask a question, answer
        it concretely and revise the plan rather than inventing a new task. A measurement taken on an
        incomplete plan fails: it spends budget, carries no reading, and cannot inform the surrogate -- so
        keep checklists short, precise, and physically executable.
    """


def _turn(lab_state: LabState) -> str:
    used = len(lab_state.experiment_graph.experiments)
    return (
        f"SCIENTIST'S TURN. Budget: {used}/{lab_state.budget} experiments used, "
        f"{lab_state.budget - used} left.\n\n"
        f"Reasoning graph so far:\n{lab_state.experiment_graph.to_text()}\n\n"
        f"Current experiment plans:\n{lab_state.task_plans_summary}\n\n"
        "Answer anything the technician raised above, call `bayes_opt_suggest`, then leave exactly "
        "one plan ready to execute -- `create_plan` with the condition, its parent node and your "
        "reasoning. Submit instead if you are confident and every open branch is closed."
    )


@solver
def scientist_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        lab_state = store_as(LabState)
        scientist_model = get_model()

        state.messages.append(ChatMessageUser(content=_turn(lab_state)))

        messages, _ = await scientist_model.generate_loop(
            state.messages,
            tools=[bayes_opt_suggest(), create_plan(), view_plan(), view_graph(),
                   add_reasoning(), close_branch(), submit()]
        )

        state.messages.extend(messages)

        return state

    return solve
