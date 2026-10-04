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


def _briefing(lab_state: LabState) -> str:
    return f"""
You are the Strategic Planner Agent (Scientist), an autonomous experimentalist searching an
experimental space for the optimal configuration. You share this transcript with a lab technician
who physically carries out the experiments you design with a held pipette (`dispense`,
`transfer_sample`, `mix`) and then reads the result with `take_measurement`. You never operate the
lab yourself.

{experiment_space(lab_state.env, lab_state.budget)}

{lab_inventory()}

Choosing the next experiment: call `bayes_opt_suggest`, which fits a Gaussian process to every
valid measurement so far and returns the condition(s) with the highest expected improvement, plus
its predicted mean and sd. It costs no budget, so call it every turn before you plan, and treat its
suggestion as your default choice -- depart from it only for a reason you state. With fewer than
two measurements it returns a random initial design.

Every experiment is a node in a single reasoning graph; link it to the node it follows from (its
parent) with a short reasoning label. You do that through `create_plan`, which takes the condition
(`params`), the `parent` node id and the `reasoning` for the step, together with the physical
checklist the technician must perform. One plan is one experiment. Plans must be executable with
the pipette alone: name the reagent and destination well for every dispense or transfer, and leave
out anything the technician cannot do -- there is no instrument, no weighing, heating, incubation,
cleaning, storage or visual inspection. Human technicians handle that later; the reading you get
back is already adjusted for experimental deficiencies.

When the technician reports a measurement it appears in the graph (`view_graph`); add any further
inferences with `add_reasoning`. When the evidence shows a branch cannot contain the optimum, close
it with `close_branch`; closed branches cannot be extended. Before submitting, every experiment you
did not continue from must be closed with a reason explaining why it was not pursued. Keep the
number of experiments small. When confident, call `submit()`.

Read the technician's messages before planning: if it reported a failure or asked a question, answer
it concretely and revise the plan rather than inventing a new task. A measurement taken on an
incomplete plan fails: it spends budget, carries no reading and cannot inform the surrogate -- so
keep checklists short and executable.
"""


def _turn(lab_state: LabState) -> str:
    return (
        f"SCIENTIST'S TURN.\n\nReasoning graph so far:\n{lab_state.experiment_graph.to_text()}\n\n"
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

        if not briefed(state.messages, MARKER):
            state.messages.append(ChatMessageSystem(content=_briefing(lab_state)))
        state.messages.append(ChatMessageUser(content=_turn(lab_state)))

        messages, _ = await scientist_model.generate_loop(
            state.messages,
            tools=[bayes_opt_suggest(), create_plan(), view_plan(), view_graph(),
                   add_reasoning(), close_branch(), submit()]
        )

        state.messages.extend(messages)

        return state

    return solve
