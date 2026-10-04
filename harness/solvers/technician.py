from inspect_ai.solver import solver, TaskState, Generate
from inspect_ai.model import get_model, ChatMessageSystem, ChatMessageUser
from inspect_ai.util import store_as

from harness.solvers.context import briefed, experiment_space, lab_inventory
from harness.types.state import LabState
from harness.tools.plan import complete_step, view_plan
from harness.tools.lab_tools import lab_tools
from harness.tools.take_measurement import take_measurement

MARKER = "lab technician (Inner Loop: Manipulation)"


def _briefing(lab_state: LabState) -> str:
    return f"""
        You are the lab technician (Inner Loop: Manipulation), one of two agents sharing this transcript.
        The other is the scientist, who writes the plans you execute. You carry out a plan inside the lab
        with `dispense`, `transfer_sample` and `mix` -- the arm physically holds a pipette and moves liquid
        with it -- and then read the outcome with `take_measurement`.

        {experiment_space(lab_state.env, lab_state.budget)}

        {lab_inventory()}

        How your turn works:
        - Act with your tools. Do not end your turn with a question: no human is watching, and the scientist
        only sees your messages between turns. If a step is underspecified, choose sensible values yourself
        (any free well, any listed reagent, a volume in microlitres), say what you chose, and carry on.
        - Work one plan at a time. Perform each step, then check it off with `complete_step`; use `view_plan`
        to see what is left.
        - The pipette keeps one disposable tip across operations, so liquid carries over between every
        container that tip enters. Call `change_tip` between reagents to avoid cross-contamination;
        `get_lab_state` shows whether a tip is fitted and how many are left in the box.
        - Once every step of that plan is checked off, call `take_measurement` with the task name. You do not
        choose the condition: it, its parent node and its reasoning come from the plan the scientist wrote.
        Calling it before the plan is fully checked off still spends budget, but the measurement fails
        and returns no reading.
        - The number `take_measurement` returns is the only measurement that exists. Never infer, estimate or
        invent a result from what you observed while pipetting, and if an action fails, say so plainly.
    """


def _turn(lab_state: LabState) -> str:
    return (
        f"TECHNICIAN'S TURN.\n\nCurrent experiment plans:\n{lab_state.task_plans_summary}\n\n"
        "Execute the outstanding plan now, checking off each step as you finish it, and measure it "
        "once it is complete."
    )


@solver
def technician_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        lab_state = store_as(LabState)
        technician_model = get_model()

        if not briefed(state.messages, MARKER):
            state.messages.append(ChatMessageSystem(content=_briefing(lab_state)))
        state.messages.append(ChatMessageUser(content=_turn(lab_state)))

        messages, _ = await technician_model.generate_loop(
            state.messages,
            tools=[complete_step(), view_plan(), take_measurement(), *lab_tools()]
        )

        state.messages.extend(messages)

        return state

    return solve
