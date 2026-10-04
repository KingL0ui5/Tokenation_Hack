from inspect_ai.solver import solver, TaskState, Generate
from inspect_ai.model import get_model, ChatMessageSystem
from inspect_ai.util import store_as

from harness.types.state import LabState
from harness.tools.plan import complete_step, view_plan
from harness.tools.lab_tools import lab_tools
from harness.tools.take_measurement import take_measurement

@solver
def technician_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        lab_state = store_as(LabState)
        technician_model = get_model()

        prompt=f"""
                You are the lab technician (Inner Loop: Manipulation).
                Your job is to execute the scientist's plan inside the lab environment using
                `dispense`, `transfer_sample` and `mix` -- the arm physically holds a pipette
                and moves liquid with it, taking a fresh disposable tip for each dispense/
                transfer and ejecting it afterwards. Call `get_lab_state` to check tips
                remaining in the box. To take your *final* measurement, you must execute the
                take_measurement tool.

                Current Experiment Plans (checklists the scientist expects you to follow):
                {lab_state.task_plans_summary}

                Follow a task's plan step by step and call `complete_step` as you finish each one
                (use `view_plan` to check progress). Once every step is checked off, the task is
                done -- You may now call `take_measurement` with the task name and the parameter
                values you actually ran at. Calling it before the plan is fully checked off still
                spends budget and returns an INVALID result with no reading, so finish the plan first.
                The number it returns is the only measurement that exists: never infer, estimate or
                invent a result from what you observed while pipetting.

                If an action fails or a step cannot be completed as planned, say so plainly rather
                than inventing a result.
            """

        state.messages.append(ChatMessageSystem(content=prompt))
        
        messages, _ = await technician_model.generate_loop(
            state.messages,
            tools=[complete_step(), view_plan(), take_measurement(), *lab_tools()]
        )
        
        state.messages.extend(messages)

        return state
    
    return solve
