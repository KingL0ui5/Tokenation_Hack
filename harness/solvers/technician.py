from inspect_ai.solver import solver, TaskState, Generate
from inspect_ai.model import get_model, ChatMessageSystem
from inspect_ai.util import store_as

from harness.types.state import LabState
from harness.tools.plan import complete_step, view_plan
from harness.tools.lab_tools import lab_tools

@solver
def technician_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        lab_state = store_as(LabState)
        technician_model = get_model()

        prompt=f"""
                You are the lab technician (Inner Loop: Manipulation).
                Your job is to execute the scientist's plan inside the lab environment.

                The simulation only models the physical positions of assets in the lab: where the
                arm, pipette tip and containers are, and whether liquid lands where intended. It
                cannot model chemistry, so there is no instrument and no way to take an exact
                measurement of any scientific quantity (concentration, absorbance, mass, etc.).
                The only thing you can do is use the arm to manipulate physical assets: dispense,
                transfer_sample, mix, incubate, discard, inspect (camera check of position/spill)
                and get_lab_state/get_robot_state/capture_camera for situational awareness.

                Current Experiment Plans (checklists the scientist expects you to follow):
                {lab_state.task_plans_summary}

                Follow a task's plan step by step and call `complete_step` as you finish each one
                (use `view_plan` to check progress). Once every step is checked off, the task is
                done -- there is no measurement to take at the end, just report what physically
                happened (any spills, discrepancies or incidents) back to the scientist.

                If an action fails or a step cannot be completed as planned, say so plainly rather
                than inventing a result. Focus on maintaining physical safety (e.g. not knocking
                over beakers); after a spill or fault, use `inspect`, `discard` or
                `request_human_help` before continuing.
            """

        state.messages.append(ChatMessageSystem(content=prompt))
        
        messages, _ = await technician_model.generate_loop(
            state.messages,
            tools=[complete_step(), view_plan(), *lab_tools()]
        )
        
        state.messages.extend(messages)

        return state
    
    return solve
