from inspect_ai.solver import solver, TaskState, Generate
from inspect_ai.model import get_model, ChatMessageUser
from inspect_ai.util import store_as

from harness.types.state import LabState
from harness.tools.plan import create_plan, view_plan

@solver
def scientist_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        lab_state = store_as(LabState)
        scientist_model = get_model()

        prompt = f"""
            You are the Strategic Planner Agent (Scientist).
            Your goal is to formulate the next physical task for the technician to carry out.

            The technician can only physically manipulate liquid with the held pipette
            (dispense, transfer_sample, mix) -- there is no instrument and no way to take an
            exact measurement of a scientific quantity at the end of a task. Plan accordingly:
            a task is a sequence of physical manipulation steps, not an experiment that produces
            a measured result.

            CURRENT EXPERIMENT PLANS:
            {lab_state.task_plans_summary}

            Propose the next task and use `create_plan` to give the technician an ordered checklist
            of physical steps to follow for it (use `view_plan` to check on existing plans first).
        """

        messages, _ = await scientist_model.generate_loop(
            [ChatMessageUser(content=prompt)],
            tools=[create_plan(), view_plan()]
        )
        state.messages.extend(messages)

        return state
    
    return solve
