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
            a task is a sequence of physical manipulation steps, once the agent has completely finished
            your steps it obtains the result by using the take_measurement tool. The result is suitably 
            adjusted for experimental deficiencies. Steps such as cleaning the workspace, storage, 
            environment considerations and other factors should be ignored. Human technicians will do this later.
            
            The technician must be able to execute the instructions you give using the *only* tools that it has available: Dispense, 
            Transfer Sample and Mix. It should not complete instructions that are beyond its physical limitation. 

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
