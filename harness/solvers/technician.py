from inspect_ai.solver import solver, TaskState, Generate
from inspect_ai.model import get_model, ChatMessage
from inspect_ai.util import store_as

from harness.types.state import LabState
from harness.tools.measurement import take_measurement
from harness.tools.graph import add_reasoning, close_branch, view_graph
from harness.tools.lab_tools import lab_tools

@solver
def technician_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        lab_state = store_as(LabState)
        technician_model = get_model()

        prompt=f"""
                You are the lab technician (Inner Loop: Manipulation).
                Your job is to execute the scientist's plan using physical mechanics and trajectory vectors inside the lab envioronment.
                Use `run_experiment` for tracked experiments and the lab tools for physical lab actions.
                For each action/skill you perform (like 'pick_up_test_tube'), it will be tracked in a task-specific ReasoningGraph.

                Current Task Graphs:
                {lab_state.task_graphs_summary}

                If an action fails, use `add_reasoning` or `close_branch` to record WHY it failed in that task's graph, 
                so you do not repeat the mistake. Focus on maintaining physical safety (e.g. not knocking over beakers).
            """

        state.messages.append(ChatMessage(
            role="system", 
            content=prompt
        ))  # ty: ignore[call-non-callable]
        
        messages, _ = await technician_model.generate_loop(
            state.messages,
            tools=[take_measurement(), add_reasoning(), close_branch(), view_graph(), *lab_tools()]
        )
        
        state.messages.extend(messages)

        return state
    
    return solve
