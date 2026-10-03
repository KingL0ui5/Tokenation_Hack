from inspect_ai.solver import solver, TaskState, Generate
from inspect_ai.model import get_model, ChatMessage
from inspect_ai.util import store_as

from harness.types.state import LabState
from harness.tools.experiment import run_experiment
from harness.tools.graph import add_reasoning, close_branch, view_graph

@solver
def technician_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        lab_state = store_as(LabState)
        technician_model = get_model()
        
        graphs_summary = "\n\n".join(
            f"Task: {task_name}\nGraph:\n{graph.to_text()}" 
            for task_name, graph in lab_state.graphs.items()
        )
        if not graphs_summary:
            graphs_summary = "No task graphs have been created yet."

        state.messages.append(ChatMessage(
            role="system", 
            content=f"""
                You are the lab technician (Inner Loop: Manipulation).
                Your job is to execute the scientist's plan using physical mechanics and trajectory vectors inside the lab envioronment.
                When you attempt an action, you must use the `run_experiment` tool.
                For each action/skill you perform (like 'pick_up_test_tube'), it will be tracked in a task-specific ReasoningGraph.

                Current Task Graphs:
                {graphs_summary}

                If an action fails, use `add_reasoning` or `close_branch` to record WHY it failed in that task's graph, 
                so you do not repeat the mistake. Focus on maintaining physical safety (e.g. not knocking over beakers).
            """
        ))  # ty: ignore[call-non-callable]
        
        response = await technician_model.generate(
            state.messages,
            tools=[run_experiment(), add_reasoning(), close_branch(), view_graph()]
        )
        
        state.messages.append(response.message)

        return state
    
    return solve
