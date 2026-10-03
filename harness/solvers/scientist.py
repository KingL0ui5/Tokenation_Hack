from inspect_ai.solver import solver, TaskState, Generate
from inspect_ai.model import get_model, ChatMessage
from inspect_ai.util import store_as

from harness.types.state import LabState

@solver
def scientist_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        lab_state = store_as(LabState)
        scientist_model = get_model()


        experiment_graph = lab_state.experiment_graph.to_text()
        
        prompt = f"""
            You are the Strategic Planner Agent (Scientist).
            Your goal is to formulate the next experimental parameters or high-level tasks based on past outcomes.
            
            OVERARCHING EXPERIMENT GRAPH:
            {experiment_graph}
            
            INNER-LOOP MANIPULATION TASK GRAPHS:
            {lab_state.task_graphs_summary}
            
            Please propose the next task or experimental parameters to be executed by the technician.
            You should update the overarching experiment graph to reflect your hypothesis and planning.
        """
        
        response = await scientist_model.generate([ChatMessage(role="user", content=prompt)])  # ty: ignore[call-non-callable]
        state.messages.append(ChatMessage(role="assistant", content=response.completion))  # ty: ignore[call-non-callable]

        return state
    
    return solve
