from inspect_ai.solver import solver, TaskState, Generate
from inspect_ai.model import get_model, ChatMessage
from inspect_ai.util import store_as

from harness.types.state import LabState

@solver
def scientist_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        s = store_as(LabState)
        scientist_model = get_model()
        
        graphs_summary = "\n\n".join(
            f"Task: {task_name}\nGraph:\n{graph.to_text()}" 
            for task_name, graph in s.graphs.items()
        )
        if not graphs_summary:
            graphs_summary = "No tasks have been attempted yet."

        prompt = f"""
            You are the Strategic Planner Agent (Scientist).
            Your goal is to formulate the next experimental parameters or high-level tasks based on past outcomes.
            
            Current known task graphs (falsified avenues and successes):
            {graphs_summary}
            
            Please propose the next task or experimental parameters to be executed by the technician.
        """
        
        response = await scientist_model.generate([ChatMessage(role="user", content=prompt)])
        state.messages.append(ChatMessage(role="assistant", content=response.completion))

        return state
    
    return solve
