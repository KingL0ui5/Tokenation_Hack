from inspect_ai.tool import ToolError, tool
from inspect_ai.util import store_as

from harness.types.state import LabState, Edge, ReasoningGraph


@tool
def add_reasoning():
    async def execute(task_name: str, source: str, target: str, reasoning: str) -> str:
        """Add a reasoning edge between two existing nodes of the reasoning graph for a specific task.

        Args:
            task_name: Name of the task
            source: Id of the node the reasoning starts from.
            target: Id of the node the reasoning leads to.
            reasoning: The inference linking the two nodes.
        """
        s = store_as(LabState)

        if task_name not in s.task_graphs:
            raise ToolError(f"Task graph '{task_name}' does not exist.")
        
        g = s.task_graphs[task_name]

        for nid in (source, target):
            if nid not in g.nodes:
                raise ToolError(f"Unknown node '{nid}'.")
            
        g.edges.append(Edge(source=source, target=target, reasoning=reasoning))
        s.task_graphs[task_name] = g
        
        return f"Added edge {source} -> {target} to task '{task_name}'."

    return execute


@tool
def close_branch():
    async def execute(task_name: str, node: str, reason: str) -> str:
        """Close a branch: mark a node and all its descendants as unable to contain the optimum. Closed branches cannot be extended.

        Args:
            task_name: Name of the task
            node: Id of the node at the top of the branch.
            reason: Why this branch cannot contain the optimum.
        """
        s = store_as(LabState)
        if task_name not in s.task_graphs:
            raise ToolError(f"Task graph '{task_name}' does not exist.")
        g = s.task_graphs[task_name]
        if node not in g.nodes:
            raise ToolError(f"Unknown node '{node}'.")
        closed = g.close(node, reason)
        s.task_graphs[task_name] = g
        return f"Closed: {', '.join(closed) or 'nothing (already closed)'} in task '{task_name}'."

    return execute


@tool
def view_graph():
    async def execute(task_name: str) -> str:
        """View the reasoning graph: experiment nodes (inputs -> result), reasoning edges and closed branches for a specific task.
        
        Args:
            task_name: Name of the task graph to view
        """
        s = store_as(LabState)
        if task_name not in s.task_graphs:
            return f"Task graph '{task_name}' is empty."
        return s.task_graphs[task_name].to_text()

    return execute
