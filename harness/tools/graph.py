from inspect_ai.tool import ToolError, tool
from inspect_ai.util import store_as

from harness.types.state import Edge, LabState


@tool
def add_reasoning():
    async def execute(source: str, target: str, reasoning: str) -> str:
        """Add a reasoning edge between two existing nodes of the reasoning graph.

        Args:
            source: Id of the node the reasoning starts from.
            target: Id of the node the reasoning leads to.
            reasoning: The inference linking the two nodes.
        """
        s = store_as(LabState)
        g = s.experiment_graph
        for nid in (source, target):
            if nid not in g.nodes:
                raise ToolError(f"Unknown node '{nid}'.")
        g.edges.append(Edge(source=source, target=target, reasoning=reasoning))
        s.experiment_graph = g
        return f"Added edge {source} -> {target}."

    return execute


@tool
def close_branch():
    async def execute(node: str, reason: str) -> str:
        """Close a branch: mark a node and all its descendants as unable to contain the optimum. Closed branches cannot be extended.

        Args:
            node: Id of the node at the top of the branch.
            reason: Why this branch cannot contain the optimum.
        """
        s = store_as(LabState)
        g = s.experiment_graph
        if node not in g.nodes:
            raise ToolError(f"Unknown node '{node}'.")
        closed = g.close(node, reason)
        s.experiment_graph = g
        return f"Closed: {', '.join(closed) or 'nothing (already closed)'}."

    return execute


@tool
def view_graph():
    async def execute() -> str:
        """View the reasoning graph: experiment nodes (inputs -> result), reasoning edges and closed branches."""
        return store_as(LabState).experiment_graph.to_text()

    return execute
