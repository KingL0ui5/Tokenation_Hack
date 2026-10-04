"""The technician's action-level reasoning: for a physical action that could collide (tube
handling, working between packed racks), map the different ways it could be performed BEFORE
acting, then record what actually happened -- including the mistake -- on the approach that was
tried. Closed approaches are ruled out for the rest of the run, so a mistake is made at most once.

This sits below the scientist's experiment graph: the experiment graph asks "which condition
next?", the action graph asks "how do I physically do this step without an incident?".
"""

from inspect_ai.tool import ToolError, tool
from inspect_ai.util import store_as

from harness.types.state import ActionGraph, ActionNode, Edge, LabState


@tool
def map_action():
    async def execute(action: str, approaches: list[str]) -> str:
        """Map the ways a physical action could be performed, before trying one. Creates (or
        extends) the action's graph with one node per approach; record what happens with
        record_attempt.

        Args:
            action: Short name of the physical action, e.g. "stage zncl2 tube by the plate".
            approaches: One line per way the action could be done, e.g. ["carry the tube low
                across the bench", "carry at safe height, lifting clear first", "skip the carry
                and pipette from the rack"].
        """
        if not approaches:
            raise ToolError("Give at least one approach.")
        s = store_as(LabState)
        g = s.action_graphs.get(action) or ActionGraph(action=action)
        added = []
        for a in approaches:
            nid = f"A{len(g.nodes) + 1}"
            g.nodes[nid] = ActionNode(id=nid, approach=a)
            g.edges.append(Edge(source="root", target=nid, reasoning=""))
            added.append(nid)
        s.action_graphs[action] = g
        return f"Mapped '{action}': {', '.join(f'{i}={g.nodes[i].approach!r}' for i in added)}."

    return execute


@tool
def record_attempt():
    async def execute(action: str, node: str, outcome: str, mistake: str = "",
                      close: bool = False) -> str:
        """Record what happened when an approach to an action was tried. If it failed, state the
        mistake plainly -- it is stored on the node so the same mistake is never made twice --
        and close the node if that way of doing the action is ruled out.

        Args:
            action: The action name used in map_action.
            node: The approach node that was tried, e.g. "A1".
            outcome: What happened, e.g. "collision: knocked over the enzyme tube" or "done".
            mistake: What went wrong and why, if it failed. Empty for a success.
            close: Rule this approach out for the rest of the run.
        """
        s = store_as(LabState)
        if action not in s.action_graphs:
            raise ToolError(f"No action graph for '{action}'. Map it first with map_action.")
        g = s.action_graphs[action]
        if node not in g.nodes:
            raise ToolError(f"Unknown approach '{node}' for '{action}'. Nodes: {list(g.nodes)}.")
        n = g.nodes[node]
        n.outcome = outcome
        if mistake:
            n.mistake = mistake
        if close:
            n.closed, n.closed_reason = True, mistake or outcome
        s.action_graphs[action] = g
        left = g.open_approaches
        return (f"Recorded {node}: {outcome}." + (" Approach CLOSED." if close else "")
                + (f" Open approaches: {left}." if left else " No open approaches left: ask the "
                   "scientist to re-plan this step."))

    return execute


@tool
def view_actions():
    async def execute(action: str = "") -> str:
        """View the action graphs: every approach mapped, what happened when it was tried, and
        the recorded mistakes. Check this BEFORE a physical action you have attempted before.

        Args:
            action: One action's name, or empty for all of them.
        """
        s = store_as(LabState)
        if action:
            if action not in s.action_graphs:
                return f"No action graph for '{action}'."
            return s.action_graphs[action].to_text()
        return s.action_graphs_summary

    return execute
