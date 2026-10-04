"""Per-sample state: the reasoning graph whose nodes are experiments (inputs + output)."""

from pydantic import BaseModel, Field
from inspect_ai.util import StoreModel


class Node(BaseModel):
    id: str
    params: dict[str, float] | None = None
    result: float | None = None
    closed: bool = False
    closed_reason: str | None = None


class Edge(BaseModel):
    source: str
    target: str
    reasoning: str


def _root() -> dict[str, Node]:
    return {"root": Node(id="root")}


class ReasoningGraph(BaseModel):
    nodes: dict[str, Node] = Field(default_factory=_root)
    edges: list[Edge] = Field(default_factory=list)

    @property
    def experiments(self) -> list[Node]:
        return [n for n in self.nodes.values() if n.params is not None]

    def open_leaves(self) -> list[str]:
        """Experiment nodes that were neither extended nor closed."""
        sources = {e.source for e in self.edges}
        return [n.id for n in self.experiments if not n.closed and n.id not in sources]

    def close(self, node_id: str, reason: str) -> list[str]:
        """Close a branch (Hintikka-style): the node and all its descendants."""
        closed, stack = [], [node_id]
        while stack:
            nid = stack.pop()
            if self.nodes[nid].closed:
                continue
            self.nodes[nid].closed, self.nodes[nid].closed_reason = True, reason
            closed.append(nid)
            stack += [e.target for e in self.edges if e.source == nid]
        return closed

    def _label(self, n: Node) -> str:
        if n.params is None:
            return "root"
        p = ", ".join(f"{k}={v:g}" for k, v in n.params.items())
        return f"{n.id} [{p}] -> {n.result:.4g}" + (" (CLOSED)" if n.closed else "")

    def to_text(self) -> str:
        lines = [self._label(n) for n in self.nodes.values()]
        lines += [f"{e.source} -> {e.target}: {e.reasoning}" for e in self.edges]
        closed = [f"{n.id}: {n.closed_reason}" for n in self.nodes.values() if n.closed]
        if closed:
            lines += ["Closed branches:"] + closed
        return "\n".join(lines)

    def to_mermaid(self) -> str:
        def q(s: str) -> str:
            return s.replace('"', "'")[:120]

        lines = ["graph TD"]
        for n in self.nodes.values():
            if n.params is None:
                lines.append(f'  {n.id}(("start"))')
            else:
                p = "<br/>".join(f"{k}={v:g}" for k, v in n.params.items())
                lines.append(f'  {n.id}["{n.id}<br/>{p}<br/><b>{n.result:.4g}</b>"]')
        lines += [f'  {e.source} -->|"{q(e.reasoning)}"| {e.target}' for e in self.edges]
        closed = [n.id for n in self.nodes.values() if n.closed]
        if closed:
            lines.append("  classDef closed fill:#f8d7da,stroke:#c92a2a,stroke-width:2px,"
                         "stroke-dasharray:4,color:#5c0b0b")
            lines.append(f"  class {','.join(closed)} closed")
        return "\n".join(lines)


class BOState(StoreModel):
    env: str = ""
    budget: int = 0
    seed: int = 0
    graph: ReasoningGraph = Field(default_factory=ReasoningGraph)
    submission: dict[str, float] | None = None
