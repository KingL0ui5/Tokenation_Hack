"""Per-sample state: the reasoning graph whose nodes are experiments (inputs + output)."""

from pydantic import BaseModel, Field
from inspect_ai.util import StoreModel


class Node(BaseModel):
    id: str
    params: dict[str, float] | None = None
    result: float | None = None
    valid: bool = True
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
        tag = (" (CLOSED)" if n.closed else "") + ("" if n.valid else " (INVALID: plan incomplete)")
        return f"{n.id} [{p}] -> {'no reading' if n.result is None else f'{n.result:.4g}'}" + tag

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
                tag = "" if n.valid else "<br/><i>INVALID</i>"
                r = "no reading" if n.result is None else f"{n.result:.4g}"
                lines.append(f'  {n.id}["{n.id}<br/>{p}<br/><b>{r}</b>{tag}"]')
        lines += [f'  {e.source} -->|"{q(e.reasoning)}"| {e.target}' for e in self.edges]
        closed = [n.id for n in self.nodes.values() if n.closed]
        if closed:
            lines.append("  classDef closed fill:#eee,stroke:#999,stroke-dasharray:4")
            lines.append(f"  class {','.join(closed)} closed")
        return "\n".join(lines)


class Plan(BaseModel):
    """A technician checklist written by the scientist for a task. take_measurement only
    returns a trustworthy result for that task once every step here is checked off."""
    steps: list[str] = Field(default_factory=list)
    completed: list[bool] = Field(default_factory=list)

    @property
    def finished(self) -> bool:
        return bool(self.steps) and all(self.completed)

    @property
    def remaining(self) -> list[str]:
        return [s for s, done in zip(self.steps, self.completed) if not done]

    def to_text(self) -> str:
        checklist = "\n".join(
            f"  [{'x' if done else ' '}] {i + 1}. {step}"
            for i, (step, done) in enumerate(zip(self.steps, self.completed))
        )
        return f"({'FINISHED' if self.finished else 'INCOMPLETE'})\n{checklist}"


class LabState(StoreModel):
    env: str = ""
    budget: int = 0
    seed: int = 0
    experiment_graph: ReasoningGraph = Field(default_factory=ReasoningGraph)
    task_graphs: dict[str, ReasoningGraph] = Field(default_factory=dict)
    task_plans: dict[str, Plan] = Field(default_factory=dict)
    submission: dict[str, float] | None = None

    @property
    def task_graphs_summary(self) -> str:
        """Serializes all task graphs into a string summary for the agent prompts."""
        summary = "\n\n".join(
            f"Task/Skill: {task_name}\nGraph:\n{graph.to_text()}" 
            for task_name, graph in self.task_graphs.items()
        )
        return summary if summary else "No manipulation tasks have been attempted yet."

    @property
    def task_plans_summary(self) -> str:
        """Serializes all task plans into a string summary for the agent prompts."""
        summary = "\n\n".join(
            f"Task/Skill: {task_name}\nPlan {plan.to_text()}"
            for task_name, plan in self.task_plans.items()
        )
        return summary if summary else "No experiment plans have been created yet."
