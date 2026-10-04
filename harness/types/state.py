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


# Mermaid node styling. A closed branch is ruled out; a failed measurement was taken before its
# plan was finished and carries no reading -- both need to stand out from live experiments.
_NODE_STYLES = {
    "closed": "fill:#f8d7da,stroke:#c92a2a,stroke-width:2px,stroke-dasharray:4,color:#5c0b0b",
    "failed": "fill:#fff3bf,stroke:#e8a90c,stroke-width:2px,color:#5c4405",
}


class ReasoningGraph(BaseModel):
    nodes: dict[str, Node] = Field(default_factory=_root)
    edges: list[Edge] = Field(default_factory=list)

    @property
    def experiments(self) -> list[Node]:
        return [n for n in self.nodes.values() if n.params is not None]

    @property
    def measured(self) -> list[Node]:
        """Experiments that carry a valid reading (an incomplete plan yields none)."""
        return [n for n in self.experiments if n.valid and n.result is not None]

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
        r = "MEASUREMENT FAILED" if n.result is None else f"{n.result:.4g}"
        tag = (" (CLOSED)" if n.closed else "") + ("" if n.valid else " (plan incomplete)")
        return f"{n.id} [{p}] -> {r}" + tag

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
                tag = "" if n.valid else "<br/><i>plan incomplete</i>"
                r = "MEASUREMENT FAILED" if n.result is None else f"{n.result:.4g}"
                lines.append(f'  {n.id}["{n.id}<br/>{p}<br/><b>{r}</b>{tag}"]')
        lines += [f'  {e.source} -->|"{q(e.reasoning)}"| {e.target}' for e in self.edges]
        groups: dict[str, list[str]] = {"closed": [], "failed": []}
        for n in self.nodes.values():
            if n.closed:
                groups["closed"].append(n.id)
            elif not n.valid:
                groups["failed"].append(n.id)
        for name, ids in groups.items():
            if ids:
                lines.append(f"  classDef {name} {_NODE_STYLES[name]}")
                lines.append(f"  class {','.join(ids)} {name}")
        return "\n".join(lines)


class Plan(BaseModel):
    """One experiment, as handed to the technician: the condition the scientist chose, where it
    belongs in the reasoning graph, and the checklist of physical steps that set it up.
    take_measurement only returns a trustworthy reading once every step here is checked off."""
    steps: list[str] = Field(default_factory=list)
    completed: list[bool] = Field(default_factory=list)
    params: dict[str, float] = Field(default_factory=dict)
    parent: str = "root"
    reasoning: str = ""

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
        cond = ", ".join(f"{k}={v:g}" for k, v in self.params.items())
        return (f"({'FINISHED' if self.finished else 'INCOMPLETE'})\n"
                f"  condition: {cond or '(none)'} | parent: {self.parent}\n{checklist}")


class LabState(StoreModel):
    env: str = ""
    budget: int = 0
    seed: int = 0
    experiment_graph: ReasoningGraph = Field(default_factory=ReasoningGraph)
    task_plans: dict[str, Plan] = Field(default_factory=dict)
    submission: dict[str, float] | None = None

    @property
    def task_plans_summary(self) -> str:
        """Serializes all task plans into a string summary for the agent prompts."""
        summary = "\n\n".join(
            f"Task/Skill: {task_name}\nPlan {plan.to_text()}"
            for task_name, plan in self.task_plans.items()
        )
        return summary if summary else "No experiment plans have been created yet."
