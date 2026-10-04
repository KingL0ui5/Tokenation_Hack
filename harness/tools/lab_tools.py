"""Lab tools: the agent's only interface to the simulated lab, driving Lok's lab_sim scene
through `LabBackend`. Nothing in the scene has a free joint, so no container can be grasped
or knocked over -- the only thing physically simulated is the held pipette's nozzle reaching
named sites. There is no liquid, no volume and no instrument: a call reports only whether it
succeeded, never a fabricated spill or reading, and never the robot/pipette's internal state
(tilt, tracking error, joint positions) that caused a failure -- that diagnostic detail is
recorded in the hidden ledger only, never returned to the agent.

Usage:
    tools = lab_tools()            # one lab per Inspect sample, created on first use
    tools = lab_tools(backend)     # or bind an explicit LabBackend
"""

from __future__ import annotations

import json

from inspect_ai.tool import Tool, ToolError, tool

from harness.tools.lab_backend import LabBackend, LedgerEntry, current_backend


def _result(b: LabBackend, tool_name: str, args: dict, ok: bool, reason: str | None = None,
            **ledger_only) -> str:
    res = {"ok": ok, "lab_time_min": round(b.clock_min, 2)}
    # `reason` (tilt/tracking-error/IK diagnostics) and `ledger_only` (which tip was used, which
    # containers it touched) are hidden lab truth, not agent-visible.
    observed = {**res, "reason": reason} if reason else dict(res)
    observed.update(ledger_only)
    b.ledger.append(LedgerEntry(round(b.clock_min, 3), tool_name, intended=args, observed=observed))
    return json.dumps(res)


def _site(b: LabBackend, name: str) -> str:
    """Accept "B3", "well_B3", "enzyme" or "reagent_enzyme" (reagents are the stock tubes)."""
    for candidate in (name, f"well_{name}", f"reagent_{name}"):
        if candidate in b.sites:
            return candidate
    raise ToolError(f"unknown site {name!r}. Wells: {', '.join(b.wells)}; reagents: {', '.join(b.reagents)}")


def lab_tools(backend: LabBackend | None = None) -> list[Tool]:
    """All lab tools bound to `backend` (or to a per-sample backend created on first use)."""

    def B() -> LabBackend:
        return backend if backend is not None else current_backend()

    @tool
    def dispense() -> Tool:
        async def execute(reagent: str, destination: str, volume_ul: float) -> str:
            """Pipette a reagent from its stock tube into a well (intended volume only;
            not tracked).

            Args:
                reagent: One of the lab's reagents, e.g. "enzyme", "pnpp", "mgcl2", "water".
                destination: Well, e.g. "B3".
                volume_ul: Intended volume in microlitres.
            """
            b = B()
            args = {"reagent": reagent, "destination": destination, "volume_ul": volume_ul}
            src = _site(b, reagent)
            if src not in b.contract.reagents.values():
                raise ToolError(f"{reagent!r} is not a reagent; use transfer_sample to move liquid between containers")
            r = b.pipette(src, _site(b, destination))
            return _result(b, "dispense", args, r["ok"], r.get("reason"),
                           tip_slot=r.get("tip_slot"), touched=r.get("touched", []))
        return execute

    @tool
    def transfer_sample() -> Tool:
        async def execute(source: str, destination: str, volume_ul: float) -> str:
            """Move liquid from one container to another (intended volume only; not tracked).

            Args:
                source: Container to take from, e.g. "B3".
                destination: Container to add to.
                volume_ul: Intended volume in microlitres.
            """
            b = B()
            args = {"source": source, "destination": destination, "volume_ul": volume_ul}
            r = b.pipette(_site(b, source), _site(b, destination))
            return _result(b, "transfer_sample", args, r["ok"], r.get("reason"),
                           tip_slot=r.get("tip_slot"), touched=r.get("touched", []))
        return execute

    @tool
    def mix() -> Tool:
        async def execute(container: str, cycles: int = 3) -> str:
            """Mix a container by pipetting up and down.

            Args:
                container: Well or tube, e.g. "B3".
                cycles: Up-down cycles.
            """
            b = B()
            args = {"container": container, "cycles": cycles}
            r = b.mix(_site(b, container), cycles)
            return _result(b, "mix", args, r["ok"], r.get("reason"))
        return execute

    @tool
    def get_lab_state() -> Tool:
        async def execute() -> str:
            """Report lab state visible to the agent: elapsed time and the disposable-tip box."""
            b = B()
            ts = b.skills.tip_status()
            res = {"lab_time_min": round(b.clock_min, 2),
                   "tips_remaining": ts["tips_remaining"], "box_empty": ts["box_empty"]}
            return json.dumps(res)
        return execute

    return [dispense(), transfer_sample(), mix(), get_lab_state()]


__all__ = ["lab_tools", "LabBackend", "current_backend"]
