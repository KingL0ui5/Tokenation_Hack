"""Lab tools: the agent's only interface to the simulated lab, driving Lok's lab_sim scene
through `LabBackend`. The simulation models physical asset positions only (MuJoCo rigid-body
physics) -- where the arm, pipette tip and containers are, and whether liquid lands where
intended. It cannot model chemistry or sense anything, so manipulating assets with the arm
(dispense, transfer_sample, mix) is the only thing possible; there is no inspection, camera,
instrument or recovery tool here. Spill/collision handling is deferred to a later iteration.

Usage:
    tools = lab_tools()            # one lab per Inspect sample, created on first use
    tools = lab_tools(backend)     # or bind an explicit LabBackend
"""

from __future__ import annotations

import json

from inspect_ai.tool import Tool, ToolError, tool

from harness.tools.lab_backend import LabBackend, LedgerEntry, current_backend


def _result(b: LabBackend, tool_name: str, args: dict, observation: dict, expected: dict | None = None,
            discrepancies: list[str] | None = None, actual: dict | None = None, ok: bool = True) -> str:
    res = {"ok": ok, "observation": observation}
    if expected:
        res["expected"] = expected
    if discrepancies:
        res["discrepancies"] = discrepancies
    b.ledger.append(LedgerEntry(round(b.clock_min, 3), tool_name, intended=args,
                                actual=actual or {}, observed=res))
    return json.dumps(res, default=str)


def _reject(b: LabBackend, tool_name: str, args: dict, message: str) -> None:
    b.ledger.append(LedgerEntry(round(b.clock_min, 3), tool_name, intended=args, actual={},
                                observed={"ok": False, "error": message}))
    raise ToolError(message)


def _container(b: LabBackend, name: str) -> str:
    """Accept "B3", "well_B3", "enzyme" or "reagent_enzyme" (reagents are the stock tubes)."""
    for candidate in (name, f"well_{name}", f"reagent_{name}"):
        if candidate in b.actual:
            return candidate
    raise ToolError(f"unknown container {name!r}. Wells: {', '.join(b.wells)}; "
                    f"reagents: {', '.join(b.reagents)}")


def _check_volume(b: LabBackend, tool_name: str, args: dict, source: str, dest: str, volume: float) -> None:
    if volume <= 0:
        _reject(b, tool_name, args, "volume must be positive")
    src, dst = b.nominal[source], b.nominal[dest]
    if src.volume_ul + 1e-6 < volume:
        _reject(b, tool_name, args, f"{source} holds {src.volume_ul:.1f} uL; requested {volume:.1f} uL")
    if dst.volume_ul + volume > dst.capacity_ul + 1e-6:
        _reject(b, tool_name, args, f"{dest} holds {dst.volume_ul:.1f} of {dst.capacity_ul:.0f} uL; "
                f"requested {volume:.1f} uL")
    if dest.startswith("well_") and dest not in b.wells_used and len(b.wells_used) >= b.budget["wells"]:
        _reject(b, tool_name, args, f"well budget spent ({b.budget['wells']} wells)")


def _move_liquid(b: LabBackend, tool_name: str, args: dict, source: str, dest: str, volume: float) -> str:
    _check_volume(b, tool_name, args, source, dest, volume)
    if source.startswith("reagent_"):
        reagent = source[len("reagent_"):]
        used = b.reagent_used_ul.get(reagent, 0.0)
        cap = b.budget["reagent_ul"].get(reagent, float("inf"))
        if used + volume > cap + 1e-6:
            _reject(b, tool_name, args, f"{reagent} budget: {cap - used:.1f} uL left; requested {volume:.1f} uL")
        b.reagent_used_ul[reagent] = used + volume
    if dest.startswith("well_"):
        b.wells_used.add(dest)

    taken = b.nominal[source].remove(volume)          # the agent's bookkeeping: as commanded
    b.nominal[dest].add(taken, volume, b.clock_min)
    out = b.pipette(source, dest, volume)              # what physically happens
    b.park()

    discrepancies = []
    for ev in out.get("events", []):
        if ev["kind"] == "spill":
            discrepancies.append(f"spill at {ev['container']}: liquid did not reach the container")
        elif ev["kind"] == "wrong_well":
            discrepancies.append(f"liquid landed in {ev['actual']} instead of {ev['intended']}")
    if not out["ok"]:
        discrepancies.append(out["reason"])
    observation = {"nominal_contents": {dest: b.nominal[dest].summary()["composition"]},
                   "lab_time_min": round(b.clock_min, 2)}
    actual = {"dest": out.get("actual_dest"), "delivered_ul": round(out.get("delivered_ul", 0.0), 3)}
    return _result(b, tool_name, args, observation, expected={"dest": dest, "volume_ul": volume},
                   discrepancies=discrepancies, actual=actual, ok=out["ok"] and not discrepancies)


def lab_tools(backend: LabBackend | None = None) -> list[Tool]:
    """All lab tools bound to `backend` (or to a per-sample backend created on first use)."""

    def B() -> LabBackend:
        return backend if backend is not None else current_backend()

    @tool
    def dispense() -> Tool:
        async def execute(reagent: str, destination: str, volume_ul: float) -> str:
            """Pipette a reagent from its stock tube into a well (fresh tip each call).

            Args:
                reagent: Reagent stock, one of the lab's reagents (see get_lab_state), e.g.
                    "dea", "pnpp", "enzyme", "mgcl2", "nacl", "water", "naoh".
                destination: Well ("B3").
                volume_ul: Volume in microlitres.
            """
            b = B()
            args = {"reagent": reagent, "destination": destination, "volume_ul": volume_ul}
            src = _container(b, reagent)
            if not src.startswith("reagent_"):
                raise ToolError(f"{reagent!r} is not a reagent; use transfer_sample to move liquid between containers")
            return _move_liquid(b, "dispense", args, src, _container(b, destination), volume_ul)
        return execute

    @tool
    def transfer_sample() -> Tool:
        async def execute(source: str, destination: str, volume_ul: float) -> str:
            """Move liquid from one container to another (well, tube or reservoir; fresh tip).

            Args:
                source: Container to take from, e.g. "rack_1" or "B3".
                destination: Container to add to.
                volume_ul: Volume in microlitres.
            """
            b = B()
            args = {"source": source, "destination": destination, "volume_ul": volume_ul}
            return _move_liquid(b, "transfer_sample", args, _container(b, source), _container(b, destination),
                                volume_ul)
        return execute

    @tool
    def mix() -> Tool:
        async def execute(container: str, cycles: int = 3) -> str:
            """Mix a well or tube by pipetting up and down.

            Args:
                container: Well ("B3") or tube ("rack_2").
                cycles: Up-down cycles.
            """
            b = B()
            args = {"container": container, "cycles": cycles}
            cid = _container(b, container)
            if b.nominal[cid].volume_ul <= 0:
                _reject(b, "mix", args, f"{cid} is empty")
            sk = b.skills
            sk.set_active_point("nozzle")
            r = sk.travel_to(cid, clearance=0.04)
            if r.ok:
                r = sk.descend(0.05)                         # nozzle into the liquid
            for _ in range(max(1, cycles)):                  # pipette up and down
                if not r.ok:
                    break
                sk.descend(-0.015)
                r = sk.descend(0.015)
            landed = b._resolve_landing(cid, sk.tip()[:2]) if r.ok else None
            if landed == cid:
                b.actual[cid].mixed = True
            b.nominal[cid].mixed = True
            sk.ascend()
            disc = [] if landed == cid else [f"tip was not inside {cid}; liquid not mixed"]
            return _result(b, "mix", args, {"lab_time_min": round(b.clock_min, 2)},
                           discrepancies=disc, actual={"mixed": landed == cid}, ok=not disc)
        return execute

    return [dispense(), transfer_sample(), mix()]


__all__ = ["lab_tools", "LabBackend", "current_backend"]
