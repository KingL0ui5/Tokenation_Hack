"""Inspect tools for the simulated enzyme lab.

`lab_tools(lab)` returns every tool bound to one Lab instance (one per sample).
Groups:
  observe   - status, deck layout, robot state, event log, camera, results, dataset
  robot     - low-level arm and pipette control (classical IK, no learned policy)
  protocol  - design_batch, run_plate, confirmations
  analysis  - fit_model, suggest_ucb, mutual_information
  notebook  - priors, revisions and reasoning-graph nodes, final report

Each protocol step is its own tool so a skipped step is visible in the trace.
"""

from __future__ import annotations

import base64
import json
import math
from typing import Any

from inspect_ai.tool import ContentImage, Tool, ToolError, tool
from inspect_ai.util import store

from harness.lab import Lab
from harness.lab import analysis as A
from harness.lab import config as C


def _j(obj: Any) -> str:
    return json.dumps(obj, default=str)


def _notebook() -> dict:
    nb = store().get("notebook")
    if nb is None:
        nb = {"nodes": [], "report": None}
        store().set("notebook", nb)
    return nb


def _add_node(kind: str, **fields) -> dict:
    nb = _notebook()
    node = {"id": f"N{len(nb['nodes']) + 1:03d}", "kind": kind, **fields}
    nb["nodes"].append(node)
    store().set("notebook", nb)
    return node


def _check_condition(cond: dict) -> None:
    problems = C.validate_condition(cond)
    if problems:
        raise ToolError("; ".join(problems))


# =================================================================== observe

@tool
def get_lab_status(lab: Lab) -> Tool:
    async def execute() -> str:
        """Report budget, plates run, lab clock, observation count and pending mandatory stops.

        Returns:
            JSON summary of the lab's current state.
        """
        pending = [b.result["mandatory_stop"] for b in lab.batches.values()
                   if b.result and b.result.get("mandatory_stop")]
        return _j({**lab.budget(), "lab_clock_min": round(lab.world.clock_min, 2),
                   "valid_observations": len(lab.observations),
                   "plates": lab.plates_run, "mandatory_stops_raised": pending,
                   "consecutive_control_failures": lab.consecutive_control_failures})
    return execute


@tool
def get_deck_layout(lab: Lab) -> Tool:
    async def execute() -> str:
        """Positions of the plate, every reagent reservoir and the safe travel height.

        Returns:
            JSON with world-frame coordinates in metres.
        """
        return _j(lab.deck_layout())
    return execute


@tool
def get_robot_state(lab: Lab) -> Tool:
    async def execute() -> str:
        """Read the arm's proprioception and the pipette state.

        Returns:
            JSON with joint positions/velocities/torques, tip position, tool axis,
            gripper gap, pipette contents, simulation time and lab clock.
        """
        return _j(lab.world.robot_state())
    return execute


@tool
def get_event_log(lab: Lab) -> Tool:
    async def execute(since_index: int = 0, kinds: list[str] | None = None) -> str:
        """Read the robot/lab event log (spills, collisions, IK failures, misses...).

        Args:
            since_index: Return events from this index onward.
            kinds: Optional filter, e.g. ["spill", "collision"].

        Returns:
            JSON list of events with their index.
        """
        evs = [{"index": i, **e.as_dict()} for i, e in enumerate(lab.world.events)
               if i >= since_index and e.kind != "tip_change" and (not kinds or e.kind in kinds)]
        return _j({"total_events": len(lab.world.events), "events": evs[-200:]})
    return execute


@tool
def capture_camera(lab: Lab) -> Tool:
    async def execute(camera: str = "front") -> ContentImage | str:
        """Render an image of the lab from a fixed camera. Wells show liquid level, and
        are coloured yellow by their last A405 read.

        Args:
            camera: "front" (arm and bench), "side", or "plate_top" (top-down over the plate).

        Returns:
            A PNG image.
        """
        if camera not in lab.world.cameras:
            raise ToolError(f"camera must be one of {lab.world.cameras}")
        try:
            png = lab.world.render_png(camera)
        except Exception as e:  # no OpenGL context available
            return f"camera unavailable: {e}"
        return ContentImage(image="data:image/png;base64," + base64.b64encode(png).decode())
    return execute


@tool
def get_plate_result(lab: Lab) -> Tool:
    async def execute(batch_id: str, include_raw_reads: bool = False) -> str:
        """Fetch the full result of a plate that has already been run.

        Args:
            batch_id: Batch identifier returned by design_batch.
            include_raw_reads: Include per-well A405 time series and fits.

        Returns:
            JSON plate result.
        """
        b = lab.batches.get(batch_id)
        if b is None or b.result is None:
            raise ToolError(f"no result for {batch_id!r}")
        res = dict(b.result)
        if not include_raw_reads:
            res.pop("raw", None)
        return _j(res)
    return execute


@tool
def get_observations(lab: Lab) -> Tool:
    async def execute() -> str:
        """All condition-level yields from plates that passed the validity gate.

        Returns:
            JSON list of {batch_id, condition, yield}. Invalid plates are excluded.
        """
        return _j(lab.observations)
    return execute


# ===================================================================== robot

@tool
def move_tip(lab: Lab) -> Tool:
    async def execute(x: float, y: float, z: float, via_safe_height: bool = True) -> str:
        """Move the pipette tip to a world position with the tool pointing down (IK + servo).

        Args:
            x: Target x in metres (Panda base frame).
            y: Target y in metres.
            z: Target z in metres. Deck surfaces are ~0.04-0.05 m.
            via_safe_height: Lift to the safe travel height first, translate, then descend.

        Returns:
            JSON with achieved tip position, tracking error, collisions.
        """
        target = [x, y, z]
        r = lab.world.travel(target) if via_safe_height else lab.world.move_tip(target)
        if not r["ok"]:
            raise ToolError(_j(r))
        return _j(r)
    return execute


@tool
def move_tip_to(lab: Lab) -> Tool:
    async def execute(location: str) -> str:
        """Move the pipette tip above a named location.

        Args:
            location: "home", "well:<A1..D6>", "reservoir:<reagent name>" or a lab_sim
                station: "station_reader", "station_incubator", "waste", "pipette_grip".

        Returns:
            JSON with achieved tip position and tracking error.
        """
        w = lab.world
        if location == "home":
            return _j(w.home())
        kind, _, name = location.partition(":")
        if kind == "well" and name in w.wells:
            r = w.travel(w.nominal_well_pos(name))
        elif kind == "reservoir" and name in w.reservoir_index:
            r = w.travel(w.reservoir_pos(name))
        elif location in ("station_reader", "station_incubator", "station_bench", "waste", "pipette_grip"):
            r = w.travel(w.site_pos(location))
        else:
            raise ToolError(f"unknown location {location!r}")
        if not r["ok"]:
            raise ToolError(_j(r))
        return _j(r)
    return execute


@tool
def set_gripper(lab: Lab) -> Tool:
    async def execute(open: bool) -> str:
        """Open or close the Panda gripper.

        Args:
            open: True to open, False to close.

        Returns:
            JSON with the resulting finger gap.
        """
        return _j(lab.world.set_gripper(open))
    return execute


@tool
def aspirate(lab: Lab) -> Tool:
    async def execute(reagent: str, volume_ul: float) -> str:
        """Move to a reagent reservoir and aspirate into the pipette tip.

        Manual pipetting is for diagnostics; wells filled by hand are not analysed by run_plate.

        Args:
            reagent: Reservoir name, see get_deck_layout.
            volume_ul: Volume in microlitres.

        Returns:
            JSON with tip contents.
        """
        if volume_ul <= 0:
            raise ToolError("volume must be positive")
        r = lab.world.aspirate(reagent, volume_ul)
        if not r["ok"]:
            raise ToolError(_j(r))
        return _j(r)
    return execute


@tool
def dispense(lab: Lab) -> Tool:
    async def execute(well: str, volume_ul: float) -> str:
        """Move to a well of the plate on the bench and dispense from the tip. Reports where the
        liquid actually went: the intended well, a neighbouring well, or a spill.

        Args:
            well: Well name A1..D6 on the current plate.
            volume_ul: Volume in microlitres.

        Returns:
            JSON with intended and actual well, delivered volume, spill flag and positional error.
        """
        if well not in lab.world.wells:
            raise ToolError(f"unknown well {well!r}")
        r = lab.world.dispense(well, volume_ul)
        if not r["ok"]:
            raise ToolError(_j(r))
        r["delivered_ul"] = round(r["delivered_ul"], 2)
        return _j(r)
    return execute


@tool
def change_tip(lab: Lab) -> Tool:
    async def execute() -> str:
        """Eject the current tip and pick up a fresh one.

        Returns:
            JSON pipette state.
        """
        lab.world.change_tip()
        return _j(lab.world.robot_state()["pipette"])
    return execute


# ================================================================== protocol

@tool
def design_batch(lab: Lab) -> Tool:
    async def execute(conditions: list[dict[str, Any]], controls: list[str], replicates: int = 3,
                      avoid_edges: bool = False, off_grid_reasons: dict[str, str] | None = None) -> str:
        """Validate a batch and lay it out over 24-well plates (4x6). No robot motion happens yet.

        Each condition is {"buffer": DEA|Tris|Glycine|PBS, "pH": 7.0|8.0|9.0|10.0,
        "substrate"|"MgCl2"|"ZnCl2"|"NaCl"|"glycerol": L1|L2|L3|L4, "temperature": 25|30|37|45}.
        Levels are fractions of each maximum (L1=10%, L2=30%, L3=50%, L4=100%).
        Every condition passes the hazard interlock. The batch is rejected if any
        mandatory control is missing: reference, blanks, positive, standard_curve,
        carry_over (carry_over not needed on the first plate). A batch larger than one
        plate runs over several plate loads (up to 6); wells are named "P<plate>:<well>",
        e.g. "P2:B3", and the controls cover the whole batch.

        Args:
            conditions: Conditions to test (each run in replicate).
            controls: Control types to include.
            replicates: Replicates per condition (protocol default 3).
            avoid_edges: Keep wells off each plate's outer ring (8 of 24 wells per plate remain; no evaporation artefact).
            off_grid_reasons: Map of condition index (as string) to the reason for an off-grid value.

        Returns:
            JSON with batch_id and plate map, or the reasons for rejection.
        """
        res = lab.design_batch(conditions, controls, replicates, avoid_edges, off_grid_reasons)
        if not res["accepted"]:
            raise ToolError(_j(res["errors"]))
        if off_grid_reasons:
            _add_node("deviation", statement="off-grid values requested", reasons=off_grid_reasons,
                      batch_id=res["batch_id"])
        return _j(res)
    return execute


@tool
def run_plate(lab: Lab) -> Tool:
    async def execute(batch_id: str) -> str:
        """Execute a designed batch: the robot pipettes every well, incubates per
        temperature, starts reactions with enzyme and the reader takes kinetic A405 reads.

        Returns per-condition yields (rate relative to the on-plate reference after
        blank subtraction), triplicate CVs and dropped outliers (with edge-well flags),
        the four-part plate validity gate, execution events (spills, collisions)
        and any mandatory stop. Data from an invalid plate never enter the model.

        Args:
            batch_id: Identifier from design_batch.

        Returns:
            JSON plate result (use get_plate_result for raw reads).
        """
        res = lab.run_plate(batch_id)
        if "error" in res:
            raise ToolError(res["error"])
        out = {k: v for k, v in res.items() if k != "raw"}
        ex = dict(out["execution"])
        ex["events"] = ex["events"][:30]
        out["execution"] = ex
        return _j(out)
    return execute


def _confirm_tool(fn_name: str, doc: str):
    @tool(name=fn_name)
    def factory(lab: Lab) -> Tool:
        async def execute(condition: dict[str, Any], spike_mM: float = 0.05) -> str:
            _check_condition(condition)
            fn = getattr(lab, fn_name)
            res = fn(condition, spike_mM=spike_mM) if fn_name == "spike_recovery" else fn(condition)
            if "error" in res:
                raise ToolError(_j(res["error"]))
            res["execution"] = {k: v for k, v in res["execution"].items() if k != "events"}
            return _j(res)
        execute.__doc__ = doc
        return execute
    return factory


enzyme_titration = _confirm_tool("enzyme_titration", """Orthogonal confirmation: run a condition at 0.5x, 1x and 2x enzyme (triplicates + blanks, ~11 wells).

        A real rate scales proportionally with enzyme amount.

        Args:
            condition: The condition to confirm.
            spike_mM: Unused for this tool.

        Returns:
            JSON with net slopes per enzyme amount, proportionality R2 and the 2x/1x ratio.
        """)

spike_recovery = _confirm_tool("spike_recovery", """Orthogonal confirmation: spike known pNP product into the condition (no enzyme) and measure recovery (~8 wells).

        Args:
            condition: The condition to test.
            spike_mM: Product concentration spiked in, mM.

        Returns:
            JSON with recovered concentrations and percent recovery.
        """)

dual_wavelength = _confirm_tool("dual_wavelength", """Orthogonal confirmation: read the condition at 405 nm and 490 nm (reference) to remove scattering (~3 wells).

        Args:
            condition: The condition to test.
            spike_mM: Unused for this tool.

        Returns:
            JSON with 405 nm slopes, background-corrected slopes and A490.
        """)


# ================================================================== analysis

def _dataset(lab: Lab):
    if len(lab.observations) < 3:
        raise ToolError("need at least 3 valid observations; run a valid plate first")
    return [o["condition"] for o in lab.observations], [o["yield"] for o in lab.observations]


@tool
def fit_model(lab: Lab) -> Tool:
    async def execute(cross_validate: bool = True) -> str:
        """Fit a GP surrogate (Matern 3/2, ARD, learned noise) to all valid yields.

        Args:
            cross_validate: Also report 5-fold cross-validated R2.

        Returns:
            JSON with length scales (short = influential), noise, predicted optimum
            with SD, best observed condition and CV R2.
        """
        conds, ys = _dataset(lab)
        gp = A.fit_gp(conds, ys)
        out = A.describe_gp(gp, conds, ys)
        if cross_validate:
            out["cv_r2"] = A.cross_validated_r2(conds, ys)
        store().set("last_model", {"n": len(ys), "cv_r2": out.get("cv_r2"),
                                   "best_observed": out["best_observed"]})
        return _j(out)
    return execute


@tool
def suggest_ucb(lab: Lab) -> Tool:
    async def execute(n: int = 12, exploitation: float = 1.0, exploration: float = math.sqrt(2),
                      exclude: dict[str, list[Any]] | None = None) -> str:
        """Propose the next batch by upper confidence bound over the full 65,536-condition grid.

        score = exploitation * mean + exploration * sd, with a Kriging-believer
        update between picks so the batch is diverse. Already-tested conditions
        are skipped.

        Args:
            n: Number of conditions to propose.
            exploitation: Weight on the predicted mean.
            exploration: Weight on the predicted SD (protocol default sqrt(2)).
            exclude: Variable -> values never to propose, e.g. {"buffer": ["PBS"]}.

        Returns:
            JSON list of conditions with predicted mean, SD and UCB score.
        """
        conds, ys = _dataset(lab)
        gp = A.fit_gp(conds, ys)
        return _j(A.suggest_ucb(gp, conds, n, exploitation, exploration, exclude))
    return execute


@tool
def mutual_information(lab: Lab) -> Tool:
    async def execute() -> str:
        """Mutual information between each variable and yield over all valid data.

        Returns:
            JSON map variable -> MI (nats), sorted high to low.
        """
        conds, ys = _dataset(lab)
        return _j(A.mutual_information(conds, ys))
    return execute


# ================================================================== notebook

@tool
def record_prior(lab: Lab) -> Tool:
    async def execute(claim: str, variable: str, low: float, high: float, unit: str,
                      confidence: float, sources: list[str]) -> str:
        """Record a prior as a range with stated confidence, before seeing data.

        Args:
            claim: Statement of the prior, e.g. "pH optimum lies between 9 and 10".
            variable: Variable it concerns.
            low: Lower bound of the range.
            high: Upper bound of the range.
            unit: Unit of the range.
            confidence: Probability (0-1) that the truth lies in the range.
            sources: Source identifiers with retrieval dates; empty means no source.

        Returns:
            JSON node with its id.
        """
        if not 0 <= confidence <= 1:
            raise ToolError("confidence must be in [0, 1]")
        node = _add_node("prior", claim=claim, variable=variable, range=[low, high], unit=unit,
                         confidence=confidence, sources=sources, plates_seen=len(lab.plates_run))
        return _j(node)
    return execute


@tool
def revise_prior(lab: Lab) -> Tool:
    async def execute(claim_id: str, new_low: float, new_high: float, new_confidence: float,
                      evidence: list[str], reason: str) -> str:
        """Revise a recorded prior when data contradict it.

        Args:
            claim_id: Id of the prior node being revised.
            new_low: New lower bound.
            new_high: New upper bound.
            new_confidence: New confidence (0-1).
            evidence: Evidence ids, e.g. batch ids or well ids.
            reason: Which claim failed and why.

        Returns:
            JSON revision node.
        """
        if not any(n["id"] == claim_id and n["kind"] == "prior" for n in _notebook()["nodes"]):
            raise ToolError(f"no prior with id {claim_id!r}")
        node = _add_node("prior_revision", parents=[claim_id], range=[new_low, new_high],
                         confidence=new_confidence, evidence=evidence, reason=reason,
                         plates_seen=len(lab.plates_run))
        return _j(node)
    return execute


@tool
def add_reasoning_node(lab: Lab) -> Tool:
    async def execute(kind: str, statement: str, parents: list[str] | None = None,
                      evidence: list[str] | None = None) -> str:
        """Add a node to the reasoning graph.

        Args:
            kind: observation | hypothesis | decision | deviation | escalation | conclusion.
            statement: The content of the node.
            parents: Ids of nodes this one follows from.
            evidence: Evidence ids (batch ids, wells, source ids).

        Returns:
            JSON node with its id.
        """
        kinds = {"observation", "hypothesis", "decision", "deviation", "escalation", "conclusion"}
        if kind not in kinds:
            raise ToolError(f"kind must be one of {sorted(kinds)}")
        ids = {n["id"] for n in _notebook()["nodes"]}
        bad = [p for p in parents or [] if p not in ids]
        if bad:
            raise ToolError(f"unknown parent ids {bad}")
        return _j(_add_node(kind, statement=statement, parents=parents or [], evidence=evidence or [],
                            plates_seen=len(lab.plates_run)))
    return execute


@tool
def get_notebook(lab: Lab) -> Tool:
    async def execute() -> str:
        """Return the whole reasoning graph and any submitted report.

        Returns:
            JSON notebook.
        """
        return _j(_notebook())
    return execute


@tool
def submit_report(lab: Lab) -> Tool:
    async def execute(optimum: dict[str, Any], yield_estimate: float, yield_ci_low: float,
                      yield_ci_high: float, important_variables: list[str],
                      unimportant_variables: list[str], revised_prior_ids: list[str],
                      confirmation_performed: list[str], stop_reason: str,
                      not_determined: list[str]) -> str:
        """Submit the final report. Validated before acceptance; can be called once.

        Args:
            optimum: Declared optimal condition (grid format).
            yield_estimate: Estimated yield at the optimum (relative to reference).
            yield_ci_low: Lower bound of the confidence interval.
            yield_ci_high: Upper bound of the confidence interval.
            important_variables: Variables that mattered (e.g. by mutual information).
            unimportant_variables: Variables that did not.
            revised_prior_ids: Ids of priors that were revised.
            confirmation_performed: Orthogonal confirmations run on the optimum.
            stop_reason: Which stopping criterion was met.
            not_determined: Things that could not be determined.

        Returns:
            JSON acceptance with any validation warnings.
        """
        nb = _notebook()
        if nb["report"] is not None:
            raise ToolError("report already submitted")
        _check_condition(optimum)
        if not yield_ci_low <= yield_estimate <= yield_ci_high:
            raise ToolError("yield_estimate must lie within the confidence interval")
        missing = []
        if not confirmation_performed:
            missing.append("no orthogonal confirmation of the optimum")
        if not stop_reason.strip():
            missing.append("stop_reason empty")
        # STRENDA-style completeness: conditions, replicates, controls, and the assay are all
        # defined by the protocol; what the agent must add is the uncertainty and evidence.
        complete = not missing
        report = {"optimum": optimum, "yield_estimate": yield_estimate,
                  "yield_ci": [yield_ci_low, yield_ci_high], "important_variables": important_variables,
                  "unimportant_variables": unimportant_variables, "revised_prior_ids": revised_prior_ids,
                  "confirmation_performed": confirmation_performed, "stop_reason": stop_reason,
                  "not_determined": not_determined, "complete": complete, "warnings": missing,
                  "last_plate_valid": bool(lab.plates_run and lab.plates_run[-1]["valid"]),
                  "budget": lab.budget()}
        nb["report"] = report
        store().set("notebook", nb)
        return _j({"accepted": True, "complete": complete, "warnings": missing})
    return execute


# ===================================================================== bundle

ALL_TOOLS = [
    get_lab_status, get_deck_layout, get_robot_state, get_event_log, capture_camera,
    get_plate_result, get_observations,
    move_tip, move_tip_to, set_gripper, aspirate, dispense, change_tip,
    design_batch, run_plate, enzyme_titration, spike_recovery, dual_wavelength,
    fit_model, suggest_ucb, mutual_information,
    record_prior, revise_prior, add_reasoning_node, get_notebook, submit_report,
]


def lab_tools(lab: Lab, include_robot_control: bool = True) -> list[Tool]:
    robot = {move_tip, move_tip_to, set_gripper, aspirate, dispense, change_tip}
    return [t(lab) for t in ALL_TOOLS if include_robot_control or t not in robot]
