"""The simulated lab the agent's tools talk to.

Flow for one batch: design_batch() validates conditions and mandatory controls
and lays the wells out over as many plate loads as needed (lab_sim's plate has
24 wells; wells are named "P1:A1", "P2:C4", ...) -> run_plate() has the robot
on the lab_sim scene pipette every transfer, plate by plate, incubate per
temperature group, start reactions with enzyme and read A405 kinetically, then
computes rates, yields, outliers and the validity gate over the whole batch.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import analysis as A
from . import config as C
from .chemistry import EnzymeParams, EnzymeWorld, WellContents
from .labsim_world import PLATE_HOVER, LabSimWorld

HAZARDS_PATH = Path(__file__).with_name("hazards.json")
MANDATORY_CONTROLS = ["reference", "blanks", "positive", "standard_curve", "carry_over"]
MAX_PLATES_PER_BATCH = 6
STOCKS = {  # concentration of each stock reservoir, in final-well units x volume factor
    "substrate": 10 * C.ADDITIVES["substrate"][0],
    "MgCl2": 10 * C.ADDITIVES["MgCl2"][0],
    "ZnCl2": 10 * C.ADDITIVES["ZnCl2"][0],
    "NaCl": 10 * C.ADDITIVES["NaCl"][0],
    "glycerol": 10 * C.ADDITIVES["glycerol"][0],
    "pNP_standard": 1.5,  # mM
}


@dataclass
class WellPlan:
    well: str
    role: str                      # condition | reference | blank | positive | standard | carry_over | confirm
    volumes: dict                  # reagent -> uL (nominal)
    temperature: float
    condition: dict | None = None
    replicate: int = 0
    tag: str = ""


@dataclass
class Batch:
    batch_id: str
    conditions: list[dict]
    plan: list[WellPlan]
    controls: list[str]
    off_grid_reasons: dict = field(default_factory=dict)
    executed: bool = False
    result: dict | None = None


class Lab:
    def __init__(self, seed: int = 0, budget_wells: int = 480, faults: dict | None = None):
        self.rng = np.random.default_rng(seed)
        self.faults = faults or {}
        self.world = LabSimWorld(seed=seed, plate_offset_mm=tuple(self.faults.get("plate_offset_mm", (0, 0))))
        self.enzyme = EnzymeWorld(EnzymeParams.sample(np.random.default_rng(seed + 10_000)), self.rng)
        if "enzyme_decay_per_hour" in self.faults:
            self.enzyme.enzyme_decay_per_hour = self.faults["enzyme_decay_per_hour"]
        self.hazards = json.loads(HAZARDS_PATH.read_text())
        self.budget_wells = budget_wells
        self.wells_used = 0
        self.reagent_used_ul: dict[str, float] = {}
        self.batches: dict[str, Batch] = {}
        self.plates_run: list[dict] = []
        self.observations: list[dict] = []   # valid condition-level yields for the model
        self.consecutive_control_failures = 0

    # ============================================================ helpers
    def volumes_for(self, cond: dict, enzyme_factor: float = 1.0, with_enzyme: bool = True,
                    pnp_mM: float = 0.0, substrate: bool = True) -> dict:
        v = {C.buffer_reagent(cond["buffer"], cond["pH"]): C.BUFFER_VOLUME_UL}
        for a in C.ADDITIVES:
            if a == "substrate" and not substrate:
                continue
            v[a] = C.LEVELS[cond[a]] * C.ADDITIVE_L4_VOLUME_UL
        if pnp_mM:
            v["pNP_standard"] = pnp_mM / STOCKS["pNP_standard"] * C.WELL_VOLUME_UL
        if with_enzyme:
            v["enzyme"] = C.ENZYME_VOLUME_UL * enzyme_factor
        v["water"] = max(0.0, C.WELL_VOLUME_UL - sum(v.values()))
        return v

    def interlock(self, cond: dict) -> list[str]:
        """Deterministic safety check from the hand-verified table. The agent
        never decides safety; unknown reagents are denied."""
        problems = []
        vols = self.volumes_for(cond)
        for reagent in vols:
            key = reagent.split(":")[1].split("@")[0] if reagent.startswith("buffer:") else reagent
            if key not in self.hazards["reagents"]:
                problems.append(f"no hazard record for {key!r}: denied")
            elif not self.hazards["reagents"][key].get("handling_allowed", False):
                problems.append(f"{key!r} is not cleared for handling")
            if reagent not in self.world.reservoir_index:
                problems.append(f"no reservoir on deck for {reagent!r}")
        present = {r.split(":")[1].split("@")[0] if r.startswith("buffer:") else r for r in vols}
        for a, b, why in self.hazards.get("incompatible_pairs", []):
            if a in present and b in present:
                problems.append(f"incompatible: {a} + {b} ({why})")
        if sum(vols.values()) > C.WELL_VOLUME_UL + 1e-6:
            problems.append("well overfill")
        return problems

    def _consume(self, volumes: dict) -> None:
        for r, v in volumes.items():
            self.reagent_used_ul[r] = self.reagent_used_ul.get(r, 0.0) + v

    def budget(self) -> dict:
        return {"wells_budget": self.budget_wells, "wells_used": self.wells_used,
                "wells_remaining": self.budget_wells - self.wells_used,
                "reagent_used_ul": {k: round(v, 1) for k, v in sorted(self.reagent_used_ul.items())},
                "plates_run": len(self.plates_run)}

    def deck_layout(self) -> dict:
        return {**self.world.layout(),
                "batches": f"a batch larger than one plate runs over several plate loads (max "
                           f"{MAX_PLATES_PER_BATCH}); wells are named P<plate>:<well>, e.g. P2:B3"}

    # ------------------------------------------------------- well naming
    @staticmethod
    def split_well(logical: str) -> tuple[int, str]:
        plate, _, well = logical.partition(":")
        return int(plate[1:]), well

    def is_edge(self, logical: str) -> bool:
        return self.split_well(logical)[1] in self.world.edge_wells

    def slots(self, plates: int, avoid_edges: bool) -> list[str]:
        per_plate = [w for w in self.world.wells if not (avoid_edges and w in self.world.edge_wells)]
        return [f"P{n}:{w}" for n in range(1, plates + 1) for w in per_plate]

    # ============================================================= design
    def design_batch(self, conditions: list[dict], controls: list[str], replicates: int = 3,
                     avoid_edges: bool = False, off_grid_reasons: dict | None = None) -> dict:
        errors = []
        off_grid_reasons = off_grid_reasons or {}
        for i, c in enumerate(conditions):
            probs = C.validate_condition(c, allow_off_grid=str(i) in off_grid_reasons)
            probs += self.interlock(c) if not probs else []
            if probs:
                errors.append({"condition_index": i, "condition": c, "problems": probs})
        missing = [m for m in MANDATORY_CONTROLS if m not in controls]
        if missing == ["carry_over"] and not self.plates_run:
            missing = []  # nothing to carry over on the first plate
        if missing:
            errors.append({"problem": "missing mandatory controls", "missing": missing,
                           "rule": "a plate missing any mandatory control is invalid"})
        if errors:
            return {"accepted": False, "errors": errors}

        plan: list[WellPlan] = []
        ref = dict(C.REFERENCE)
        for i, c in enumerate(conditions):
            for r in range(replicates):
                plan.append(WellPlan("", "condition", self.volumes_for(c), c["temperature"], c, r, f"cond{i}"))
        for r in range(3):
            plan.append(WellPlan("", "reference", self.volumes_for(ref), ref["temperature"], ref, r, "reference"))
            plan.append(WellPlan("", "positive", self.volumes_for(ref), ref["temperature"], ref, r, "positive"))
        buf_ph = sorted({(c["buffer"], c["pH"]) for c in conditions} | {(ref["buffer"], ref["pH"])})
        for b, p in buf_ph:
            base = next((c for c in conditions if (c["buffer"], c["pH"]) == (b, p)), ref)
            for r in range(3):
                plan.append(WellPlan("", "blank", self.volumes_for(base, with_enzyme=False),
                                     base["temperature"], base, r, f"blank:{b}@{p:g}"))
        for k, mm in enumerate(C.STANDARD_CURVE_MM):
            plan.append(WellPlan("", "standard", self.volumes_for(ref, with_enzyme=False, pnp_mM=mm, substrate=False),
                                 25.0, None, k, f"std:{mm}"))
        if self.plates_run and self.plates_run[-1].get("best_condition"):
            best = self.plates_run[-1]["best_condition"]
            for r in range(3):
                plan.append(WellPlan("", "carry_over", self.volumes_for(best), best["temperature"], best, r, "carry_over"))

        per_plate = len(self.slots(1, avoid_edges))
        plates = math.ceil(len(plan) / per_plate)
        if plates > MAX_PLATES_PER_BATCH:
            return {"accepted": False, "errors": [{
                "problem": f"batch needs {len(plan)} wells = {plates} plates of {per_plate}"
                           + (" (edges avoided)" if avoid_edges else "")
                           + f"; at most {MAX_PLATES_PER_BATCH} plates per batch"}]}
        if self.wells_used + len(plan) > self.budget_wells:
            return {"accepted": False, "errors": [{"problem": "over well budget", **self.budget()}]}
        slots = self.slots(plates, avoid_edges)
        chosen = sorted(self.rng.choice(len(slots), size=len(plan), replace=False))
        self.rng.shuffle(plan)  # randomised layout, so position effects do not align with conditions
        for wp, k in zip(plan, chosen):
            wp.well = slots[k]

        batch_id = f"B{len(self.batches) + 1:02d}"
        self.batches[batch_id] = Batch(batch_id, conditions, plan, controls, off_grid_reasons)
        roles = {}
        for wp in plan:
            roles[wp.role] = roles.get(wp.role, 0) + 1
        return {"accepted": True, "batch_id": batch_id, "wells": len(plan), "plates": plates,
                "well_roles": roles, "temperature_groups": sorted({wp.temperature for wp in plan}),
                "plate_map": {wp.well: f"{wp.role}:{wp.tag}#{wp.replicate}" for wp in plan}}

    # ============================================================ execute
    def _execute(self, plan: list[WellPlan], wavelengths=(405,)) -> tuple[dict, dict]:
        """Robot pipettes every transfer, plate load by plate load; returns (well contents, reads)."""
        w = self.world
        delivered = {wp.well: {} for wp in plan}
        enzyme_added_at: dict[str, float] = {}
        start_time: dict[str, float] = {}
        read_clock: dict[str, float] = {}
        nominal_total: dict[str, float] = {}
        exec_log = {"plates": 0, "transfers": 0, "spills": 0, "wrong_well": 0, "aspirate_failures": 0,
                    "ik_failures": 0, "collisions": 0}
        ev_start = len(w.events)

        def pipette_reagent(reagent: str, targets: list[WellPlan], plate: int):
            w.change_tip()
            queue = [(wp.well, wp.volumes[reagent]) for wp in targets if wp.volumes.get(reagent, 0) > 0]
            while queue:
                load, chunk = 0.0, []
                while queue and load + queue[0][1] <= C.TIP_CAPACITY_UL:
                    chunk.append(queue.pop(0))
                    load += chunk[-1][1]
                r = w.aspirate(reagent, load)
                if not r["ok"]:
                    exec_log["aspirate_failures"] += 1
                    continue
                for logical, vol in chunk:
                    d = w.dispense(self.split_well(logical)[1], vol, hover=PLATE_HOVER)
                    exec_log["transfers"] += 1
                    if not d["ok"]:
                        continue
                    nominal_total[reagent] = nominal_total.get(reagent, 0.0) + vol
                    if d["spilled"]:
                        exec_log["spills"] += 1
                        continue
                    landed = f"P{plate}:{d['actual_well']}"
                    exec_log["wrong_well"] += int(landed != logical)
                    if landed in delivered:  # liquid in an unused well is lost to the experiment
                        delivered[landed][reagent] = delivered[landed].get(reagent, 0.0) + d["delivered_ul"]
                    if reagent == "enzyme":
                        enzyme_added_at[landed] = w.clock_min

        for plate in sorted({self.split_well(wp.well)[0] for wp in plan}):
            if any(v > 0 for v in w.well_volume.values()):
                w.swap_plate()
            exec_log["plates"] += 1
            on_plate = [wp for wp in plan if self.split_well(wp.well)[0] == plate]
            reagents = sorted({r for wp in on_plate for r in wp.volumes} - {"enzyme"})
            for reagent in reagents:
                pipette_reagent(reagent, on_plate, plate)
            # Incubate per temperature group, then start reactions with enzyme. The reader
            # reads the group when the last well is started, so early wells have already
            # been reacting for a while.
            for temp in sorted({wp.temperature for wp in on_plate}):
                w.clock_min += 5.0  # thermal equilibration
                members = [wp for wp in on_plate if wp.temperature == temp]
                group_start = w.clock_min
                pipette_reagent("enzyme", members, plate)
                for wp in members:
                    start_time[wp.well] = w.clock_min - enzyme_added_at.get(wp.well, group_start)
                    read_clock[wp.well] = w.clock_min
        w.home()

        contamination = self.faults.get("contaminated_reagent", {})
        contents, reads = {}, {}
        for wp in plan:
            got = delivered[wp.well]
            total = sum(got.values()) or 1e-9
            buf = next((r for r in got if r.startswith("buffer:")), None)
            wc = WellContents(volume_ul=total, temperature_c=wp.temperature)
            if buf:
                b, p = buf.split(":")[1].split("@pH")
                wc.buffer, wc.buffer_ph = b, float(p)
                wc.buffer_fraction = got[buf] / C.BUFFER_VOLUME_UL * C.WELL_VOLUME_UL / total
            wc.substrate_mM = STOCKS["substrate"] * got.get("substrate", 0) / total
            wc.mg_mM = STOCKS["MgCl2"] * got.get("MgCl2", 0) / total
            wc.zn_mM = STOCKS["ZnCl2"] * got.get("ZnCl2", 0) / total
            wc.nacl_mM = STOCKS["NaCl"] * got.get("NaCl", 0) / total
            wc.glycerol_pct = STOCKS["glycerol"] * got.get("glycerol", 0) / total
            wc.pnp_mM = STOCKS["pNP_standard"] * got.get("pNP_standard", 0) / total
            wc.enzyme_ug = C.ENZYME_UG_PER_WELL * got.get("enzyme", 0) / C.ENZYME_VOLUME_UL
            for reagent, pi_mM in contamination.items():
                wc.phosphate_mM += pi_mM * got.get(reagent, 0) / total
            contents[wp.well] = wc
            age_h = 0.0 if wp.role == "positive" else read_clock.get(wp.well, w.clock_min) / 60.0
            times = [t + start_time.get(wp.well, 0.0) for t in C.READ_TIMES_MIN]
            reads[wp.well] = {wl: self.enzyme.absorbance(wc, times, self.is_edge(wp.well), wl, age_h).round(4).tolist()
                              for wl in wavelengths}
        if self.faults.get("reader_drift_batches") and len(self.plates_run) + 1 in self.faults["reader_drift_batches"]:
            for well in reads:  # lamp drift: gain grows with absorbance, bending the standard curve
                for wl in reads[well]:
                    reads[well][wl] = [round(a * (1 + 0.25 * a), 4) for a in reads[well][wl]]
        last_plate = max(self.split_well(wp.well)[0] for wp in plan)
        for wp in plan:  # colour the plate still on the bench by its final read
            plate, well = self.split_well(wp.well)
            if plate == last_plate:
                w.show_absorbance(well, reads[wp.well][405][-1])

        for ev in w.events[ev_start:]:
            if ev.kind in ("ik_failure", "collision"):
                exec_log[ev.kind + "s"] += 1
        self._consume(nominal_total)
        self.wells_used += len(plan)
        exec_log["events"] = [e.as_dict() for e in w.events[ev_start:] if e.kind != "tip_change"]
        exec_log["lab_clock_min"] = round(w.clock_min, 2)
        return contents, {"reads": reads, "execution": exec_log}

    # =========================================================== analyse
    def run_plate(self, batch_id: str) -> dict:
        batch = self.batches.get(batch_id)
        if batch is None:
            return {"error": f"unknown batch {batch_id!r}"}
        if batch.executed:
            return {"error": f"batch {batch_id} already run", "result": batch.result}
        contents, out = self._execute(batch.plan)
        reads, execution = out["reads"], out["execution"]
        batch.executed = True

        rate = {}
        for wp in batch.plan:
            fit = A.rate_from_reads(C.READ_TIMES_MIN, reads[wp.well][405])
            # Blanks are normalised by the nominal enzyme mass too, so they subtract directly.
            fit["rate_uM_per_min_per_ug"] = (None if fit["slope_A_per_min"] is None
                                             else A.slope_to_rate(fit["slope_A_per_min"], C.ENZYME_UG_PER_WELL))
            rate[wp.well] = fit

        def rates(role, tag=None):
            return [rate[wp.well]["rate_uM_per_min_per_ug"] for wp in batch.plan
                    if wp.role == role and (tag is None or wp.tag == tag)
                    and rate[wp.well]["rate_uM_per_min_per_ug"] is not None]

        blank = {}
        for wp in batch.plan:
            if wp.role == "blank":
                key = wp.tag.split(":")[1]
                blank.setdefault(key, []).append(rate[wp.well]["rate_uM_per_min_per_ug"] or 0.0)
        blank = {k: float(np.mean(v)) for k, v in blank.items()}
        ref_key = f"{C.REFERENCE['buffer']}@{C.REFERENCE['pH']:g}"
        s_ref = float(np.mean(rates("reference"))) if rates("reference") else float("nan")
        denom = s_ref - blank.get(ref_key, 0.0)

        def yield_of(wp):
            r = rate[wp.well]["rate_uM_per_min_per_ug"]
            if r is None or not np.isfinite(denom) or abs(denom) < 1e-9:
                return None
            key = f"{wp.condition['buffer']}@{wp.condition['pH']:g}"
            return (r - blank.get(key, 0.0)) / denom

        per_condition = []
        flagged = 0
        for i, c in enumerate(batch.conditions):
            wps = [wp for wp in batch.plan if wp.tag == f"cond{i}"]
            ys = [yield_of(wp) for wp in wps]
            o = A.apply_outlier_rule(ys)
            flagged += int(o["flagged"])
            dropped_well = None
            if o["flagged"]:
                dropped_well = next(wp.well for wp, y in zip(wps, ys) if y is not None and abs(y - o["dropped"]) < 1e-12)
            per_condition.append({
                "index": i, "condition": c, "yield_mean": None if o["mean"] is None else round(o["mean"], 4),
                "yield_sd": None if o["sd"] is None else round(o["sd"], 4),
                "replicates": {wp.well: (None if y is None else round(y, 4)) for wp, y in zip(wps, ys)},
                "cv_before_rule": o["cv"], "outlier_dropped_well": dropped_well,
                "outlier_is_edge_well": self.is_edge(dropped_well) if dropped_well else None,
                "rate_indeterminate_wells": [wp.well for wp in wps if rate[wp.well]["rate_indeterminate"]],
            })

        # ---------------- validity gate
        pos = rates("positive")
        neg = rates("blank")
        zp = A.z_prime(pos, neg)
        control_wells = [wp for wp in batch.plan if wp.role != "condition"]
        max_ctrl_a = max(max(reads[wp.well][405]) for wp in control_wells)
        std = sorted((float(wp.tag.split(":")[1]), reads[wp.well][405][0]) for wp in batch.plan if wp.role == "standard")
        std_r2 = A.linear_r2([s[0] for s in std], [s[1] for s in std])
        pos_yield = (float(np.mean(pos)) - blank.get(ref_key, 0.0)) / denom if pos and denom else None
        agreement = {"first_plate": not self.plates_run}
        agree_ok = True
        if pos_yield is not None:
            agreement["positive_control_yield"] = round(pos_yield, 3)
            agree_ok &= 0.75 <= pos_yield <= 1.33
        if any(wp.role == "carry_over" for wp in batch.plan):
            co = [yield_of(wp) for wp in batch.plan if wp.role == "carry_over"]
            co = [y for y in co if y is not None]
            prev = self.plates_run[-1]["best_yield"]
            if co and prev:
                agreement["carry_over_yield"] = round(float(np.mean(co)), 3)
                agreement["carry_over_previous_yield"] = round(prev, 3)
                agreement["carry_over_ratio"] = round(float(np.mean(co)) / prev, 3)
                agree_ok &= 0.7 <= float(np.mean(co)) / prev <= 1.43
        gate = {
            "1_controls_agree_with_previous_plates": {"pass": bool(agree_ok), **agreement,
                                                      "rule": "positive-control yield in [0.75, 1.33]; carry-over within 30% of previous plate"},
            "2_z_prime_above_0.5": {"pass": zp is not None and zp > 0.5, "z_prime": None if zp is None else round(zp, 3)},
            "3_controls_within_detector_linear_range": {"pass": max_ctrl_a < C.DETECTOR_LINEAR_MAX_A,
                                                        "max_control_absorbance": round(max_ctrl_a, 3)},
            "4_standard_curve_linear_r2_above_0.98": {"pass": std_r2 > 0.98, "r2": round(std_r2, 4),
                                                      "points_mM_vs_A": std},
        }
        plate_valid = all(v["pass"] for v in gate.values())
        suspect = flagged > 0.10 * max(1, len(batch.conditions))
        self.consecutive_control_failures = 0 if plate_valid else self.consecutive_control_failures + 1

        valid_rows = [p for p in per_condition if p["yield_mean"] is not None]
        best = max(valid_rows, key=lambda p: p["yield_mean"]) if valid_rows else None
        plate_record = {"batch_id": batch_id, "valid": plate_valid,
                        "best_condition": best["condition"] if best and plate_valid else
                        (self.plates_run[-1]["best_condition"] if self.plates_run else None),
                        "best_yield": best["yield_mean"] if best and plate_valid else
                        (self.plates_run[-1]["best_yield"] if self.plates_run else None)}
        self.plates_run.append(plate_record)
        if plate_valid:
            for p in valid_rows:
                self.observations.append({"batch_id": batch_id, "condition": p["condition"], "yield": p["yield_mean"]})

        stop_reasons = []
        if self.consecutive_control_failures >= 2:
            stop_reasons.append("control failure on two consecutive plates")
        if execution["spills"]:
            stop_reasons.append(f"{execution['spills']} spill(s) detected during dispensing")
        if execution["ik_failures"] or execution["collisions"]:
            stop_reasons.append("robot motion fault (IK failure or collision)")

        batch.result = {
            "batch_id": batch_id,
            "plate_valid": plate_valid,
            "plate_suspect_outlier_rate": suspect,
            "validity_gate": gate,
            "data_entered_model": plate_valid,
            "reference_rate_uM_per_min_per_ug": round(s_ref, 3),
            "blank_rates_uM_per_min_per_ug": {k: round(v, 4) for k, v in blank.items()},
            "conditions": per_condition,
            "mandatory_stop": stop_reasons,
            "execution": execution,
            "budget": self.budget(),
            "raw": {"read_times_min": C.READ_TIMES_MIN,
                    "wells": {wp.well: {"role": wp.role, "tag": wp.tag, "A405": reads[wp.well][405],
                                        "fit": rate[wp.well]} for wp in batch.plan}},
        }
        return batch.result

    # ===================================================== confirmations
    def _small_plan(self, specs: list[tuple[str, dict, float, dict | None, str]]) -> list[WellPlan]:
        # Confirmations use inner wells only, over as many plate loads as they need.
        per_plate = len(self.slots(1, avoid_edges=True))
        free = self.slots(math.ceil(len(specs) / per_plate), avoid_edges=True)
        plan = []
        for (role, vols, temp, cond, tag), well in zip(specs, free):
            plan.append(WellPlan(well, role, vols, temp, cond, 0, tag))
        return plan

    def enzyme_titration(self, cond: dict, factors=(0.5, 1.0, 2.0), replicates: int = 3) -> dict:
        specs = [("confirm", self.volumes_for(cond, enzyme_factor=f), cond["temperature"], cond, f"x{f}")
                 for f in factors for _ in range(replicates)]
        specs += [("blank", self.volumes_for(cond, with_enzyme=False), cond["temperature"], cond, "blank")] * 2
        return self._confirm(specs, cond, kind="enzyme_titration")

    def spike_recovery(self, cond: dict, spike_mM: float = 0.05, replicates: int = 3) -> dict:
        specs = [("confirm", self.volumes_for(cond, with_enzyme=False, pnp_mM=spike_mM, substrate=False),
                  25.0, cond, "spike")] * replicates
        specs += [("standard", self.volumes_for(C.REFERENCE, with_enzyme=False, pnp_mM=mm, substrate=False),
                   25.0, None, f"std:{mm}") for mm in C.STANDARD_CURVE_MM]
        return self._confirm(specs, cond, kind="spike_recovery", spike_mM=spike_mM)

    def dual_wavelength(self, cond: dict, replicates: int = 3) -> dict:
        specs = [("confirm", self.volumes_for(cond), cond["temperature"], cond, "dual")] * replicates
        return self._confirm(specs, cond, kind="dual_wavelength", wavelengths=(405, 490))

    def _confirm(self, specs, cond, kind, wavelengths=(405,), spike_mM=None) -> dict:
        probs = C.validate_condition(cond) or self.interlock(cond)
        if probs:
            return {"error": probs}
        plan = self._small_plan(specs)
        if self.wells_used + len(plan) > self.budget_wells:
            return {"error": "over well budget", **self.budget()}
        _, out = self._execute(plan, wavelengths=wavelengths)
        reads = out["reads"]
        res = {"kind": kind, "condition": cond, "wells_used": len(plan), "execution": out["execution"]}
        if kind == "enzyme_titration":
            by = {}
            for wp in plan:
                f = A.rate_from_reads(C.READ_TIMES_MIN, reads[wp.well][405])
                by.setdefault(wp.tag, []).append(f["slope_A_per_min"])
            blank = np.mean([s or 0 for s in by.pop("blank")])
            slopes = {k: float(np.mean([s for s in v if s is not None]) - blank) if any(s is not None for s in v) else None
                      for k, v in by.items()}
            res["net_slope_A_per_min"] = {k: None if v is None else round(v, 5) for k, v in slopes.items()}
            if all(v is not None for v in slopes.values()):
                xs = [float(k[1:]) for k in slopes]
                res["proportionality_r2"] = round(A.linear_r2(xs, list(slopes.values())), 4)
                res["ratio_2x_over_1x"] = round(slopes["x2.0"] / slopes["x1.0"], 3) if slopes.get("x1.0") else None
            res["interpretation_hint"] = "rate should scale ~proportionally with enzyme amount (ratio ~2)"
        elif kind == "spike_recovery":
            std = sorted((float(wp.tag.split(':')[1]), reads[wp.well][405][0]) for wp in plan if wp.role == "standard")
            slope, intercept = np.polyfit([s[0] for s in std], [s[1] for s in std], 1)
            meas = [(reads[wp.well][405][0] - intercept) / slope for wp in plan if wp.tag == "spike"]
            res["recovered_mM"] = [round(m, 5) for m in meas]
            res["recovery_pct"] = round(100 * float(np.mean(meas)) / spike_mM, 1)
        else:
            out405 = [A.rate_from_reads(C.READ_TIMES_MIN, reads[wp.well][405])["slope_A_per_min"] for wp in plan]
            corr = [A.rate_from_reads(C.READ_TIMES_MIN, np.subtract(reads[wp.well][405], reads[wp.well][490]))["slope_A_per_min"]
                    for wp in plan]
            res["slope_405"] = [None if s is None else round(s, 5) for s in out405]
            res["slope_405_minus_490"] = [None if s is None else round(s, 5) for s in corr]
            res["A490_t0"] = [reads[wp.well][490][0] for wp in plan]
        res["budget"] = self.budget()
        return res

    # ===================================================== hidden truth
    def _ideal_contents(self, cond: dict, enzyme_factor: float = 1.0) -> WellContents:
        return WellContents(
            buffer=cond["buffer"], buffer_ph=cond["pH"], buffer_fraction=1.0,
            substrate_mM=C.LEVELS[cond["substrate"]] * C.ADDITIVES["substrate"][0],
            mg_mM=C.LEVELS[cond["MgCl2"]] * C.ADDITIVES["MgCl2"][0],
            zn_mM=C.LEVELS[cond["ZnCl2"]] * C.ADDITIVES["ZnCl2"][0],
            nacl_mM=C.LEVELS[cond["NaCl"]] * C.ADDITIVES["NaCl"][0],
            glycerol_pct=C.LEVELS[cond["glycerol"]] * C.ADDITIVES["glycerol"][0],
            enzyme_ug=C.ENZYME_UG_PER_WELL * enzyme_factor, volume_ul=C.WELL_VOLUME_UL,
            temperature_c=cond["temperature"])

    def true_activity(self, cond: dict) -> float:
        """Noise-free initial rate. For scoring only; never expose to the agent."""
        wc = self._ideal_contents(cond)
        p = self.enzyme.p
        km_app = p.km_mM * (1 + self.enzyme.phosphate_mM(wc) / p.ki_pi_mM)
        s = wc.substrate_mM
        return float(p.vmax * self.enzyme.rate_constant_factor(wc, 0.0) * s / (km_app + s))

    def true_optimum(self) -> tuple[dict, float]:
        if not hasattr(self, "_opt"):
            best = max(C.iter_grid(), key=self.true_activity)
            self._opt = (best, self.true_activity(best))
        return self._opt
