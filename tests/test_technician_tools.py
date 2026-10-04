"""Technician tools (harness/tools/technician_tools.py): check_collisions reports what LabBackend
logged as a collision; redo_plan unchecks a plan so the attempt is redone.

Collisions reach `backend.incidents` through `backend.log("collision", ...)` -- the channel Lok's
held-object collision checks feed (feat/held-object-collisions). Here one is logged directly.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lab_sim"))

from inspect_ai.util import store_as
from inspect_ai.util._store import Store, init_subtask_store

from harness.tools.lab_backend import LabBackend
from harness.tools.lab_tools import lab_tools
from harness.tools.technician_tools import technician_tools
from harness.types.state import LabState, Plan


def _run(coro):
    return json.loads(asyncio.run(coro))


def test_technician_tools_are_in_lab_tools():
    names = {t.__qualname__.split(".<locals>.")[-2] for t in lab_tools(LabBackend())}
    assert {"check_collisions", "redo_plan"} <= names


def test_check_collisions_reports_logged_collisions():
    b = LabBackend()
    check_collisions, _ = technician_tools(b)
    assert _run(check_collisions()) == {"lab_time_min": 0.0, "collision": False, "collisions": []}

    b.clock_min = 3.0
    b.log("collision", carried="pipette_shaft", other="collide_tube_nacl", dist=-0.002)
    out = _run(check_collisions())
    assert out["collision"]
    assert out["collisions"] == [{"t_min": 3.0, "carried": "pipette_shaft", "hit": "collide_tube_nacl"}]
    assert _run(check_collisions(since_min=5.0))["collision"] is False   # before the current plan


def test_redo_plan_unchecks_every_step_and_logs_it():
    init_subtask_store(Store())
    s = store_as(LabState)
    s.task_plans["t1"] = Plan(steps=["dispense water into B3", "mix B3"], completed=[True, True])
    b = LabBackend()
    _, redo_plan = technician_tools(b)

    out = _run(redo_plan(task_name="t1", reason="collision with the nacl tube"))
    assert out["steps_to_redo"] == ["dispense water into B3", "mix B3"]
    assert s.task_plans["t1"].completed == [False, False]
    assert b.events[-1]["kind"] == "redo" and b.events[-1]["task"] == "t1"
    assert s.experiment_graph.experiments == []                           # no budget spent
