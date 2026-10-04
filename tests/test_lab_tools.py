"""Smoke tests for the lab tools (harness/tools/lab_tools.py + lab_backend.py): dispense,
transfer_sample, mix, change_tip, get_lab_state, tip reuse, and the tip-box-empty / tip-not-seated failure paths.

Builds the real lab_sim scene (mujoco/mink) rather than mocking it, so these exercise the
actual motion stack. `lab_sim` isn't a package root on sys.path by default (its own modules
import each other as top-level `scenes`/`robot`), so it's added here the same way the `uv run
python -m ...` entry points in CLAUDE.md are run from inside `lab_sim/`.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lab_sim"))

import pytest

from harness.tools.lab_backend import LabBackend
from harness.tools.lab_tools import lab_tools
from inspect_ai.tool import ToolDef


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def backend():
    return LabBackend()


@pytest.fixture
def tools(backend):
    """Tools by name, so adding a tool to lab_tools() can't shift what a test is calling."""
    return {ToolDef(t).name: t for t in lab_tools(backend)}


def _state(tools):
    return json.loads(_run(tools["get_lab_state"]()))


def test_dispense_mounts_a_tip_and_keeps_it(backend, tools):
    before = _state(tools)["tips_remaining"]

    res = json.loads(_run(tools["dispense"](reagent="water", destination="B3", volume_ul=50)))
    assert res["ok"]

    state = _state(tools)
    assert state["tips_remaining"] == before - 1
    assert state["has_tip"]                         # kept on the nozzle, not ejected

    entry = backend.ledger[-1]
    assert entry.tool == "dispense"
    assert entry.observed["tip_slot"] == 0
    assert entry.observed["touched"] == ["reagent_water", "well_B3"]


def test_tip_persists_across_transfers_until_change_tip(backend, tools):
    """One tip is reused (and carries over) until change_tip; the ledger records which tip
    touched which containers, before and after the change."""
    _run(tools["dispense"](reagent="water", destination="B3", volume_ul=50))
    tips_after_dispense = _state(tools)["tips_remaining"]

    res = json.loads(_run(tools["transfer_sample"](source="B3", destination="B4", volume_ul=20)))
    assert res["ok"]
    assert _state(tools)["tips_remaining"] == tips_after_dispense     # no new tip taken
    entry = backend.ledger[-1]
    assert entry.tool == "transfer_sample"
    assert entry.observed["tip_slot"] == 0                            # same tip as dispense
    assert entry.observed["touched"] == ["reagent_water", "well_B3", "well_B4"]   # carry-over

    res = json.loads(_run(tools["change_tip"]()))
    assert res["ok"]
    entry = backend.ledger[-1]
    assert entry.observed["ejected_slot"] == 0
    assert entry.observed["ejected_contacts"] == ["reagent_water", "well_B3", "well_B4"]
    assert entry.observed["new_slot"] == 1

    _run(tools["transfer_sample"](source="B4", destination="B5", volume_ul=20))
    entry = backend.ledger[-1]
    assert entry.observed["tip_slot"] == 1
    assert entry.observed["touched"] == ["well_B4", "well_B5"]       # fresh tip, clean history


def test_mix_does_not_touch_the_tip_box(backend, tools):
    before = _state(tools)["tips_remaining"]

    res = json.loads(_run(tools["mix"](container="B3", cycles=2)))
    assert res["ok"]

    assert _state(tools)["tips_remaining"] == before
    assert "tip_slot" not in backend.ledger[-1].observed


def test_tip_box_empty_fails_dispense_with_a_clear_reason(backend, tools):
    backend.skills.used_slots = set(range(len(backend.skills.tip_slots)))  # drain the box

    res = json.loads(_run(tools["dispense"](reagent="water", destination="B3", volume_ul=10)))
    assert not res["ok"]

    entry = backend.ledger[-1]
    assert entry.observed["reason"] == "tip: tip box empty"
    assert entry.observed["tip_slot"] is None
    assert _state(tools)["box_empty"]


def test_tip_not_seated_fails_pipette_with_a_clear_reason(backend):
    orig = backend.skills.pick_up_tip
    backend.skills.pick_up_tip = lambda seat=True, press=0.004: orig(seat=False, press=press)

    out = backend.pipette("reagent_water", "well_B3")
    assert not out["ok"]
    assert out["reason"] == "tip: tip not seated (fault) at slot 0"
    assert out["tip_slot"] is None
