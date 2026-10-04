"""Smoke tests for the lab tools (harness/tools/lab_tools.py + lab_backend.py): dispense,
transfer_sample, mix, get_lab_state, and the tip-box-empty / tip-not-seated failure paths.

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


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def backend():
    return LabBackend()


@pytest.fixture
def tools(backend):
    return lab_tools(backend)  # (dispense, transfer_sample, mix, get_lab_state)


def test_dispense_mounts_and_ejects_a_tip(backend, tools):
    dispense, _, _, get_lab_state = tools
    before = json.loads(_run(get_lab_state()))["tips_remaining"]

    res = json.loads(_run(dispense(reagent="water", destination="B3", volume_ul=50)))
    assert res["ok"]

    after = json.loads(_run(get_lab_state()))["tips_remaining"]
    assert after == before - 1

    entry = backend.ledger[-1]
    assert entry.tool == "dispense"
    assert entry.observed["tip_slot"] == 0
    assert entry.observed["touched"] == ["reagent_water", "well_B3"]


def test_transfer_sample_uses_a_fresh_tip(backend, tools):
    dispense, transfer_sample, _, _ = tools
    _run(dispense(reagent="water", destination="B3", volume_ul=50))

    res = json.loads(_run(transfer_sample(source="B3", destination="B4", volume_ul=20)))
    assert res["ok"]

    entry = backend.ledger[-1]
    assert entry.tool == "transfer_sample"
    assert entry.observed["tip_slot"] == 1          # a new tip, not dispense's slot 0
    assert entry.observed["touched"] == ["well_B3", "well_B4"]


def test_mix_does_not_touch_the_tip_box(backend, tools):
    _, _, mix, get_lab_state = tools
    before = json.loads(_run(get_lab_state()))["tips_remaining"]

    res = json.loads(_run(mix(container="B3", cycles=2)))
    assert res["ok"]

    after = json.loads(_run(get_lab_state()))["tips_remaining"]
    assert after == before
    assert "tip_slot" not in backend.ledger[-1].observed


def test_tip_box_empty_fails_dispense_with_a_clear_reason(backend, tools):
    dispense, _, _, get_lab_state = tools
    backend.skills.used_slots = set(range(len(backend.skills.tip_slots)))  # drain the box

    res = json.loads(_run(dispense(reagent="water", destination="B3", volume_ul=10)))
    assert not res["ok"]

    entry = backend.ledger[-1]
    assert entry.observed["reason"] == "tip: tip box empty"
    assert entry.observed["tip_slot"] is None
    assert json.loads(_run(get_lab_state()))["box_empty"]


def test_tip_not_seated_fails_pipette_with_a_clear_reason(backend):
    orig = backend.skills.pick_up_tip
    backend.skills.pick_up_tip = lambda seat=True, press=0.004: orig(seat=False, press=press)

    out = backend.pipette("reagent_water", "well_B3")
    assert not out["ok"]
    assert out["reason"] == "tip: tip not seated (fault) at slot 0"
    assert out["tip_slot"] is None
