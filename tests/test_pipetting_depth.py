"""Pipetting depth: pipette() must put the tip INTO each vessel, not hover over its site.

The tip end goes a fixed depth into the vessel (no liquid tracking): TUBE_TIP_DEPTH below a
tube's opening, WELL_FLOOR_GAP above a well's floor (robot/skills.py), using vessel sizes from
scene_contract(). Builds the real lab_sim scene, same convention as test_lab_tools.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lab_sim"))

import pytest

from harness.tools.lab_backend import LabBackend
from lab_sim.robot.skills import TUBE_TIP_DEPTH, WELL_FLOOR_GAP


@pytest.fixture
def backend():
    return LabBackend()


def _floor_z(backend, site):
    m, d = backend.model, backend.data
    g = m.geom(backend.contract.liquid_geom(site)).id
    return d.geom_xpos[g][2] - m.geom_size[g][1]


def test_pipette_enters_tube_and_well_to_fixed_depth(backend):
    src, dst = "reagent_water", "well_B3"
    # vessel geometry captured before the run (nothing moves the stock tube or the plate)
    tube_open = _floor_z(backend, src) + backend.contract.vessels["tube"].height_m
    well_floor = _floor_z(backend, dst)

    r = backend.pipette(src, dst)

    assert r["ok"], r.get("reason")
    tips = {site: res.tip_pos for kind, site, res in r["moves"] if kind == "descend"}
    assert tips[src][2] == pytest.approx(tube_open - TUBE_TIP_DEPTH, abs=2e-3)
    assert well_floor < tips[dst][2] <= well_floor + WELL_FLOOR_GAP + 2e-3
    assert backend.incidents == []
