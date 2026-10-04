"""Rack hole geometry vs the real AutoBio mesh, and place()'s post-snap penetration check.

The 15 mL hole grid in scenes/build_lab.py (RACK_15ML_HOLES_X/_ROWS_Y/_HOLE_R, RACK_FLOOR_Z) was
measured by ray-casting the rack mesh; these tests re-measure it so the constants -- and every
site derived from them (stocked slots, spare holes) -- can't drift from the mesh. Builds the real
lab_sim scene, same convention as test_lab_tools.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lab_sim"))

import mujoco
import numpy as np
import pytest

from harness.tools.lab_backend import LabBackend
from lab_sim.scenes.build_lab import (RACK_A, RACK_B, RACK_15ML_HOLE_R, RACK_FLOOR_Z,
                                      load_model, rack_holes)

DOWN = np.array([0, 0, -1.0])


@pytest.fixture(scope="module")
def scene():
    m = load_model()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    return m, d


def _hits(m, d, rack, part, x, y):
    return mujoco.mj_rayMesh(m, d, m.geom(f"{rack}_{part}").id, np.array([x, y, 0.2]), DOWN) >= 0


@pytest.mark.parametrize("rack,centre", [("rackA", RACK_A), ("rackB", RACK_B)])
def test_measured_holes_match_the_mesh(scene, rack, centre):
    """Every listed hole is open (centre + a ring just inside its radius miss the upper plane)
    and really is r~8.5 mm (a ring just outside hits it)."""
    m, d = scene
    angles = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    for hx, hy in rack_holes(*centre):
        inside = [(hx + 0.98 * RACK_15ML_HOLE_R * np.cos(a), hy + 0.98 * RACK_15ML_HOLE_R * np.sin(a))
                  for a in angles] + [(hx, hy)]
        assert not any(_hits(m, d, rack, "upper_plane", x + 1e-5, y + 1e-5) for x, y in inside), (hx, hy)
        outside = [(hx + 1.08 * RACK_15ML_HOLE_R * np.cos(a), hy + 1.08 * RACK_15ML_HOLE_R * np.sin(a))
                   for a in angles]
        assert all(_hits(m, d, rack, "upper_plane", x + 1e-5, y + 1e-5) for x, y in outside), (hx, hy)


def test_rack_floor_height_matches_the_mesh(scene):
    m, d = scene
    g = m.geom("rackA_lower_plane").id
    z = 0.2 - mujoco.mj_rayMesh(m, d, g, np.array([RACK_A[0] + 0.018, RACK_A[1], 0.2]), DOWN)
    assert z == pytest.approx(RACK_FLOOR_Z, abs=2e-4)


@pytest.mark.parametrize("site,centre", [("spare_hole", RACK_A), ("spare_hole_b", RACK_B)])
def test_spare_holes_sit_over_measured_holes(scene, site, centre):
    m, d = scene
    p = d.site_xpos[m.site(site).id]
    assert min(np.hypot(p[0] - hx, p[1] - hy) for hx, hy in rack_holes(*centre)) < 1e-6


def test_stocked_tubes_start_seated_without_penetrating(scene):
    backend = LabBackend()
    sk = backend.skills
    for body in sk.welds:
        if body.startswith("tubebody_"):
            assert sk._static_penetration(body) is None, body


def test_place_over_solid_plate_is_reported():
    """The old hand-placed spare_hole_b (rack centre x, back row) sits over solid upper plate:
    place() must report the snapped tube as penetrating the rack, not as seated."""
    backend = LabBackend()
    m, d = backend.model, backend.data
    sid = m.site("spare_hole_b").id
    m.site_pos[sid][:2] = [RACK_B[0], RACK_B[1] + 0.036]
    mujoco.mj_forward(m, d)

    r = backend.move_tube("dea", "spare_hole_b")

    assert not r["ok"]
    assert "penetrates rackB_upper_plane" in r["reason"]
