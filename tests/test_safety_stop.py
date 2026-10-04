"""Latched safety stop (harness/tools/lab_backend.py): the first collision stops the arm dead
(PipetteSkills.halt_on_incident), latches the stop, and every motion tool is then refused --
naming the collision and counting the attempts -- until an explicit reset_safety_stop().
Builds the real lab_sim scene, same convention as test_lab_tools.py."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lab_sim"))

import pytest

from harness.tools.lab_backend import LabBackend


@pytest.fixture
def collided():
    """A backend right after a deliberate collision: dea carried sideways through tris."""
    b = LabBackend()
    sk = b.skills
    sk.put_down_pipette()
    sk.close_gripper(0.022)
    sk.travel_to("tube_grip_dea", clearance=0.04)
    sk.descend(0.04)
    assert sk.grasp("dea").ok
    sk.ascend()
    tris = sk.data.xpos[sk.model.body("tubebody_tris").id]
    ok, reason = sk._goto([tris[0], tris[1], tris[2] + 0.05 + sk._held_lowest_z_offset()], 0.8)
    return b, ok, reason


def test_collision_stops_the_move_and_latches(collided):
    b, ok, reason = collided
    assert not ok and reason.startswith("stopped: collision collide_tube_dea vs collide_tube_tris")
    status = b.safety_status()
    assert status["halted"]
    assert status["collision"]["other"] == "tube_tris"


def test_every_motion_tool_is_refused_and_counted(collided):
    b, _, _ = collided
    calls = [lambda: b.pipette("reagent_water", "well_A1"), lambda: b.mix("well_A1", 2),
             lambda: b.change_tip(), lambda: b.move_tube("nacl", "spare_hole")]
    for n, call in enumerate(calls, start=1):
        r = call()
        assert not r["ok"] and r["refused"]
        assert "collision with tube_tris" in r["reason"]
        assert b.refused_attempts == n
    assert [e["kind"] for e in b.events].count("refused") == len(calls)


def test_explicit_reset_unlatches(collided):
    b, _, _ = collided
    b.safety_status()
    r = b.reset_safety_stop()
    assert r["ok"] and r["cleared"]["other"] == "tube_tris"
    assert not b.safety_status()["halted"]
    assert not b.mix("well_A1", 1).get("refused")


def test_camera_check_names_tubes_by_slot_from_the_scene_map():
    """The simulated camera check compares tube poses with the scene map: clean bench -> nothing;
    a tipped or shifted tube is named by the (now empty) slot it left, not by contact data."""
    import mujoco
    import numpy as np

    b = LabBackend()
    sk, m, d = b.skills, b.model, b.data
    assert b.camera_check() == "no displaced or knocked-over tubes"

    qa = sk.free_qadr["tubebody_enzyme"]
    sk._world_weld("tubebody_enzyme", False)
    d.qpos[qa + 3:qa + 7] = [np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0]      # 90 deg about x
    mujoco.mj_forward(m, d)
    assert b.camera_check() == "tube near tube_enzyme's slot knocked over (tilt 90°)"

    d.qpos[qa + 3:qa + 7] = [1, 0, 0, 0]
    d.qpos[qa] += 0.02                                                        # upright, 2 cm off
    mujoco.mj_forward(m, d)
    assert b.camera_check() == "tube near tube_enzyme's slot displaced (20 mm)"
