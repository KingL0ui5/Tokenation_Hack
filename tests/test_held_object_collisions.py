"""Collision checks for held objects (robot/skills.py's PipetteSkills._record_incidents),
limited to what the robot actually carries: the mounted pipette (with or without a disposable
tip) and a grasped 15 mL tube. See CLAUDE.md "Next tasks" for the bug this fixes (the tube
carry path used to brush neighbouring tubes) and harness/tools/lab_backend.py (`_drain_incidents`,
`move_tube`, `pipette`) for how an incident fails a tool the same way any other motion failure
does.

Builds the real lab_sim scene (mujoco/mink), same convention as test_lab_tools.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lab_sim"))

import pytest

from harness.tools.lab_backend import LabBackend


@pytest.fixture
def backend():
    return LabBackend()


def test_carry_tube_rack_a_to_spare_hole_rack_b_zero_incidents(backend):
    """(a) Carry a tube from rack A to the spare hole in rack B: the gripper has to thread
    down between tightly-packed neighbours (36 mm pitch) and travel across the bench without
    brushing anything."""
    r = backend.move_tube("dea", "spare_hole_b")

    assert r["ok"], r.get("reason")
    assert backend.skills.incidents == []
    assert backend.incidents == []
    assert backend.skills.held_object is None    # tube was placed, not left in the gripper
    assert backend.skills.held                   # pipette was re-mounted after the carry


def test_pipette_with_tip_travels_between_racks_zero_incidents(backend):
    """(b) The mounted pipette (with a disposable tip) travelling between rack A, rack B and a
    well should never touch a rack wall, the bench, or any tube it isn't targeting."""
    sk = backend.skills
    assert sk.pick_up_tip().ok

    assert sk.travel_to("reagent_dea", clearance=0.04).ok      # rack A
    assert sk.travel_to("reagent_zncl2", clearance=0.04).ok    # rack B
    assert sk.travel_to("well_A1", clearance=0.04).ok          # plate

    assert sk.incidents == []
    assert sk.eject_tip().ok


def test_deliberate_collision_is_caught_and_reported(backend):
    """(c) A tube deliberately driven sideways through a neighbour's slot (skipping the safe
    lift) must register as an incident at the skill layer, and the backend must surface it
    through the same channel a real collision would use (log("collision", ...) /
    backend.incidents), not silently succeed."""
    sk = backend.skills
    sk.put_down_pipette()
    sk.close_gripper(0.022)
    sk.travel_to("tube_grip_dea", clearance=0.04)
    sk.descend(0.04)
    assert sk.grasp("dea").ok
    sk.ascend()
    assert sk.incidents == []            # the normal approach+lift is clean

    # Deliberately skip travel_to's lift-clear-then-descend shape: go straight at a neighbour's
    # height, through its slot.
    tris_id = sk.model.body("tubebody_tris").id
    target_z = sk.data.xpos[tris_id][2] + 0.05 + sk._held_lowest_z_offset()
    sk._goto([sk.data.xpos[tris_id][0], sk.data.xpos[tris_id][1], target_z], 0.8)

    assert sk.incidents, "expected the carried tube to be caught brushing its neighbour"
    inc = sk.incidents[0]
    assert inc["carried"] == "collide_tube_dea"
    assert inc["other"] == "collide_tube_tris"

    bad = backend._drain_incidents()
    assert bad == sk.incidents
    assert backend.incidents and backend.incidents[-1]["kind"] == "collision"
    assert backend.incidents[-1]["other"] == "collide_tube_tris"


def test_liftoff_exemption_ends_once_lifted_clear(backend):
    """(d) What a tube rested on at grasp time (its rack floor) is only exempt while it lifts
    off. Once it's lifted clear, driving it back into that same floor is a real collision."""
    sk = backend.skills
    sk.put_down_pipette()
    sk.close_gripper(0.022)
    sk.travel_to("tube_grip_dea", clearance=0.04)
    sk.descend(0.04)
    assert sk.grasp("dea").ok
    floor = sk.model.geom("collide_rackA_floor").id
    assert floor in sk._held_liftoff_exempt          # it was standing on the rack floor
    sk.ascend()
    assert sk.incidents == []                        # lifting off the floor is expected
    assert not sk._held_liftoff_exempt               # ...but the exemption ended with liftoff

    # Deliberately push it back down through the floor (bypassing the avoidance limit, as a
    # mis-planned motion would): this must now be caught, not silently exempted.
    sk.limits = sk.limits[:1]
    sk.halt_on_incident = False                      # record every contact, don't stop at the first
    tp = sk.tip()
    base_z = sk.data.xpos[sk.model.body("tubebody_dea").id][2]
    sk._goto([tp[0], tp[1], tp[2] - base_z + 0.003], 0.8)   # tube base 3 mm into the 6 mm floor

    assert any(i["carried"] == "collide_tube_dea" and i["other"] == "collide_rackA_floor"
               for i in sk.incidents), sk.incidents
