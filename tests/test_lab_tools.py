"""Tests for harness/tools/lab_tools.py against Lok's lab_sim scene.

Run from the repo root:  pytest tests/test_lab_tools.py -q
"""

import asyncio
import json
import sys
from pathlib import Path

import pytest
from inspect_ai.tool import ContentImage, ToolError
from inspect_ai.tool._tool_def import ToolDef

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.tools.lab_backend import LabBackend  # noqa: E402
from harness.tools.lab_tools import lab_tools  # noqa: E402

EXPECTED = {"dispense", "transfer_sample", "mix", "incubate", "measure", "measure_standard",
            "get_lab_state", "inspect", "discard", "check_pipette", "request_human_help",
            "capture_camera", "get_robot_state"}


def run(coro):
    return asyncio.run(coro)


def tools_for(**kwargs):
    b = LabBackend(seed=0, **kwargs)
    return b, {ToolDef(t).name: t for t in lab_tools(b)}


@pytest.fixture(scope="module")
def lab():
    return tools_for()


def test_tool_names_and_schemas(lab):
    _, t = lab
    assert set(t) == EXPECTED
    for name, fn in t.items():
        assert ToolDef(fn).description, name


def test_assay_happy_path(lab):
    b, t = lab
    for reagent, vol in [("buffer", 800), ("substrate", 100), ("enzyme", 50)]:
        r = json.loads(run(t["dispense"](reagent=reagent, destination="B3", volume_ul=vol)))
        assert r["ok"] and not r["discrepancies"], r
    run(t["dispense"](reagent="buffer", destination="B4", volume_ul=950))   # no-enzyme blank
    assert json.loads(run(t["mix"](container="B3")))["ok"]
    run(t["incubate"](minutes=10, temperature_c=37))
    m = json.loads(run(t["measure"](wells=["B3", "B4"])))
    readings = {x["well"]: x["absorbance_405nm"] for x in m["observation"]["readings"]}
    assert readings["B3"] > readings["B4"] + 0.05, readings
    state = json.loads(run(t["get_lab_state"]()))["observation"]
    assert state["containers"]["well_B3"]["volume_ul"] == pytest.approx(950)  # nominal, as commanded
    assert state["budget_remaining"]["wells"] == 24 - 2
    assert not state["unresolved_incidents"]
    assert len(b.ledger) >= 7 and b.ledger[0].actual["delivered_ul"] != 800  # real volume differs


def test_validation_rejects_before_motion(lab):
    _, t = lab
    with pytest.raises(ToolError, match="holds .* of .* requested"):
        run(t["dispense"](reagent="buffer", destination="B3", volume_ul=5000))
    with pytest.raises(ToolError, match="unknown container"):
        run(t["dispense"](reagent="buffer", destination="Z9", volume_ul=10))
    with pytest.raises(ToolError, match="is empty"):
        run(t["mix"](container="D6"))
    with pytest.raises(ToolError, match="not a reagent"):
        run(t["dispense"](reagent="B3", destination="B4", volume_ul=10))


def test_standards_and_pipette_check(lab):
    _, t = lab
    s = json.loads(run(t["measure_standard"](standard="high")))["observation"]
    assert abs(s["reading_AU"] - 1.5) < 0.3
    p = json.loads(run(t["check_pipette"](volume_ul=100)))["observation"]
    assert 90 < p["mass_mg"] < 110


def test_camera_and_robot_state(lab):
    _, t = lab
    img = run(t["capture_camera"](camera="plate_top"))
    assert isinstance(img, ContentImage) or img.startswith("camera unavailable")
    assert "joint_positions_rad" in json.loads(run(t["get_robot_state"]()))


def test_spill_blocks_until_response():
    b, t = tools_for(plate_offset_mm=(0.0, 9.0))   # 7 mm wells on an 18 mm pitch -> spills
    r = json.loads(run(t["dispense"](reagent="buffer", destination="B3", volume_ul=200)))
    assert not r["ok"] and any("spill" in d for d in r["discrepancies"])
    with pytest.raises(ToolError, match="unresolved incident"):
        run(t["dispense"](reagent="buffer", destination="B4", volume_ul=200))
    assert b.blocked_attempts == 1
    seen = json.loads(run(t["inspect"](container="B3")))["observation"]
    assert seen["liquid_outside_container"] and seen["incidents_acknowledged"]
    assert not b.incidents


def test_wrong_well_and_discard():
    b, t = tools_for(plate_offset_mm=(0.0, 18.0))  # one well pitch -> lands in the neighbour
    r = json.loads(run(t["dispense"](reagent="buffer", destination="B3", volume_ul=200)))
    assert any("instead of" in d for d in r["discrepancies"])
    landed = b.ledger[-1].actual["dest"]
    assert landed not in (None, "well_B3")
    run(t["request_human_help"](reason="plate looks misaligned"))
    assert not b.incidents
    run(t["discard"](container=landed))
    assert b.actual[landed].volume_ul == 0


def test_inside_inspect_eval(tmp_path):
    """Tools work inside a real Inspect eval, one backend per sample, created on first use."""
    from inspect_ai import Task, eval
    from inspect_ai.dataset import Sample
    from inspect_ai.model import ModelOutput, get_model
    from inspect_ai.solver import generate, use_tools

    calls = [("dispense", {"reagent": "buffer", "destination": "A1", "volume_ul": 300}),
             ("measure", {"wells": ["A1"]}),
             ("get_lab_state", {})]
    outputs = [ModelOutput.for_tool_call("mockllm/model", n, a) for n, a in calls]
    outputs.append(ModelOutput.from_content("mockllm/model", "done"))
    task = Task(dataset=[Sample(input="run the lab", metadata={"seed": 3})],
                solver=[use_tools(lab_tools()), generate()])
    log = eval(task, model=get_model("mockllm/model", custom_outputs=outputs), display="none",
               log_dir=str(tmp_path))[0]
    assert log.status == "success"
    tool_msgs = [m for m in log.samples[0].messages if m.role == "tool"]
    assert len(tool_msgs) == 3 and not any(m.error for m in tool_msgs)
