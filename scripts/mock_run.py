"""End-to-end check of the tool layer through Inspect with a scripted mock model (no API calls)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inspect_ai import eval
from inspect_ai.model import ModelOutput, get_model

from harness.lab import config as C
from harness.task import enzyme_optimisation

REF = dict(C.REFERENCE)
CONDS = [dict(REF, pH=9.0), dict(REF, pH=10.0), dict(REF, buffer="Glycine", pH=9.0),
         dict(REF, substrate="L4", MgCl2="L4"), dict(REF, pH=9.0, temperature=30),
         dict(REF, pH=9.0, ZnCl2="L1"), dict(REF, buffer="Tris", pH=8.0)]
CTRL = ["reference", "blanks", "positive", "standard_curve", "carry_over"]
calls = [
    ("record_prior", dict(claim="pH optimum between 9 and 10.5", variable="pH", low=9.0, high=10.5,
                          unit="pH", confidence=0.7, sources=[])),
    ("get_deck_layout", {}),
    ("move_tip_to", dict(location="well:A1")),
    ("get_robot_state", {}),
    ("capture_camera", dict(camera="overview")),
    ("move_tip_to", dict(location="home")),
    ("design_batch", dict(conditions=CONDS, controls=CTRL[:3])),   # should be rejected
    ("design_batch", dict(conditions=CONDS, controls=CTRL, avoid_edges=True)),
    ("run_plate", dict(batch_id="B01")),
    ("fit_model", {}),
    ("mutual_information", {}),
    ("suggest_ucb", dict(n=4, exclude={"buffer": ["PBS"]})),
    ("add_reasoning_node", dict(kind="decision", statement="confirm best observed", parents=["N001"],
                                evidence=["B01"])),
    ("enzyme_titration", dict(condition=dict(REF, pH=9.0))),
    ("get_event_log", dict(kinds=["spill", "collision"])),
    ("submit_report", dict(optimum=dict(REF, pH=9.0), yield_estimate=1.0, yield_ci_low=0.8,
                           yield_ci_high=1.2, important_variables=["buffer"], unimportant_variables=[],
                           revised_prior_ids=[], confirmation_performed=["enzyme_titration"],
                           stop_reason="mock run", not_determined=["everything else"])),
]
outputs = [ModelOutput.for_tool_call("mockllm/model", name, args) for name, args in calls]
outputs.append(ModelOutput.from_content("mockllm/model", "done"))
model = get_model("mockllm/model", custom_outputs=outputs)

log = eval(enzyme_optimisation(), model=model, sample_id="nominal-1", log_dir="logs/mock")[0]
print("status:", log.status)
sample = log.samples[0]
for m in sample.messages:
    if m.role == "tool":
        txt = m.text if isinstance(m.text, str) else str(m.content)
        err = f" ERROR={m.error.message[:200]}" if m.error else ""
        print(f"- {m.function}: {txt[:160]!r}{err}")
print("score:", sample.scores)
