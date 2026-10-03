# Lab tools

`lab_tools(lab)` in `lab_tools.py` binds every tool to one `harness.lab.Lab` (one per sample).

| Group | Tools | What they touch |
| --- | --- | --- |
| Observe | `get_lab_status`, `get_deck_layout`, `get_robot_state`, `get_event_log`, `capture_camera`, `get_plate_result`, `get_observations` | lab_sim MuJoCo state (joints, torques, tip pose, contacts), event log (spills, wrong-well, collisions, IK failures), lab_sim cameras, plate results |
| Robot | `move_tip`, `move_tip_to`, `set_gripper`, `aspirate`, `dispense`, `change_tip` | Panda arm on Lok's lab_sim scene: mink IK with collision avoidance + joint-space trajectory on the position servos (no learned policy) |
| Protocol | `design_batch`, `run_plate`, `enzyme_titration`, `spike_recovery`, `dual_wavelength` | Full plate execution by the robot, kinetic A405 reads, yields, outlier rule, validity gate, budget |
| Analysis | `fit_model`, `suggest_ucb`, `mutual_information` | GP (Matern 3/2, ARD) on valid yields; batch UCB over the 65,536-point grid |
| Notebook | `record_prior`, `revise_prior`, `add_reasoning_node`, `get_notebook`, `submit_report` | Reasoning graph and final report, kept in the Inspect store |

What MuJoCo provides versus what is modelled: MuJoCo (Lok's `lab_sim` scene, driven by
`harness/lab/labsim_world.py`) decides where the pipette tip actually
goes (and so whether liquid lands in the well). Liquid volumes, enzyme kinetics and the
plate reader are software models in `harness/lab/chemistry.py`, driven by what each well
actually received. The literature lookups (`lookup_kinetics`, `lookup_hazards`,
`search_literature`, `check_compatibility`) from the protocol are not built yet.
