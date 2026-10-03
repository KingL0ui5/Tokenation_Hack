# CLAUDE.md

An agentic virtual lab: an LLM agent runs closed-loop experiments in a MuJoCo-simulated
lab with a Franka Panda robot. This repo holds the simulation (scene + motion skills) and a
teammate's harness (backend + Layer-3 tools) that drives it.

## Running things

Uses the `uv` environment (has `mujoco`, `mink`, `daqp`, `opencv-python`, `inspect_ai`).
Run from `lab_sim/`:

- Reach test:  `cd lab_sim && uv run python -m robot.ik_smoke_test`
- Skills test: `cd lab_sim && uv run python -m robot.skills_test`
- Viewer (macOS GUI needs mjpython): `mjpython -m scenes.view` (static) or
  `mjpython -m robot.ik_viewer_live` (IK tour)

## Repo layout

- `lab_sim/` — mine. `scenes/` (build_lab.py, the scene + `scene_contract()`),
  `robot/` (skills.py, IK, tests), `models/` (Panda + AutoBio assets).
- `harness/` — teammate's. `tools/lab_backend.py` (backend) and `tools/lab_tools.py`
  (Layer-3 Inspect tools). Reads `scene_contract()`; don't refactor it unasked.

## Rules

- Never edit anything under `lab_sim/models/` (Panda stays pristine).
- AutoBio meshes are visual-only (contype/conaffinity 0); colliders are separate primitives.
  Source + licence in `lab_sim/models/autobio/NOTICE.md` — never modify the mesh files.
- No randomness in motion code (that belongs in the noise layer).
- Skills return result objects and never raise.
- Check placed meshes against their slots (position/seating), not just that IK converged.
- Watch recorded videos before reporting. Keep reports short.

## Gotchas

- Scene is generated: edit `build_lab.py`, never the generated `lab.xml`. The EE site,
  pipette mount, and welds are injected in `load_model()` via MjSpec, not by editing panda.xml.
- Free joints (tubes, plate) shift qpos/qvel/ctrl indices. Never address the arm by `[:7]`;
  use `jnt_qposadr`/`jnt_dofadr`/actuator ids. This bug broke every skill once.

## Git

- Branch from main. Commit/push only when asked; re-fetch before push.
- I review the recorded video before any merge. Include the Co-Authored-By line.

## Current status

- Pipette mounted on the hand; swapped to/from a stand via the visibility+collision toggle trick.
- Gap-based gripper (`close_gripper(gap)`); `grasp()` verifies both finger pads by contact.
- `place()` snaps the tube kinematically into its slot and welds it to the world.
- `ascend()` uses a stepped IK solve (reseeds from home if a solve stalls).

## Next tasks

1. Tips (attach/eject; active point switches to tip end, no-collision over the whole tip).
2. Held-object collisions (the carried tube/tip vs the scene).
3. Pipetting depth (how far the tip descends into a vessel).
