# CLAUDE.md

An agentic virtual lab: an LLM agent runs closed-loop experiments in a MuJoCo-simulated
lab with a Franka Panda robot. This repo holds the simulation (scene + motion skills) and a
teammate's harness (backend + Layer-3 tools) that drives it.

## Running things

Uses the `uv` environment (has `mujoco`, `mink`, `daqp`, `opencv-python`, `inspect_ai`).
Run from `lab_sim/`:

- Reach test:  `cd lab_sim && uv run python -m robot.ik_smoke_test`
- Skills test: `cd lab_sim && uv run python -m robot.skills_test`
- Demos/videos: `cd lab_sim && uv run python -m demos.tips` (e.g. -> `experiments/tips.mp4`)
- Viewer (macOS GUI needs mjpython): `mjpython -m scenes.view` (static) or
  `mjpython -m robot.ik_viewer_live` (IK tour)

Demo/video scripts live in `lab_sim/demos/` and are always committed; run them from `lab_sim/`,
never from /tmp. Videos (`lab_sim/experiments/*.mp4`) are gitignored — regenerate from the script.

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
- Weld contract: tubes start welded in their slots; `grasp()` frees its target's world weld
  itself; `place()` snaps the object to its slot and re-welds it.
- Disposable tips (on `feat/pipette-tips`, branched off `feat/gripper-control`): AutoBio 200 uL
  tip mesh (50 mm) in all 24 box slots, each with a top site. `pick_up_tip()`/`eject_tip()`
  switch the active point to `pipette_tip_end` (travel height then follows the whole tip) and
  rebuild the no-collision region over the tip; the mounted tip is excluded from vessel contacts
  ONLY (so it can enter tubes/wells) while still avoided vs bench/racks. Tip inventory +
  per-tip carry-over tracking live in the skill; `pick_up_tip(seat=False)` is the "tip not
  seated" fault hook. Backend tip capacity is 200 uL; `pipette()` takes a fresh tip per transfer
  and ejects it. Demo: `demos/tips.py`.

## Next tasks

1. Held-object collisions (the carried tube/tip vs the scene). Note: the carry path
   currently brushes neighbouring tubes (startup welds hide it); catch and fix it here.
2. Pipetting depth (how far the tip descends into a vessel).
