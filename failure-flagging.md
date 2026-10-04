# Technician tools: flag a collision, redo the plan

Branch `feat/failure-flagging` (Isaac). Two new technician tools in a new file, `harness/tools/technician_tools.py`.
Nobody else's code changes. The simulation, `lab_backend.py`, both solvers, measurement and scoring are untouched.
The only edit to an existing file is two lines in `lab_tools.py`: an import, plus `*technician_tools(backend)`
added to the list `lab_tools()` returns. That is how the technician gets the tools, since it loads `*lab_tools()`.

## How it fits with what is already on `main`

- **Lok (PR #9):** real collision checks on everything the robot carries. Each collision is logged with
  `backend.log("collision", ...)` into `backend.incidents`, and the action fails.
- **Louis (`dfc6f61`):** a failed action now returns its `reason` (for example `"collision: pipette_shaft vs collide_tube_nacl"`)
  plus `must_retry`. `take_measurement` is refused until that same action is repeated successfully.

These two tools sit on top of that and change neither.

## The tools

**`check_collisions(since_min=0.0)`** gives every collision logged so far in one place, with when it happened, what
was carried and what was hit. A failed tool call reports only its own collision. This tool shows the whole
history, so the technician can see whether anything in the current plan was hit:

```json
{"lab_time_min": 0.08, "collision": true,
 "collisions": [{"t_min": 0.083, "carried": "pipette_shaft", "hit": "collide_tube_nacl"}]}
```

`since_min` limits the report to the current plan, for example the `lab_time_min` at which the plan started.

**`redo_plan(task_name, reason)`** unchecks *every* step of the plan so the technician redoes the whole attempt
from step 1. This covers what retrying only the failed action does not: a collision can spoil steps that had
already succeeded (a knocked tube or a contaminated well). It costs no budget, and the redo is logged as a
`"redo"` lab event (`backend.events`).

## Verified against the current `main` (`fae0647`)

- The branch merges cleanly, and all 20 tests pass after merging (`tests/test_technician_tools.py` plus everyone
  else's).
- End to end: a tube knocked into the pipette's path during `dispense` gives
  `ok: false, reason: "collision: pipette_shaft vs collide_tube_nacl"`, and `check_collisions` then reports that same
  collision.

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_technician_tools.py
```
