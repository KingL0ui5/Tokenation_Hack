# Technician tools: flag a collision, redo the plan

Branch `feat/failure-flagging` (Isaac). Two new technician tools in a new file, `harness/tools/technician_tools.py`.
Nobody else's code changes. The simulation, `lab_backend.py`, both solvers, measurement and scoring are untouched.
The only edit to an existing file is two lines in `lab_tools.py`: an import, plus `*technician_tools(backend)`
added to the list `lab_tools()` returns. That is how the technician gets the tools, since it loads `*lab_tools()`.

## The tools

**`check_collisions(since_min=0.0)`** reads the collisions `LabBackend` has logged (`backend.incidents`, fed by
`backend.log("collision", ...)`) and tells the technician what was carried and what it hit:

```json
{"lab_time_min": 3.1, "collision": true,
 "collisions": [{"t_min": 3.0, "carried": "collide_tube_dea", "hit": "collide_tube_tris"}]}
```

`since_min` limits the report to the current plan, for example the `lab_time_min` at which the plan started.

**`redo_plan(task_name, reason)`** unchecks every step of the plan so the technician redoes the attempt from step 1,
then measures. It costs no budget, and the redo is logged as a `"redo"` lab event (`backend.events`). So a spoiled
attempt is redone *before* `take_measurement`, and the bad data point is never taken.

## Works with Lok's collision checks

Lok's `feat/held-object-collisions` (not merged yet) feeds every real collision into `backend.log("collision", ...)`.
I ran his deliberate-collision scenario (a carried tube driven through its neighbour) with these tools added,
and `check_collisions` reported `collide_tube_dea` hitting `collide_tube_tris` with no change to his code. On
`main` today nothing logs collisions yet, so the tool reports none until his branch is merged.

## Tests

`tests/test_technician_tools.py` (3 tests) covers that both tools are in `lab_tools()`, that a logged collision is
reported, and that `redo_plan` unchecks the plan, logs the redo, and spends no budget.

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_technician_tools.py
```

`tests/test_lab_tools.py` already fails on `main` (4 tests: it unpacks `lab_tools()` as 4 tools and assumes the
tip is ejected after every transfer). It's unchanged here, and it's for whoever owns those tests.
