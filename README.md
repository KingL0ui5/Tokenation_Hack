See our demo here: https://docs.google.com/videos/d/1DoNS7-TxRDyN1gSrPa-8kNS4-aSiLtfukvEGQ5S2K7w/play?usp=sharing

# Autonomous Lab: LLM-driven experimental optimisation in a simulated wet lab

Two LLM agents — a **scientist** and a **lab technician** — collaborate to find the optimal
reaction condition for an enzyme assay. The scientist chooses conditions with a Bayesian
optimiser and maintains a legible reasoning graph; the technician physically executes each
experiment with a robot arm in a MuJoCo-simulated lab, where motions can genuinely fail
(collisions, knocked-over tubes, spent pipette tips) and failures have consequences.

Built on [Inspect AI](https://inspect.aisi.org.uk/). Everything is scored against real measured
data (`data/upo_abts.csv`: 814 feasible conditions for unspecific peroxygenase oxidising ABTS).

```bash
uv sync                      # or: pip install -e .
.venv/bin/inspect eval harness/task.py \
    --model anthropic/claude-opus-5-5 \
    -T budget=25 -T seed=0 \
    --log-dir logs --display plain
```

`--display plain` is recommended: the full-screen TUI has crashed mid-run on a stray mouse click,
cancelling the eval before the scorer could write its artifacts.

---

## How a run works

```
            ┌────────────── lab_loop (one round = one experiment) ──────────────┐
            v                                                                   |
  SCIENTIST turn                          TECHNICIAN turn                       |
  bayes_opt_suggest  ->  create_plan  ->  dispense / mix / change_tip  ->  take_measurement
  (GP + EI over all      (condition,      (real MuJoCo motions that        (one unit of budget;
   valid readings)        parent node,     can collide and be refused)      fails if the plan
                          reasoning,                                        is not checked off)
                          checklist)
```

- `lab_loop` (`harness/task.py`) alternates the two agents until the **experiment budget** is
  spent or the scientist **submits**; the scientist gets one final turn to submit if the budget
  runs out first. One shared transcript; each turn is addressed with `SCIENTIST'S TURN` /
  `TECHNICIAN'S TURN`.
- A **plan** (`create_plan`) is one experiment: the condition (`params`, snapped to the nearest
  feasible condition), where it hangs in the reasoning graph (`parent`, `reasoning`), and the
  physical checklist. `create_plan` rejects conditions that snap onto an already-measured point
  (unless `allow_repeat=True`) and parents on closed branches.
- `take_measurement(task_name)` consumes budget and adds the node + edge to the graph. The
  reading is drawn from N(mean, sd) of the matched dataset row. It is **refused** (no budget)
  while any physical action is in a failed state, and returns **MEASUREMENT FAILED** (a node
  with no value) if the checklist was not fully checked off.
- `submit(params)` ends the episode; every open leaf must first be closed with `close_branch`.
  (Note: with an empty graph there are no open leaves, so a scientist that submits before
  measuring anything ends the run immediately — a guard against this existed briefly and was
  removed; re-add in `harness/tools/submit.py` if it bites.)

### Task parameters (`-T name=value`)

| parameter | default | meaning |
|---|---|---|
| `env` | `upo_abts` | environment from `bo_eval/env.py` (`ph`, `salt_conc`, `cosubstrate_conc`, `organic_solvent_conc`, `temperature`) |
| `budget` | 30 | experiments the run may consume (failed measurements count) |
| `seed` | 0 | seeds measurement noise and the BO initial design |
| `max_rounds` | budget | cap on scientist/technician rounds |
| `tolerance` | 0.0 | relative regret within which `found_optimal` counts |
| `graph_dir` | `logs/graphs` | where graph artifacts are exported |

---

## The two reasoning graphs

**Experiment graph** (`LabState.experiment_graph`, scientist-level) — nodes are experiments
(condition -> reading), edges carry the reasoning that led from parent to child, exactly as in
`bo_eval`. `close_branch` marks a node *and all its descendants* as unable to contain the
optimum; closed branches cannot be extended, are excluded from BO suggestions, and must all be
closed before `submit`. Guards learned the hard way:

- closing a branch that contains the current best result is refused;
- the parent of a new experiment should be the node whose *result* it builds on (the BO tool
  reports `suggested_parent` = the incumbent), not simply the previous experiment — a chain-shaped
  graph lets one careless `close_branch` discard the whole search.

**Action graphs** (`LabState.action_graphs`, technician-level) — one graph per *physical action*,
mapping the different ways it could be performed. `map_action` lays out the approaches before
acting; `record_attempt` stores what happened, including the **mistake** on a failed approach;
closing an approach rules it out for the rest of the run. The graphs are shown to the technician
every turn, so a mistake (e.g. "carried the tube below rack height across the cold block") is
made at most once.

Both graphs render to the same Mermaid dialect: closed nodes red/dashed/crossed, failed
measurements amber.

## The simulated lab (`lab_sim/` + `harness/tools/lab_*`)

A Franka Panda on a bench with a well plate, two 15 mL reagent racks, a chilled cold block
(enzyme tube), a 24-slot disposable-tip box and waste bins, driven through IK
(`lab_sim/robot/skills.py`). The scene publishes its own contract (site names, obstacles, vessel
dimensions) from `lab_sim/scenes/build_lab.py`.

What is real and what is not:

- **Real**: arm kinematics, collision avoidance and collision *detection* (penetrating contacts
  of whatever the robot carries), travel/descend motion shaping, tip pick-up/eject physics,
  tube grasp/place with free-joint physics.
- **Not simulated**: liquid, volumes, chemistry. A tool reports whether the *motion* succeeded
  and why it failed; the measurement comes from the dataset, not the well contents. Intended
  volumes, tip identity and carry-over history go to a hidden ledger (`LabBackend.ledger`),
  never to the agent.

### Failure machinery

- Any new penetrating contact while carrying something (mounted tip, grasped tube) is an
  **incident**; with `halt_on_incident` the arm stops dead within `halt_latency_s` (100 ms in
  the demos).
- The first incident latches a **safety stop**: every subsequent motion tool is refused
  (`"refused: safety stop latched after collision with ..."`, with a simulated **camera check**
  naming displaced or knocked-over tubes) until `reset_safety_stop()` — a human clearing the
  bench.
- A failed action is remembered in `unresolved_failures`: `take_measurement` is refused, at no
  budget cost, until the same action is repeated successfully or the scientist replaces the plan.
- **Tips**: the pipette keeps one disposable tip across operations (carry-over accrues until
  `change_tip`); the box holds 24 and `refresh_tips` swaps in a full box at a lab-time cost.

### Tool summary

| agent | tools |
|---|---|
| scientist | `bayes_opt_suggest`, `create_plan`, `view_plan`, `view_graph`, `add_reasoning`, `close_branch`, `submit` |
| technician | `complete_step`, `view_plan`, `take_measurement`, `map_action`, `record_attempt`, `view_actions`, `dispense`, `transfer_sample`, `mix`, `change_tip`, `refresh_tips`, `get_lab_state` |

`bayes_opt_suggest` fits a GP (Matern 2.5 + white noise) to every **valid** reading and ranks
unmeasured conditions by expected improvement; below two readings it returns a random initial
design. It never proposes a measured point and reports the incumbent.

## Scoring (`harness/scorer.py`)

| metric | meaning |
|---|---|
| `found_optimal` | submitted (else best measured) condition within `tolerance` relative regret of the true optimum |
| `regret` | relative gap in true mean to the optimum (1.0 if nothing measured) |
| `n_experiments` | budget spent, including failed measurements |
| `n_invalid` | measurements taken on an incomplete plan (no reading) |
| `n_plans_finished` | checklists the technician fully checked off |

The scorer exports `logs/graphs/<sample>_epoch<n>.md` (Mermaid + text for the experiment graph,
plans, and every action graph) and `.json`. View Mermaid in VS Code's markdown preview, GitHub,
or https://mermaid.live. For a cancelled/crashed run the scorer never fires, but the store
survives in the `.eval` log — the graph can be rebuilt from it (see `inspect view --log-dir logs`
for the full transcript).

## `bo_eval/`: the original single-agent eval

The harness is derived from `bo_eval`, which runs ONE react agent with `run_experiment` directly
(no technician, no physical lab) plus the no-LLM baselines:

```bash
inspect eval bo_eval/task.py --model anthropic/claude-opus-5-5 -T solver=react -T budget=30
inspect eval bo_eval/task.py -T solver=bo --model mockllm/model      # GP-EI baseline, no LLM
inspect eval bo_eval/task.py -T solver=random --model mockllm/model
```

Solvers: `react`, `react_no_bo`, `react_no_graph`, `bo`, `random` (`bo_eval/solvers.py`).
Useful as the optimiser-only reference when debugging the two-agent harness.

## Demos (`demo/` and `lab_sim/demos/`)

**Collision demo video** — a scripted but mechanically real two-act story: the technician maps an
action's approaches, a low tube-carry knocks the (deliberately unsecured) enzyme tube over —
real MuJoCo contact, red flash, 90° topple, 100 ms halt, latched safety stop, refused tools,
refused measurement — the mistake is recorded on the action graph and the approach closed, a
human resets the bench on camera, and the experiment re-runs cleanly a safer way. Rendered with
a synchronized overlay: LLM<->lab tool traffic, the live reasoning/action graph (blocked nodes
crossed out), and a status strip.

```bash
.venv/bin/python -m demo.scenario      # drive the lab; frames + events.jsonl -> demo/out/
.venv/bin/python -m demo.make_video    # composite -> demo/out/collision_demo.mp4 (1920x1080)
.venv/bin/python -m demo.fixture && \
.venv/bin/python -m demo.make_video --frames demo/out_fixture/frames \
    --events demo/out_fixture/events.jsonl --out demo/out_fixture/test.mp4   # compositor self-test
```

The agents' prose in the video is scripted for determinism; every tool call, motion, refusal and
graph mutation is the real code path. The only injected fault (labelled on screen) is releasing
the enzyme tube's weld and driving the carry low.

**Arm rave** — the arm dances to a self-synthesised 128 BPM techno track, beat-locked
choreography, strobing light, orbiting sidechain-pumped camera and a live spectrum overlay:

```bash
.venv/bin/python -m demo.rave          # -> demo/out/rave.mp4 (with audio)
```

**`lab_sim/demos/`** — standalone robotics clips (no LLM overlay): `knock_over.py`,
`pipetting_workflow.py`, `tips.py`, `held_object_collisions.py`, rendered via `render.py`.
Run from the repo root, e.g. `python -m lab_sim.demos.knock_over --quick`.

## Repository layout

```
bo_eval/            original single-agent eval: env (tabular data), tools, solvers, scorer
harness/
  task.py           autonomous_lab_task: init, brief/loop solvers, task parameters
  solvers/          scientist.py, technician.py (+ shared prompt context)
  tools/            bayes_opt, plan, take_measurement, graph, action_graph, submit,
                    lab_tools (-> lab_backend -> lab_sim), with the hidden truth ledger
  types/state.py    LabState: experiment_graph, task_plans, action_graphs, submission
  scorer.py         metrics + graph export
lab_sim/
  scenes/build_lab.py   parametric scene (plates, racks, tips, cameras) -> lab.xml
  robot/skills.py       IK pipette/gripper skills, incident detection, halting
  demos/                standalone rendered clips
demo/               the composited collision demo video + the rave
data/upo_abts.csv   measured UPO/ABTS rates: 5 parameters, 814 feasible conditions
logs/               .eval logs and exported graphs
```

## Development notes

- Python >= 3.10 (repo venv is 3.13), managed with `uv`; MuJoCo renders offscreen, no display
  needed. `imageio-ffmpeg` (dev dependency) provides the ffmpeg binary used only by `demo/rave.py`.
- Smoke-test any harness change without API cost:
  `inspect eval harness/task.py --model mockllm/model -T budget=3 --display plain`
  then check the tool lists and turn alternation in the resulting log.
- Haiku-class models run the loop but tend to override BO suggestions and mis-handle the graph;
  the guards in `create_plan` / `close_branch` exist because of those runs. The technician's
  200k context can bind on long budgets — the transcript is shared and grows every round.
- Known limitation: physical actions are theatre with respect to the measurement (no liquid
  model), so the plan checklist, tip hygiene and the ledger record *procedure*, not chemistry.
  Lab time (`lab_time_min`) is tracked but not yet scored.
```
