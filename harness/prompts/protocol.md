# Agentic Enzyme Optimisation — Agent Protocol (agent-facing excerpt)

Source: "Agentic Enzyme Optimisation — Agent Protocol", Oct 3 2026, @Anabel. Sections
about how the protocol is wired into the harness, the reward function and the team's
open decisions are deliberately not shown to the agent.

This is a procedure to follow and adapt, not a script to execute. Deciding when to
deviate from it, and saying so in the notebook, is what the run is scored on.

## The discrete variable grid

Levels are fractions of each variable's maximum: L1 = 10%, L2 = 30%, L3 = 50%, L4 = 100%.
State conditions as level codes, never raw volumes; the tool layer converts levels to volumes.

| Variable | Role | Max (L4) | Notes |
| --- | --- | --- | --- |
| Buffer identity | Sets pH and may itself react | — | Categorical: DEA, Tris, Glycine, PBS |
| pH | Protonation of catalytic residues | — | Categorical: 7.0, 8.0, 9.0, 10.0 |
| Substrate (pNPP) | Saturation vs depletion | 10 mM | Below Km the rate is substrate-limited; above it, depletion risk |
| MgCl2 | Catalytic metal | 5 mM | Required cofactor |
| ZnCl2 | Catalytic metal | 0.1 mM | Required; excess inhibits |
| NaCl | Ionic strength | 300 mM | Shields active-site electrostatics |
| Glycerol | Stabiliser | 20% v/v | Also raises viscosity |
| Incubation temperature | Activity vs stability | 45 °C | Categorical: 25, 30, 37, 45 °C |

Buffer usable pH ranges: DEA 9.0–10.5, Tris 7.0–9.0, Glycine 8.5–10.5, PBS 6.0–8.0.
Investigate each buffer's hazards for this assay before relying on it.

Reference composition (run on every plate; every yield is relative to it): DEA buffer,
pH 9.8, substrate L3, MgCl2 L3, ZnCl2 L3, NaCl L2, glycerol L1, 37 °C.

Off-grid values may be requested only with a stated reason; the tool layer logs it.

## The readout and its validity rules

Yield_x = (S_x − S_blank) / (S_ref − S_blank), where S is the initial rate
(µM product min⁻¹ µg⁻¹ enzyme), S_blank the no-enzyme control for that buffer and pH,
S_ref the reference composition on that plate.

Rate, not endpoint: each condition is read at ≥3 time points and the slope is taken over
the linear region. A curve non-linear across the whole window is rate-indeterminate, not low.

Mandatory controls on every plate (a plate missing any is invalid): reference (3),
no-enzyme blank per distinct buffer and pH (3 each), positive control (3), product
standard curve (5 points), carry-over wells from the previous batch's best (3).

Replication and outliers: every condition in triplicate. If CV across three replicates
exceeds 30%, the value farthest from the other two is dropped and flagged. More than 10%
of a plate's conditions triggering this means the plate is suspect, not the data.

Plate validity gate — check before analysing any result:
1. Control yields agree with previous plates.
2. Z′ from positive and negative controls exceeds 0.5.
3. No control well outside the detector's linear range.
4. Product standard curve linear with R² > 0.98.
A failed plate is re-run. Its data must not enter the model.

Edge effects: map outliers by well position before treating them as chemistry. A cluster
on outer wells is an evaporation artefact.

Orthogonal confirmation: before declaring an optimum, confirm it with at least one of
enzyme titration, spike-and-recovery, or dual-wavelength read. Each costs budget.

## The run protocol

Phase 0 — Prior (before any experiment): state priors as ranges with stated confidence,
never point estimates; record every claim with its source (or say it has none); declare
exclusions with reasons; write the prior to the notebook before seeing any data.

Phase 1 — Initial design: one-at-a-time extremes (for each variable, all at L4 except that
one at L1, and all at L1 except that one at L4), a few random compositions, the reference
and all mandatory controls.

Phase 2 — Batch loop, each iteration:
1. Check plate validity first. If it fails, re-run and discard its data.
2. Apply the outlier rule, then map outliers by position.
3. Fit the surrogate model with the analysis tool, not arithmetic in the reply.
4. Compare against the prior. If data contradict it beyond its stated confidence, say so,
   name the failed claim, and revise it in the notebook.
5. Select the next batch by UCB: exploitation × mean + exploration × sd, exploration = √2.
   Deviating is allowed if justified in the notebook.
6. Every condition passes the hazard interlock before it is proposed.

Phase 3 — Confirmation and stop. Stop when neither the best yield nor the model's
cross-validated accuracy improved over two consecutive batches, the budget is spent, or
the optimum's uncertainty is below tolerance — and say which. Run orthogonal confirmation,
then report: the optimum with a confidence interval, the variables that mattered (mutual
information) and those that did not, every revised prior and its evidence, and anything
not determined. An honest "not determined" scores better than a confident wrong answer.

Mandatory stops — halt and escalate instead of continuing if: a control fails twice in a
row; a reagent has no hazard record; a spill or drop is detected; the interlock rejects a
proposed condition; or the data are internally inconsistent in a way you cannot explain.
