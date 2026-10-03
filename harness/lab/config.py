"""Protocol constants: the discrete condition grid, reagent stocks and deck layout.

Numbers follow "Agentic Enzyme Optimisation — Agent Protocol". The maxima are
placeholders pending verification (see the protocol's open decisions).
"""

from __future__ import annotations

import itertools

LEVELS = {"L1": 0.10, "L2": 0.30, "L3": 0.50, "L4": 1.00}

BUFFERS = {
    # name: (usable pH range, pKa at 25 C)
    "DEA": ((9.0, 10.5), 8.9),
    "Tris": ((7.0, 9.0), 8.1),
    "Glycine": ((8.5, 10.5), 9.6),
    "PBS": ((6.0, 8.0), 7.2),
}
PH_VALUES = [7.0, 8.0, 9.0, 10.0]
TEMPERATURES_C = [25, 30, 37, 45]

# Level-coded additives: final-well maximum (L4) and units.
ADDITIVES = {
    "substrate": (10.0, "mM pNPP"),
    "MgCl2": (5.0, "mM"),
    "ZnCl2": (0.1, "mM"),
    "NaCl": (300.0, "mM"),
    "glycerol": (20.0, "% v/v"),
}

VARIABLES = ["buffer", "pH", *ADDITIVES, "temperature"]

REFERENCE = {
    "buffer": "DEA", "pH": 9.8, "substrate": "L3", "MgCl2": "L3",
    "ZnCl2": "L3", "NaCl": "L2", "glycerol": "L1", "temperature": 37,
}

# Liquid handling, sized for lab_sim's 4x6 plate (14 mm wells, like a 24-well plate).
# Additive stocks are 10x the L4 maximum, so L4 = 100 uL and L1 = 10 uL.
WELL_VOLUME_UL = 1000.0
BUFFER_VOLUME_UL = WELL_VOLUME_UL / 3          # 3x buffer stock
ADDITIVE_L4_VOLUME_UL = WELL_VOLUME_UL / 10    # 10x stock
ENZYME_VOLUME_UL = WELL_VOLUME_UL / 30
ENZYME_UG_PER_WELL = 0.15         # nominal enzyme mass at 1x
TIP_CAPACITY_UL = 1000.0

READ_TIMES_MIN = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
DETECTOR_LINEAR_MAX_A = 2.5
DETECTOR_SATURATION_A = 3.5
PNP_EPSILON_405 = 18.5            # mM^-1 cm^-1 for p-nitrophenolate
PATH_LENGTH_CM = 0.65             # 1000 uL in a 14 mm well (lab_sim WELL_R = 7 mm)
STANDARD_CURVE_MM = [0.0, 0.025, 0.05, 0.08, 0.12]  # top point stays below the 2.5 A linear limit

# Plate geometry (wells, edge wells, positions) comes from the lab_sim scene; see labsim_world.py.


def buffer_reagent(buffer: str, ph: float) -> str:
    return f"buffer:{buffer}@pH{ph:g}"


def reagent_names() -> list[str]:
    names = [buffer_reagent(b, p) for b in BUFFERS for p in PH_VALUES]
    names.append(buffer_reagent("DEA", REFERENCE["pH"]))
    names += list(ADDITIVES)
    names += ["enzyme", "pNP_standard", "water"]
    return names


def grid_size() -> int:
    return len(BUFFERS) * len(PH_VALUES) * len(LEVELS) ** len(ADDITIVES) * len(TEMPERATURES_C)


def iter_grid():
    """Yield every on-grid condition (65,536 of them) as a dict."""
    for b, p, *levels, t in itertools.product(
        BUFFERS, PH_VALUES, *([list(LEVELS)] * len(ADDITIVES)), TEMPERATURES_C
    ):
        yield {"buffer": b, "pH": p, **dict(zip(ADDITIVES, levels)), "temperature": t}


def validate_condition(cond: dict, allow_off_grid: bool = False) -> list[str]:
    """Return a list of problems with a condition; empty means valid."""
    problems = []
    missing = [v for v in VARIABLES if v not in cond]
    if missing:
        problems.append(f"missing variables: {missing}")
        return problems
    if cond["buffer"] not in BUFFERS:
        problems.append(f"unknown buffer {cond['buffer']!r}; choose from {list(BUFFERS)}")
    for name in ADDITIVES:
        if cond[name] not in LEVELS:
            problems.append(f"{name} must be one of {list(LEVELS)}, got {cond[name]!r}")
    on_grid_ph = cond["pH"] in PH_VALUES or (cond["buffer"] == "DEA" and cond["pH"] == REFERENCE["pH"])
    if not on_grid_ph and not allow_off_grid:
        problems.append(f"pH {cond['pH']} is off-grid {PH_VALUES} (off-grid needs a stated reason)")
    if cond["temperature"] not in TEMPERATURES_C and not allow_off_grid:
        problems.append(f"temperature {cond['temperature']} is off-grid {TEMPERATURES_C}")
    return problems


def condition_key(cond: dict) -> tuple:
    return tuple(cond[v] for v in VARIABLES)
