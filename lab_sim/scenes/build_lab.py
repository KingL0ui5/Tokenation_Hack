"""Generate the static lab bench and compose it with the Menagerie Panda.

`build()` writes `scenes/lab.xml` — the bench ALONE (labware, stations, cameras), with
no robot. `load_model()` is the entry point everything else uses: it (re)builds lab.xml
if stale, then uses MuJoCo's MjSpec API to load the pristine Panda, inject the IK
end-effector site, and attach the arm into the bench — so `models/franka_emika_panda/`
is never edited and no `scenes/assets` symlink is needed.

Run from the repo root:  python -m scenes.build_lab   (writes lab.xml)
Load in code:            from scenes.build_lab import load_model; m = load_model()

Site names are the contract with the robot/tool layer (well_A1, reagent_pnpp, rack_1,
station_reader, pipette_grip, pipette_tip, tip_box, waste, ...). Edit the layout
constants below and re-run; never hand-edit lab.xml.

AutoBio visual meshes (CC BY-NC-SA 4.0, see models/autobio/NOTICE.md) are placed VISUAL
ONLY (group 2, contype/conaffinity 0); collisions come from our own primitives. Meshes
are registered by their measured authored bounding box so each base sits on its slot
(MuJoCo does not recentre the rendered vertices of a visual geom).

Everything here is STATIC (no free joints), so the Panda's "home" keyframe still matches.
"""

from __future__ import annotations

import math
import re
import string
from dataclasses import dataclass
from pathlib import Path

import mujoco

REPO = Path(__file__).resolve().parent.parent
PANDA_XML = REPO / "models" / "franka_emika_panda" / "panda.xml"
FRANKA_ASSETS = REPO / "models" / "franka_emika_panda" / "assets"
LAB_XML = Path(__file__).with_name("lab.xml")

# End-effector site injected into the Panda hand at load time (kept out of panda.xml so
# the vendored Menagerie model stays pristine). grasp midpoint between the fingertips;
# +45deg z-quat cancels the hand's -45deg mount.
EE_SITE = dict(name="attachment_site", pos=[0, 0, 0.1034],
               quat=[0.9238795, 0, 0, 0.3826834], group=4)

# ---------------------------------------------------------------------- bench / cameras
BENCH_CENTER = (0.40, 0.00)        # bench top centre (x, y); the top surface is z=0
BENCH_HALF = (0.55, 0.70, 0.375)   # half extents (x, y, z) of the bench box

# --------------------------------------------------------------------------- plate(s)
# Parametric: change these constants ALONE to switch plate configuration.
#   default     : PLATE_ROWS=4,  PLATE_COLS=6,  WELL_PITCH=0.018, N_PLATES=1  (current)
#   two 6x6     : PLATE_ROWS=6,  PLATE_COLS=6,  WELL_PITCH=0.018, N_PLATES=2
#   one 96-well : PLATE_ROWS=8,  PLATE_COLS=12, WELL_PITCH=0.009, N_PLATES=1
PLATE_ROWS, PLATE_COLS = 4, 6
WELL_PITCH = 0.018
N_PLATES = 1
PLATE_CENTER = (0.50, 0.00)
WELL_H = 0.012
WELL_R = WELL_PITCH * 0.38          # scales with pitch (6.8 mm at 18 mm, 3.4 mm at 9 mm)

# ----------------------------------------------------------- waste / cold block / human
# Three segregated waste bins (name suffix, rgba).
WASTE_BINS = [("aqueous", "0.20 0.45 0.85 1"), ("corrosive", "0.85 0.35 0.20 1"),
              ("solid", "0.45 0.45 0.45 1")]
WASTE_ROW_Y, WASTE_X0, WASTE_DX = -0.47, 0.24, 0.12
COLD_BLOCK_POS = (0.28, -0.24)     # chilled block beside the reagent racks, holds the enzyme
HUMAN_ZONE_POS = (0.88, 0.00)      # marked hand-off patch at the far bench edge

# ------------------------------------------------- AutoBio meshes (rel. to scenes/)
AB = "../models/autobio"
# Per mesh: authored min-z and centre (x,y) in metres, from measured raw bbox (Phase B).
# The pipette is 8 separate part meshes (MuJoCo doesn't fully load the single multi-object
# tool/pipette.obj), assembled at a common origin exactly as AutoBio's pipette.gen.xml does.
PIPETTE_PARTS = ("body", "tube", "connector", "knob", "pusher_mid",
                 "pusher_right1", "pusher_right2", "pusher_right3")
M_PIPETTE = dict(min_z=-0.008, cx=-0.00145, cy=0.0)
M_TUBE15 = dict(mesh="mesh_tube15", min_z=0.0, r=0.0084, open_z=0.1186)   # 16.8mm x 118.6mm
M_RACK = dict(parts=("pillars", "lower_plane", "upper_plane"), min_z=-0.030,
              hx=0.1025, hy=0.048, hz=0.030)                              # 205 x 96 x 60 mm
M_TIPBOX = dict(parts=("up", "low"), min_z=-0.020, cx=0.0017,
                hx=0.026, hy=0.018, hz=0.020)                            # 52 x 36 x 40 mm

# --------------------------------------------------------------- reagent racks / tubes
# 12 reagent stock tubes for the alkaline-phosphatase protocol (enzyme lives on the cold
# block, added in Phase E). (site suffix, liquid rgba).
REAGENTS = [
    ("dea", "0.80 0.90 1.00 0.6"), ("tris", "0.80 0.88 0.98 0.6"),
    ("glycine", "0.82 0.92 0.95 0.6"), ("phosphate", "0.78 0.86 1.00 0.6"),
    ("pnpp", "0.98 0.98 0.80 0.6"),
    ("mgcl2", "0.90 0.95 0.98 0.6"), ("zncl2", "0.92 0.93 0.97 0.6"),
    ("nacl", "0.95 0.95 0.98 0.6"), ("glycerol", "0.92 0.90 0.80 0.6"),
    ("water", "0.85 0.93 1.00 0.5"),
    ("naoh", "0.80 0.80 0.95 0.6"), ("pnp_standard", "0.98 0.88 0.45 0.7"),
]
RACK_A = (0.46, -0.20)             # front reagent rack centre
RACK_B = (0.46, -0.33)             # back reagent rack centre
# The rack's upper plane has small (15 mL, r=8.5 mm) and large (50 mL, r~15 mm) holes, measured
# by ray-casting the mesh (tests/test_rack_holes.py re-measures and checks these). A 16.8 mm tube
# fits SNUGLY in the 15 mL holes only: 3 rows (y = -36, 0, +36 mm) x 6 columns (x = +/-90,
# +/-54, +/-18 mm) from the rack centre. The 50 mL holes sit between rows (y = +/-18 mm).
# The lower plane is a SOLID floor (z 3..6 mm): seated tubes stand on it, not on the bench.
RACK_15ML_HOLES_X = (-0.090, -0.054, -0.018, 0.018, 0.054, 0.090)
RACK_15ML_ROWS_Y = (-0.036, 0.0, 0.036)
RACK_15ML_HOLE_R = 0.0085
RACK_FLOOR_Z = 0.006
# The empty 15 mL hole used as each rack's spare slot (rack-relative, back row; must be one of
# the measured holes above). Its nearest stocked neighbour is 36 mm away, the same pitch the
# gripper already threads between in the stocked middle row.
SPARE_HOLE = (0.018, 0.036)

TIPBOX_POS = (0.30, 0.12)
# 24-slot tip box: 6 cols x 4 rows at 8 mm pitch. Slot centres are the cells between the
# divider walls in AutoBio's tip_box.gen.xml (plate_with_box_well(.052,.036,.040,.004,.008,4,6),
# outer walls x=+/-.024 y=+/-.016, inner dividers x=+/-.016,+/-.008,0 and y=+/-.008,0) -- taken
# from the definition, not an assumed grid. See models/autobio/NOTICE.md.
TIP_COLS = (-0.020, -0.012, -0.004, 0.004, 0.012, 0.020)
TIP_ROWS = (-0.012, -0.004, 0.004, 0.012)
# Disposable 200 uL tip (AutoBio tip_200ul, visual only): 50 mm long, wide mount end at the
# authored origin (mesh z=0) tapering to a point at z=-0.050. In the box it stands wide-end up
# with its top at TIP_TOP_Z, so the nozzle descends onto the top to pick it up.
TIP_LEN = 0.050
TIP_TOP_Z = 0.052                  # world z of each in-box tip's top (and its targeting site)
TIPBOX_COLL_TOP = 0.030            # box collider top, kept below TIP_TOP_Z so pick-up isn't blocked
N_BIN_TIPS = 8                     # pre-placed hidden tips in the solid-waste bin, revealed on eject

PIPETTE_POS = (0.28, 0.30)         # the (now empty) pipette stand stays here as scenery

# Pipette mounted rigidly on the hand (fixed child body -> part of the kinematic chain).
# 180 deg about hand-x flips the authored +z (plunger) up toward the hand and sends the
# nozzle down past the fingertips; the thin 17 mm side lies along the finger slide axis.
PIPETTE_MOUNT_POS = (0.0, 0.0, 0.2573)    # hand frame; dropped so the plunger clears the palm by ~1 mm
PIPETTE_MOUNT_QUAT = (0.0, 1.0, 0.0, 0.0)  # 180 deg about hand-x
PIPETTE_NOZZLE_Z = -0.008                  # authored nozzle (pipette min-z), body-local
PIPETTE_TIP_END_Z = PIPETTE_NOZZLE_Z - TIP_LEN   # real tip end: 50 mm below the nozzle
PIPETTE_SHAFT_FROMTO = (0, 0, 0.17, 0, 0, 0.022)   # capsule: body/shaft down to ~3 cm above tip
PIPETTE_SHAFT_R = 0.006
# Handle is 16.27 mm wide along the finger-slide axis at the grip height (measured from the
# mesh); each finger sits at half that + 1 mm clearance so the pads don't clip the handle.
GRIP_HALF_M = 0.00914              # 9.14 mm each -> ~18.3 mm opening (handle 16.27 + ~1 mm/side)

GLASS = "0.90 0.95 1.00 0.25"


def rack_slots(cx: float, cy: float) -> list[tuple[float, float]]:
    """The 6 snug 15 mL hole centres (middle row) of a rack centred at (cx, cy)."""
    return [(cx + dx, cy) for dx in RACK_15ML_HOLES_X]


def rack_holes(cx: float, cy: float) -> list[tuple[float, float]]:
    """All 18 measured 15 mL hole centres of a rack centred at (cx, cy)."""
    return [(cx + dx, cy + dy) for dy in RACK_15ML_ROWS_Y for dx in RACK_15ML_HOLES_X]


def spare_hole(cx: float, cy: float) -> tuple[float, float]:
    """The rack's spare 15 mL hole, taken from the measured hole grid (never placed by hand)."""
    assert SPARE_HOLE[0] in RACK_15ML_HOLES_X and SPARE_HOLE[1] in RACK_15ML_ROWS_Y, SPARE_HOLE
    return cx + SPARE_HOLE[0], cy + SPARE_HOLE[1]


def mesh_visual(name, mesh, material, tx, ty, base_z, min_z, cx=0.0, cy=0.0) -> str:
    """Visual-only mesh geom registered so its base sits at base_z and centre at (tx,ty)."""
    return (f'    <geom name="{name}" type="mesh" mesh="{mesh}" material="{material}" '
            f'contype="0" conaffinity="0" group="2" '
            f'pos="{tx - cx:.4f} {ty - cy:.4f} {base_z - min_z:.4f}"/>\n')


def well(slot, x, y, z0, liquid_rgba) -> str:
    """Primitive well/tube: translucent cylinder + liquid geom + target site."""
    return (
        f'    <geom name="vessel_{slot}" type="cylinder" size="{WELL_R:.4f} {WELL_H / 2:.4f}" '
        f'pos="{x:.4f} {y:.4f} {z0 + WELL_H / 2:.4f}" rgba="{GLASS}" contype="0" conaffinity="0" group="1"/>\n'
        f'    <geom name="liquid_{slot}" type="cylinder" size="{WELL_R * 0.85:.4f} 0.0001" '
        f'pos="{x:.4f} {y:.4f} {z0 + 0.0001:.4f}" rgba="{liquid_rgba}" contype="0" conaffinity="0" group="1"/>\n'
        f'    <site name="{slot}" pos="{x:.4f} {y:.4f} {z0 + WELL_H + 0.01:.4f}" size="0.003" rgba="1 0 0 0.5" group="4"/>\n'
    )


def plate(prefix, cx, cy) -> str:
    """One well plate (PLATE_ROWS x PLATE_COLS) centred at (cx, cy); wells named
    <prefix>well_<row><col>. prefix is "" for a single plate, "p1_"/"p2_" for several."""
    rows = string.ascii_uppercase[:PLATE_ROWS]
    pw, pd, base_h = PLATE_COLS * WELL_PITCH + 0.01, PLATE_ROWS * WELL_PITCH + 0.01, 0.004
    # FREE body: the plate can be moved/knocked. Geoms are relative to the body at (cx, cy).
    s = (f'    <body name="{prefix}plate" pos="{cx:.4f} {cy:.4f} 0">\n'
         f'      <freejoint/>\n'
         f'      <inertial pos="0 0 0.008" mass="0.05" diaginertia="1e-4 1e-4 1e-4"/>\n'
         f'      <geom name="{prefix}plate_base" type="box" size="{pw / 2:.4f} {pd / 2:.4f} {base_h / 2:.4f}" '
         f'pos="0 0 {base_h / 2:.4f}" rgba="0.95 0.95 0.95 1"/>\n'
         f'      <geom name="{prefix}plate_collision" type="box" size="{pw / 2:.4f} {pd / 2:.4f} {WELL_H / 2:.4f}" '
         f'pos="0 0 {base_h + WELL_H / 2:.4f}" rgba="0 0 0 0" group="3"/>\n')
    for i, r in enumerate(rows):
        for j in range(PLATE_COLS):
            x = (i - (PLATE_ROWS - 1) / 2) * WELL_PITCH          # relative to the plate body
            y = (j - (PLATE_COLS - 1) / 2) * WELL_PITCH
            s += "      " + well(f"{prefix}well_{r}{j + 1}", x, y, base_h, "1 1 0.6 0.9").lstrip()
    s += "    </body>\n"
    return s


def plate_layout() -> list[tuple[str, float, float]]:
    """(prefix, cx, cy) for each plate. Single plate keeps the bare `well_` naming."""
    cx0, cy0 = PLATE_CENTER
    if N_PLATES == 1:
        return [("", cx0, cy0)]
    spacing = PLATE_COLS * WELL_PITCH + 0.03
    return [(f"p{k + 1}_", cx0 + (k - (N_PLATES - 1) / 2) * spacing, cy0) for k in range(N_PLATES)]


def reagent_tube(name, x, y, base_z, liquid_rgba) -> str:
    """15 mL stock tube as a FREE body: visual mesh + a flat-bottomed cylinder collider
    (stands stably like a can) + liquid geom + opening site. Graspable/movable."""
    h, cr = M_TUBE15["open_z"], M_TUBE15["r"]
    return (
        f'    <body name="tubebody_{name}" pos="{x:.4f} {y:.4f} {base_z:.4f}">\n'
        f'      <freejoint/>\n'
        f'      <inertial pos="0 0 0.02" mass="0.012" diaginertia="2e-5 2e-5 4e-6"/>\n'
        f'      <geom name="tube_{name}" type="mesh" mesh="mesh_tube15" material="mat_tube" contype="0" conaffinity="0" group="2"/>\n'
        f'      <geom name="collide_tube_{name}" type="cylinder" size="{cr:.4f} {h / 2:.4f}" pos="0 0 {h / 2:.4f}" rgba="0 0 0 0" group="3"/>\n'
        f'      <geom name="liquid_reagent_{name}" type="cylinder" size="0.0068 0.0001" pos="0 0 0.0001" rgba="{liquid_rgba}" contype="0" conaffinity="0" group="1"/>\n'
        f'      <site name="reagent_{name}" pos="0 0 {h + 0.01:.4f}" size="0.003" rgba="1 0 0 0.5" group="4"/>\n'
        f'      <site name="tube_grip_{name}" pos="0 0 0.1100" size="0.003" rgba="0 0 1 0.5" group="4"/>\n'
        f'    </body>\n'
    )


def reagent_rack(name, cx, cy) -> str:
    """10-slot rack: 3 visual mesh parts + perimeter-wall colliders (open interior). Tubes are
    snapped+world-welded into holes, so they don't physically settle against the walls."""
    hx, hy, hz = M_RACK["hx"], M_RACK["hy"], M_RACK["hz"]
    s = "".join(mesh_visual(f"{name}_{p}", f"mesh_rack_{p}", "mat_rack", cx, cy, 0.0, M_RACK["min_z"])
                for p in M_RACK["parts"])
    for i, (sx, sy, dx, dy) in enumerate([(hx, 0.002, 0, hy), (hx, 0.002, 0, -hy),
                                          (0.002, hy, hx, 0), (0.002, hy, -hx, 0)]):
        s += (f'    <geom name="collide_{name}_{i}" type="box" size="{sx:.4f} {sy:.4f} {hz:.4f}" '
              f'pos="{cx + dx:.4f} {cy + dy:.4f} {hz:.4f}" rgba="0 0 0 0" group="3"/>\n')
    return s


def build() -> str:
    parts: list[str] = []
    px, py = PLATE_CENTER
    (bx, by), (bhx, bhy, bhz) = BENCH_CENTER, BENCH_HALF

    # well plate(s) — parametric (rows/cols/pitch/count via constants above)
    for prefix, cx, cy in plate_layout():
        parts.append(plate(prefix, cx, cy))

    # reagent racks + stock tubes (replace the old reservoirs)
    parts.append(reagent_rack("rackA", *RACK_A))
    parts.append(reagent_rack("rackB", *RACK_B))
    slots = rack_slots(*RACK_A) + rack_slots(*RACK_B)
    for (name, rgba), (sx, sy) in zip(REAGENTS, slots):
        parts.append(reagent_tube(name, sx, sy, RACK_FLOOR_Z, rgba))
    # an empty measured 15 mL hole per rack to place a tube into. Like a tube's reagent_ site,
    # the site marks where a tube seated on the rack floor has its opening site.
    for site, rack in (("spare_hole", RACK_A), ("spare_hole_b", RACK_B)):
        hx, hy = spare_hole(*rack)
        parts.append(f'    <site name="{site}" pos="{hx:.4f} {hy:.4f} {RACK_FLOOR_Z + M_TUBE15["open_z"] + 0.01:.4f}" '
                     f'size="0.004" rgba="0 1 0 0.5" group="4"/>\n')

    # tip box (visual mesh placed by authored origin so the slot grid aligns) + 24 tips
    tbx, tby = TIPBOX_POS
    parts.append(mesh_visual("tipbox_up", "mesh_tipbox_up", "mat_tipbox", tbx, tby, 0.0, M_TIPBOX["min_z"]))
    parts.append(mesh_visual("tipbox_low", "mesh_tipbox_low", "mat_tipbox", tbx, tby, 0.0, M_TIPBOX["min_z"]))
    # The box collider's top is lowered to TIPBOX_COLL_TOP (below the tips' grab height at
    # TIP_TOP_Z) so the nozzle can descend onto a tip top without the box repelling the arm
    # during pick-up. It still protects the box body against sideways arm collisions.
    coll_hz = TIPBOX_COLL_TOP / 2
    parts.append(
        f'    <geom name="collide_tipbox" type="box" size="{M_TIPBOX["hx"]:.4f} {M_TIPBOX["hy"]:.4f} {coll_hz:.4f}" '
        f'pos="{tbx:.4f} {tby:.4f} {coll_hz:.4f}" rgba="0 0 0 0" group="3"/>\n'
        f'    <site name="tip_box" pos="{tbx:.4f} {tby:.4f} {2 * M_TIPBOX["hz"] + 0.025:.4f}" '
        f'size="0.004" rgba="0 1 0 0.5" group="4"/>\n'
    )
    # 24 visual tips (tip_00..tip_23), AutoBio 200 uL mesh, wide end up, top at TIP_TOP_Z, with a
    # targeting site at each top (tip_site_00..). pick_up_tip() hides a tip by setting its alpha 0.
    i = 0
    for ry in TIP_ROWS:
        for cxx in TIP_COLS:
            tx, ty = tbx + cxx, tby + ry
            parts.append(
                f'    <geom name="tip_{i:02d}" type="mesh" mesh="mesh_tip" '
                f'pos="{tx:.4f} {ty:.4f} {TIP_TOP_Z:.4f}" rgba="0.95 0.95 0.80 1" '
                f'contype="0" conaffinity="0" group="2"/>\n'
                f'    <site name="tip_site_{i:02d}" pos="{tx:.4f} {ty:.4f} {TIP_TOP_Z:.4f}" '
                f'size="0.003" rgba="1 0.6 0 0.6" group="4"/>\n')
            i += 1

    # (old 4-tube dilution rack removed; dilutions can use spare 15 mL holes later)

    # stations, pipette (vendored mesh), waste
    ppx, ppy = PIPETTE_POS
    parts.append(
        '    <geom name="plate_reader" type="box" size="0.10 0.08 0.05" pos="0.68 0.30 0.05" rgba="0.25 0.25 0.28 1"/>\n'
        '    <site name="station_reader" pos="0.62 0.30 0.12" size="0.004" rgba="0 1 0 0.5" group="4"/>\n'
        '    <geom name="incubator" type="box" size="0.08 0.08 0.05" pos="0.74 -0.05 0.05" rgba="0.85 0.55 0.30 1"/>\n'
        '    <site name="station_incubator" pos="0.66 -0.05 0.12" size="0.004" rgba="0 1 0 0.5" group="4"/>\n'
        f'    <site name="station_bench" pos="{px} {py} 0.03" size="0.004" rgba="0 1 0 0.5" group="4"/>\n'
        # open pipette stand: base plate + two flanking posts; the pipette rests vertically
        # between the posts with its tip on the base, so the whole pipette is visible.
        # Empty pipette stand (scenery). The pipette itself is now mounted on the hand by
        # load_model(), not placed here.
        f'    <geom name="pip_stand_base" type="box" size="0.022 0.028 0.004" pos="{ppx} {ppy} 0.004" rgba="0.45 0.45 0.50 1" contype="0" conaffinity="0"/>\n'
        f'    <geom name="pip_stand_post1" type="box" size="0.004 0.004 0.090" pos="{ppx} {ppy - 0.015:.4f} 0.0940" rgba="0.45 0.45 0.50 1" contype="0" conaffinity="0"/>\n'
        f'    <geom name="pip_stand_post2" type="box" size="0.004 0.004 0.090" pos="{ppx} {ppy + 0.015:.4f} 0.0940" rgba="0.45 0.45 0.50 1" contype="0" conaffinity="0"/>\n'
    )

    # three segregated waste bins (aqueous / corrosive / solid)
    for k, (wname, rgba) in enumerate(WASTE_BINS):
        wx = WASTE_X0 + k * WASTE_DX
        parts.append(
            f'    <geom name="bin_{wname}" type="cylinder" size="0.035 0.05" pos="{wx:.4f} {WASTE_ROW_Y} 0.05" rgba="{rgba}"/>\n'
            f'    <site name="waste_{wname}" pos="{wx:.4f} {WASTE_ROW_Y} 0.12" size="0.004" rgba="0 1 0 0.5" group="4"/>\n'
        )
    # Pre-placed ejected tips in the SOLID bin, hidden at start (alpha 0). eject_tip() reveals one
    # per call so discarded tips visibly accumulate. Visual only; deterministic ring of offsets.
    solid_x = WASTE_X0 + (len(WASTE_BINS) - 1) * WASTE_DX
    for t in range(N_BIN_TIPS):
        ang = 2 * math.pi * t / N_BIN_TIPS
        ox, oy = 0.018 * math.cos(ang), 0.018 * math.sin(ang)
        parts.append(
            f'    <geom name="bintip_{t}" type="mesh" mesh="mesh_tip" '
            f'pos="{solid_x + ox:.4f} {WASTE_ROW_Y + oy:.4f} 0.0950" '
            f'rgba="0.95 0.95 0.80 0" contype="0" conaffinity="0" group="2"/>\n')

    # cold block (chilled) holding the enzyme tube
    cbx, cby = COLD_BLOCK_POS
    parts.append(f'    <geom name="cold_block" type="box" size="0.05 0.05 0.015" pos="{cbx} {cby} 0.015" rgba="0.55 0.80 0.90 1"/>\n')
    parts.append(reagent_tube("enzyme", cbx, cby, 0.03, "0.85 0.90 0.80 0.7"))

    # human hand-off zone, marked patch at the far bench edge (outside the robot workspace)
    hzx, hzy = HUMAN_ZONE_POS
    parts.append(
        f'    <geom name="human_zone" type="box" size="0.06 0.08 0.001" pos="{hzx} {hzy} 0.001" rgba="0.95 0.85 0.10 1" contype="0" conaffinity="0"/>\n'
        f'    <site name="human_zone" pos="{hzx} {hzy} 0.02" size="0.006" rgba="0 1 0 0.5" group="4"/>\n'
    )

    # Stand pipette: a second, upright pipette in the stand, HIDDEN at start (alpha 0, no
    # collision). put_down_pipette()/pick_up_pipette() swap visibility+collision between this
    # and the hand-mounted pipette. Its pose is overwritten to match the held pipette on put-down.
    cx = M_PIPETTE["cx"]
    sz = 0.048                                   # upright default: nozzle ~z=0.04 in the stand
    for p in PIPETTE_PARTS:
        parts.append(f'    <geom name="stand_pipette_{p}" type="mesh" mesh="mesh_pip_{p}" '
                     f'rgba="0.85 0.85 0.88 0" contype="0" conaffinity="0" group="2" '
                     f'pos="{ppx - cx:.4f} {ppy:.4f} {sz:.4f}"/>\n')
    parts.append(
        f'    <geom name="stand_pipette_shaft" type="capsule" fromto="{ppx} {ppy} {sz + 0.022:.4f} {ppx} {ppy} {sz + 0.17:.4f}" '
        f'size="0.006" group="3" rgba="1 0.5 0 0" contype="0" conaffinity="0"/>\n'
        f'    <site name="pipette_stand" pos="{ppx} {ppy} 0.04" size="0.004" rgba="0 0 1 0.5" group="4"/>\n'
    )

    labware = "".join(parts)
    pipette_meshes = "\n    ".join(
        f'<mesh name="mesh_pip_{p}" file="{AB}/tool/pipette/{p}_visual.obj" scale="0.1 0.1 0.1"/>'
        for p in PIPETTE_PARTS)
    return f"""<!-- GENERATED by scenes/build_lab.py. Edit that file and re-run, not this one.
     This is the BENCH ONLY (no robot). The Panda is attached at load time by
     load_model(); load lab.xml through that, not directly. -->
<mujoco model="agentic_lab">
  <option integrator="implicitfast"/>

  <statistic center="0.45 0 0.2" extent="1.0"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="150" elevation="-25" offwidth="1280" offheight="960"/>
  </visual>

  <asset>
    <texture name="lab_sky" type="skybox" builtin="gradient" rgb1="0.35 0.45 0.55" rgb2="0.05 0.05 0.08" width="512" height="3072"/>
    <texture name="lab_floor" type="2d" builtin="checker" mark="edge" rgb1="0.25 0.27 0.30" rgb2="0.20 0.22 0.25" markrgb="0.6 0.6 0.6" width="300" height="300"/>
    <material name="lab_floor" texture="lab_floor" texuniform="true" texrepeat="5 5" reflectance="0.1"/>
    <material name="bench_top" rgba="0.82 0.84 0.86 1"/>

    <!-- AutoBio vendored visual meshes (CC BY-NC-SA 4.0; see models/autobio/NOTICE.md) -->
    {pipette_meshes}
    <mesh name="mesh_tube15" file="{AB}/container/centrifuge_15ml_body.STL" scale="0.001 0.001 0.001"/>
    <mesh name="mesh_rack_pillars" file="{AB}/rack/centrifuge_10slot/pillars.obj" scale="0.001 0.001 0.001"/>
    <mesh name="mesh_rack_lower_plane" file="{AB}/rack/centrifuge_10slot/lower_plane.obj" scale="0.001 0.001 0.001"/>
    <mesh name="mesh_rack_upper_plane" file="{AB}/rack/centrifuge_10slot/upper_plane.obj" scale="0.001 0.001 0.001"/>
    <mesh name="mesh_tipbox_up" file="{AB}/rack/tip_box_24slot/up.obj" scale="0.001 0.001 0.001"/>
    <mesh name="mesh_tipbox_low" file="{AB}/rack/tip_box_24slot/low.obj" scale="0.001 0.001 0.001"/>
    <mesh name="mesh_tip" file="{AB}/container/tip_200ul.obj" scale="0.001 0.001 0.001"/>
    <material name="mat_pipette" rgba="0.85 0.85 0.88 1"/>
    <material name="mat_tube" rgba="0.80 0.90 1.0 0.45"/>
    <material name="mat_rack" rgba="0.35 0.42 0.55 1"/>
    <material name="mat_tipbox" rgba="0.30 0.55 0.85 1"/>
  </asset>

  <worldbody>
    <light pos="0.4 0 1.5" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="0 0 0.05" pos="0 0 -0.75" material="lab_floor"/>
    <geom name="bench" type="box" size="{bhx} {bhy} {bhz}" pos="{bx} {by} {-bhz}" material="bench_top"/>

    <camera name="front" pos="1.60 0 0.55" xyaxes="0 1 0 -0.423 0 0.906"/>
    <camera name="side" pos="0.45 -1.30 0.70" xyaxes="1 0 0 0 0.5 0.866"/>
    <camera name="plate_top" pos="{px} {py} 0.45" xyaxes="0 -1 0 1 0 0"/>
    <camera name="racks" pos="1.05 -0.26 0.42" xyaxes="0 1 0 -0.5 0 0.866"/>

{labware}  </worldbody>
</mujoco>
"""


def write_lab_xml() -> Path:
    LAB_XML.write_text(build())
    return LAB_XML


def _stale() -> bool:
    """lab.xml needs rebuilding if it's missing or older than this generator."""
    return (not LAB_XML.exists()
            or LAB_XML.stat().st_mtime < Path(__file__).stat().st_mtime)


def load_model() -> mujoco.MjModel:
    """Build the full lab model: bench (lab.xml) + Panda, with the EE site injected.

    Rebuilds lab.xml if stale. Uses MjSpec so the Panda's meshes resolve via an absolute
    meshdir (no symlink) and the end-effector site is added without touching panda.xml.
    """
    if _stale():
        write_lab_xml()

    bench = mujoco.MjSpec.from_file(str(LAB_XML))

    panda = mujoco.MjSpec.from_file(str(PANDA_XML))
    panda.meshdir = str(FRANKA_ASSETS)          # absolute -> resolves without the old symlink
    hand = panda.body("hand")
    site = hand.add_site()
    site.name, site.pos, site.quat, site.group = (
        EE_SITE["name"], EE_SITE["pos"], EE_SITE["quat"], EE_SITE["group"])

    # Attach the arm (link0 subtree) into the bench; "" prefixes keep every name intact.
    frame = bench.worldbody.add_frame()
    frame.attach_body(panda.body("link0"), "", "")

    _mount_pipette(bench)
    _add_grasp_welds(bench)
    model = bench.compile()
    _set_home_keyframe(model)
    return model


def _add_grasp_welds(bench: mujoco.MjSpec) -> None:
    """One inactive weld per graspable free body (tubes + plate). grasp() sets the weld's
    relative pose to the current hand<->object pose and activates it; release()/break break it."""
    bodies = [f"tubebody_{r[0]}" for r in REAGENTS] + ["tubebody_enzyme"]
    bodies += [f"{prefix}plate" for prefix, _, _ in plate_layout()]
    for b in bodies:
        eq = bench.add_equality()                 # hand <-> object: for carrying
        eq.type, eq.objtype = mujoco.mjtEq.mjEQ_WELD, mujoco.mjtObj.mjOBJ_BODY
        eq.name, eq.name1, eq.name2, eq.active = f"weld_{b}", "hand", b, False
        we = bench.add_equality()                 # world <-> object: holds it in its slot
        we.type, we.objtype = mujoco.mjtEq.mjEQ_WELD, mujoco.mjtObj.mjOBJ_BODY
        we.name, we.name1, we.name2, we.active = f"worldweld_{b}", "world", b, False


HOME_ARM = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853]   # Franka Panda home joint angles


def _set_home_keyframe(model: mujoco.MjModel) -> None:
    """Rebuild the home keyframe robustly: free-joint bodies (plate, tubes) at their rest
    poses (qpos0), the arm at its home angles, and the fingers closed onto the pipette.
    Addressed by joint/actuator name because free joints shift qpos/ctrl indices."""
    model.key_qpos[0][:] = model.qpos0                 # everything at rest (free bodies seated)
    for i, a in enumerate(HOME_ARM):
        model.key_qpos[0][model.jnt_qposadr[model.joint(f"joint{i + 1}").id]] = a
    for fj in ("finger_joint1", "finger_joint2"):
        model.key_qpos[0][model.jnt_qposadr[model.joint(fj).id]] = GRIP_HALF_M
    model.key_ctrl[0][:] = 0
    for i, a in enumerate(HOME_ARM):
        model.key_ctrl[0][model.actuator(f"actuator{i + 1}").id] = a
    model.key_ctrl[0][model.actuator("actuator8").id] = GRIP_HALF_M / 0.04 * 255


def _mount_pipette(bench: mujoco.MjSpec) -> None:
    """Mount the pipette as a fixed child of the hand: 8 visual parts + a shaft capsule
    collider + the pipette_nozzle / pipette_tip_end sites, and close the fingers onto it."""
    pip = bench.body("hand").add_body()
    pip.name = "pipette"
    pip.pos = list(PIPETTE_MOUNT_POS)
    pip.quat = list(PIPETTE_MOUNT_QUAT)
    for p in PIPETTE_PARTS:
        g = pip.add_geom()
        g.type = mujoco.mjtGeom.mjGEOM_MESH
        g.meshname = f"mesh_pip_{p}"
        g.rgba = [0.85, 0.85, 0.88, 1]        # rgba (not material) so skills can hide it
        g.contype, g.conaffinity, g.group = 0, 0, 2
    shaft = pip.add_geom()
    shaft.name = "pipette_shaft"
    shaft.type = mujoco.mjtGeom.mjGEOM_CAPSULE
    shaft.fromto = list(PIPETTE_SHAFT_FROMTO)
    shaft.size = [PIPETTE_SHAFT_R, 0, 0]
    shaft.group = 3
    shaft.rgba = [1, 0.5, 0, 0.0]              # invisible collider
    # The shaft capsule stops at z=0.022 (body frame), ~3 cm above the nozzle, so the whole
    # disposable-tip region below it is already collider-free; the tip carries its own collider.
    for nm, z in (("pipette_nozzle", PIPETTE_NOZZLE_Z), ("pipette_tip_end", PIPETTE_TIP_END_Z)):
        s = pip.add_site()
        s.name, s.pos, s.size, s.group, s.rgba = nm, [0, 0, z], [0.004, 0, 0], 4, [0, 0, 1, 0.8]

    # Disposable tip mounted on the nozzle: a fixed child body of the pipette, HIDDEN and
    # non-colliding at start (alpha 0, contype/conaffinity 0). pick_up_tip()/eject_tip() toggle
    # its visibility and collider. Tip top sits exactly at the nozzle; it extends 50 mm down to
    # pipette_tip_end. When active, its slim capsule collider lets the arm avoid obstacles with
    # the tip included in travel height, while contact-excludes below let the tip enter vessels.
    tipb = pip.add_body()
    tipb.name = "tip_mounted"
    tv = tipb.add_geom()
    tv.name = "tip_mounted_visual"
    tv.type = mujoco.mjtGeom.mjGEOM_MESH
    tv.meshname = "mesh_tip"
    tv.pos = [0, 0, PIPETTE_NOZZLE_Z]         # mesh origin (wide top) at the nozzle
    tv.rgba = [0.95, 0.95, 0.80, 0.0]         # hidden until a tip is picked up
    tv.contype, tv.conaffinity, tv.group = 0, 0, 2
    tc = tipb.add_geom()
    tc.name = "tip_mounted_collision"
    tc.type = mujoco.mjtGeom.mjGEOM_CAPSULE
    tc.fromto = [0, 0, PIPETTE_NOZZLE_Z, 0, 0, PIPETTE_TIP_END_Z]
    tc.size = [0.002, 0, 0]
    tc.group = 3
    tc.rgba = [1, 0.5, 0, 0.0]
    tc.contype, tc.conaffinity = 0, 0         # activated by pick_up_tip()

    # don't compute finger<->pipette / finger<->tip contacts (all siblings under hand)
    for other in ("hand", "left_finger", "right_finger"):
        for body in ("pipette", "tip_mounted"):
            ex = bench.add_exclude()
            ex.bodyname1, ex.bodyname2 = body, other
    # The mounted tip must enter vessels (tubes + wells/plate) freely -> exclude those contacts
    # ONLY (not every obstacle, so the tip is still avoided against bench/racks/stations).
    vessel_bodies = [f"tubebody_{r[0]}" for r in REAGENTS] + ["tubebody_enzyme"]
    vessel_bodies += [f"{prefix}plate" for prefix, _, _ in plate_layout()]
    for body in vessel_bodies:
        ex = bench.add_exclude()
        ex.bodyname1, ex.bodyname2 = "tip_mounted", body


@dataclass
class VesselSpec:
    radius_m: float
    height_m: float
    capacity_ul: float


@dataclass
class SceneContract:
    """What the scene publishes to the tool layer, derived from the compiled model + the
    geometry constants above (not duplicated). See scene_contract()."""
    wells: list[str]                      # well site names ("well_A1" or "p1_well_A1")
    reagents: dict[str, str]              # reagent name -> source site ("enzyme" -> "reagent_enzyme")
    waste_bins: dict[str, dict]           # stream -> {"site": "waste_solid", "geom": "bin_solid"}
    stations: dict[str, str]              # name -> site (reader, incubator, bench, tip_box, ...)
    tip_slots: list[str]                  # per-tip targeting sites in the box ("tip_site_00"..)
    tip_points: dict[str, str]            # active IK points on the held pipette: name -> site
    obstacles: list[str]                  # collidable geom names the arm must avoid (no robot/floor)
    vessels: dict[str, VesselSpec]        # "well" / "tube" -> dimensions

    def liquid_geom(self, container_site: str) -> str:
        """Scene convention: the visual liquid geom for a container site."""
        if container_site.startswith("reagent_"):
            return "liquid_reagent_" + container_site[len("reagent_"):]
        return "liquid_" + container_site       # well_B3 -> liquid_well_B3


_ROBOT_BODIES = {"link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7",
                 "hand", "left_finger", "right_finger", "pipette",   # pipette is held by the arm
                 "tip_mounted"}   # disposable tip rides on the nozzle (never an obstacle)


def scene_contract(model: mujoco.MjModel | None = None) -> SceneContract:
    """Publish the scene's contract, derived from what build() actually emits.

    Everything is read from the compiled model (site/geom names, collidability) and the
    geometry constants, so it can never drift from the scene the robot loads.
    """
    m = model or load_model()
    sites = [m.site(i).name for i in range(m.nsite)]

    wells = sorted((s for s in sites if s.startswith("well_") or re.match(r"p\d+_well_", s)),
                   key=lambda s: (s.split("well_")[0], s.split("well_")[1][0], int(s.split("well_")[1][1:])))
    reagents = {s[len("reagent_"):]: s for s in sites if s.startswith("reagent_")}
    waste_bins = {s[len("waste_"):]: {"site": s, "geom": "bin_" + s[len("waste_"):]}
                  for s in sites if s.startswith("waste_")}
    tip_points = {s[len("pipette_"):]: s for s in ("pipette_nozzle", "pipette_tip_end") if s in sites}
    tip_slots = sorted(s for s in sites if s.startswith("tip_site_"))
    classified = (set(wells) | set(reagents.values()) | {v["site"] for v in waste_bins.values()}
                  | set(tip_points.values()) | set(tip_slots))
    stations = {s: s for s in sites if s not in classified and s != EE_SITE["name"]}

    def movable(body_id):   # geoms on a free-jointed body (tubes, plate) are grasp targets
        b = body_id
        while b != 0:
            adr, n = m.body_jntadr[b], m.body_jntnum[b]
            if any(m.jnt_type[adr + k] == mujoco.mjtJoint.mjJNT_FREE for k in range(n)):
                return True
            b = m.body_parentid[b]
        return False

    obstacles = []
    for g in range(m.ngeom):
        name = m.geom(g).name
        if (m.geom_contype[g] == 0 or not name or name == "floor"
                or m.body(m.geom_bodyid[g]).name in _ROBOT_BODIES or movable(m.geom_bodyid[g])):
            continue
        obstacles.append(name)

    def cap(r, h):
        return math.pi * r * r * h * 1e9      # m^3 -> uL
    vessels = {"well": VesselSpec(WELL_R, WELL_H, cap(WELL_R, WELL_H)),
               "tube": VesselSpec(M_TUBE15["r"], M_TUBE15["open_z"], cap(M_TUBE15["r"], M_TUBE15["open_z"]))}

    return SceneContract(wells, reagents, waste_bins, stations, tip_slots, tip_points, obstacles, vessels)


if __name__ == "__main__":
    out = write_lab_xml()
    model = load_model()   # validate the full compose on build
    print(f"wrote {out}  (bench + Panda compiles: {model.nbody} bodies, "
          f"{model.nsite} sites, {model.ngeom} geoms, {model.nu} actuators)")
