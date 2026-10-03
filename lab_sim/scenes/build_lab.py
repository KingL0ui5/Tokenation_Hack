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

# ---------------------------------------------------------------------------- plate
ROWS, COLS = "ABCD", 6
PLATE_CENTER = (0.50, 0.00)
WELL_PITCH = 0.018
WELL_R, WELL_H = 0.007, 0.012

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
# The 10-slot rack has small (15 mL, r~8.5 mm) holes and large (50 mL, r~15 mm) holes.
# A 16.8 mm tube fits SNUGLY in the 15 mL holes only; these are the 6 in the middle (y=0)
# row, at x = +/-90, +/-54, +/-18 mm from the rack centre (measured by ray-casting the mesh).
RACK_15ML_HOLES_X = (-0.090, -0.054, -0.018, 0.018, 0.054, 0.090)

TIPBOX_POS = (0.30, 0.12)
# 24-slot tip box: 6 cols x 4 rows at 8 mm pitch (from tip_box.gen.xml dividers).
TIP_COLS = (-0.020, -0.012, -0.004, 0.004, 0.012, 0.020)
TIP_ROWS = (-0.012, -0.004, 0.004, 0.012)

PIPETTE_POS = (0.28, 0.30)         # standing in an open stand
PIPETTE_TIP_Z = 0.008             # tip rests on the stand base plate

GLASS = "0.90 0.95 1.00 0.25"


def rack_slots(cx: float, cy: float) -> list[tuple[float, float]]:
    """The 6 snug 15 mL hole centres (middle row) of a rack centred at (cx, cy)."""
    return [(cx + dx, cy) for dx in RACK_15ML_HOLES_X]


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


def reagent_tube(name, x, y, base_z, liquid_rgba) -> str:
    """15 mL stock tube: visual mesh (no collider) + liquid geom + opening site."""
    h = M_TUBE15["open_z"]
    return (
        mesh_visual(f"tube_{name}", "mesh_tube15", "mat_tube", x, y, base_z, M_TUBE15["min_z"])
        + f'    <geom name="liquid_reagent_{name}" type="cylinder" size="0.0070 0.0001" '
          f'pos="{x:.4f} {y:.4f} {base_z + 0.0001:.4f}" rgba="{liquid_rgba}" contype="0" conaffinity="0" group="1"/>\n'
        + f'    <site name="reagent_{name}" pos="{x:.4f} {y:.4f} {base_z + h + 0.01:.4f}" '
          f'size="0.003" rgba="1 0 0 0.5" group="4"/>\n'
    )


def reagent_rack(name, cx, cy) -> str:
    """10-slot rack: 3 visual mesh parts + one solid box collider (tubes nest inside)."""
    s = "".join(mesh_visual(f"{name}_{p}", f"mesh_rack_{p}", "mat_rack", cx, cy, 0.0, M_RACK["min_z"])
                for p in M_RACK["parts"])
    s += (f'    <geom name="collide_{name}" type="box" '
          f'size="{M_RACK["hx"]:.4f} {M_RACK["hy"]:.4f} {M_RACK["hz"]:.4f}" '
          f'pos="{cx:.4f} {cy:.4f} {M_RACK["hz"]:.4f}" rgba="0 0 0 0" group="3"/>\n')
    return s


def build() -> str:
    parts: list[str] = []
    px, py = PLATE_CENTER
    plate_w = COLS * WELL_PITCH + 0.01
    plate_d = len(ROWS) * WELL_PITCH + 0.05
    base_h = 0.004

    # plate: thin base, invisible collision block over the wells, then wells
    parts.append(
        f'    <geom name="plate_base" type="box" size="{plate_w / 2:.4f} {plate_d / 2:.4f} {base_h / 2:.4f}" '
        f'pos="{px} {py} {base_h / 2:.4f}" rgba="0.95 0.95 0.95 1"/>\n'
        f'    <geom name="plate_collision" type="box" size="{plate_w / 2:.4f} {plate_d / 2:.4f} {WELL_H / 2:.4f}" '
        f'pos="{px} {py} {base_h + WELL_H / 2:.4f}" rgba="0 0 0 0" group="3"/>\n'
    )
    for i, r in enumerate(ROWS):
        for j in range(COLS):
            x = px + (i - (len(ROWS) - 1) / 2) * WELL_PITCH
            y = py + (j - (COLS - 1) / 2) * WELL_PITCH
            parts.append(well(f"well_{r}{j + 1}", x, y, base_h, "1 1 0.6 0.9"))

    # reagent racks + stock tubes (replace the old reservoirs)
    parts.append(reagent_rack("rackA", *RACK_A))
    parts.append(reagent_rack("rackB", *RACK_B))
    slots = rack_slots(*RACK_A) + rack_slots(*RACK_B)
    for (name, rgba), (sx, sy) in zip(REAGENTS, slots):
        parts.append(reagent_tube(name, sx, sy, 0.0, rgba))

    # tip box (visual mesh placed by authored origin so the slot grid aligns) + 24 tips
    tbx, tby = TIPBOX_POS
    parts.append(mesh_visual("tipbox_up", "mesh_tipbox_up", "mat_tipbox", tbx, tby, 0.0, M_TIPBOX["min_z"]))
    parts.append(mesh_visual("tipbox_low", "mesh_tipbox_low", "mat_tipbox", tbx, tby, 0.0, M_TIPBOX["min_z"]))
    parts.append(
        f'    <geom name="collide_tipbox" type="box" size="{M_TIPBOX["hx"]:.4f} {M_TIPBOX["hy"]:.4f} {M_TIPBOX["hz"]:.4f}" '
        f'pos="{tbx:.4f} {tby:.4f} {M_TIPBOX["hz"]:.4f}" rgba="0 0 0 0" group="3"/>\n'
        f'    <site name="tip_box" pos="{tbx:.4f} {tby:.4f} {2 * M_TIPBOX["hz"] + 0.025:.4f}" '
        f'size="0.004" rgba="0 1 0 0.5" group="4"/>\n'
    )
    # 24 visual tips (tip_00..tip_23); the agent/UI hides a tip by setting its rgba alpha 0.
    i = 0
    for ry in TIP_ROWS:
        for cxx in TIP_COLS:
            parts.append(
                f'    <geom name="tip_{i:02d}" type="cylinder" size="0.0022 0.016" '
                f'pos="{tbx + cxx:.4f} {tby + ry:.4f} 0.0420" rgba="0.95 0.95 0.80 0.95" '
                f'contype="0" conaffinity="0" group="2"/>\n')
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
        f'    <geom name="pip_stand_base" type="box" size="0.022 0.028 0.004" pos="{ppx} {ppy} 0.004" rgba="0.45 0.45 0.50 1" contype="0" conaffinity="0"/>\n'
        f'    <geom name="pip_stand_post1" type="box" size="0.004 0.004 0.090" pos="{ppx} {ppy - 0.015:.4f} 0.0940" rgba="0.45 0.45 0.50 1" contype="0" conaffinity="0"/>\n'
        f'    <geom name="pip_stand_post2" type="box" size="0.004 0.004 0.090" pos="{ppx} {ppy + 0.015:.4f} 0.0940" rgba="0.45 0.45 0.50 1" contype="0" conaffinity="0"/>\n'
        + "".join(mesh_visual(f"pipette_{p}", f"mesh_pip_{p}", "mat_pipette", ppx, ppy,
                              PIPETTE_TIP_Z, M_PIPETTE["min_z"], M_PIPETTE["cx"], M_PIPETTE["cy"])
                  for p in PIPETTE_PARTS)
        + f'    <site name="pipette_grip" pos="{ppx} {ppy} 0.11" size="0.004" rgba="0 0 1 0.5" group="4"/>\n'
        f'    <site name="pipette_tip" pos="{ppx} {ppy} {PIPETTE_TIP_Z:.4f}" size="0.003" rgba="0 0 1 0.5" group="4"/>\n'
        '    <geom name="waste_bin" type="cylinder" size="0.05 0.05" pos="0.30 -0.45 0.05" rgba="0.2 0.2 0.2 1"/>\n'
        '    <site name="waste" pos="0.30 -0.45 0.12" size="0.004" rgba="0 1 0 0.5" group="4"/>\n'
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
    <material name="mat_pipette" rgba="0.85 0.85 0.88 1"/>
    <material name="mat_tube" rgba="0.80 0.90 1.0 0.45"/>
    <material name="mat_rack" rgba="0.35 0.42 0.55 1"/>
    <material name="mat_tipbox" rgba="0.30 0.55 0.85 1"/>
  </asset>

  <worldbody>
    <light pos="0.4 0 1.5" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="0 0 0.05" pos="0 0 -0.75" material="lab_floor"/>
    <geom name="bench" type="box" size="0.55 0.70 0.375" pos="0.40 0 -0.375" material="bench_top"/>

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
    return bench.compile()


if __name__ == "__main__":
    out = write_lab_xml()
    model = load_model()   # validate the full compose on build
    print(f"wrote {out}  (bench + Panda compiles: {model.nbody} bodies, "
          f"{model.nsite} sites, {model.ngeom} geoms, {model.nu} actuators)")
