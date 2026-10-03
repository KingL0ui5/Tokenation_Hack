"""Generate the static lab bench and compose it with the Menagerie Panda.

`build()` writes `scenes/lab.xml` — the bench ALONE (labware, stations, cameras), with
no robot. `load_model()` is the entry point everything else uses: it (re)builds lab.xml
if stale, then uses MuJoCo's MjSpec API to load the pristine Panda, inject the IK
end-effector site, and attach the arm into the bench — so `models/franka_emika_panda/`
is never edited and no `scenes/assets` symlink is needed.

Run from the repo root:  python -m scenes.build_lab   (writes lab.xml)
Load in code:            from scenes.build_lab import load_model; m = load_model()

Site names are the contract with the robot/tool layer (well_A1, reservoir_enzyme,
rack_1, station_reader, pipette_grip, waste, ...). Edit the layout constants below and
re-run; never hand-edit lab.xml.

Everything here is STATIC (no free joints), so the Panda's "home" keyframe still
matches. When objects need to be grasped or knocked over, give them free joints and
define a new keyframe with the extra qpos entries.
"""

from __future__ import annotations

import os
from pathlib import Path

import mujoco

REPO = Path(__file__).resolve().parent.parent
PANDA_XML = REPO / "models" / "franka_emika_panda" / "panda.xml"
FRANKA_ASSETS = REPO / "models" / "franka_emika_panda" / "assets"
LAB_XML = Path(__file__).with_name("lab.xml")

# End-effector site injected into the Panda hand at load time (kept out of panda.xml so
# the vendored Menagerie model stays pristine). Same pose as the earlier in-XML version:
# grasp midpoint between the fingertips; +45deg z-quat cancels the hand's -45deg mount.
EE_SITE = dict(name="attachment_site", pos=[0, 0, 0.1034],
               quat=[0.9238795, 0, 0, 0.3826834], group=4)

ROWS, COLS = "ABCD", 6
PLATE_CENTER = (0.50, 0.00)        # x, y of plate centre (robot base at origin)
WELL_PITCH = 0.018
WELL_R, WELL_H = 0.007, 0.012
RESERVOIRS = {                      # name: rgba of its liquid
    "buffer":    "0.80 0.90 1.00 0.6",
    "enzyme":    "0.85 0.95 0.80 0.6",
    "substrate": "0.95 0.95 0.95 0.6",
    "inhibitor": "0.95 0.80 0.85 0.6",
    "stop":      "0.75 0.75 0.95 0.6",
}
RES_ROW_Y, RES_X0, RES_DX = -0.30, 0.36, 0.08
RES_R, RES_H = 0.025, 0.05
RACK_CENTER = (0.42, 0.28)
TUBE_R, TUBE_H, TUBE_DY = 0.006, 0.040, 0.025

GLASS = "0.90 0.95 1.00 0.25"


def vessel(slot: str, x: float, y: float, r: float, h: float, z0: float, liquid_rgba: str) -> str:
    """Translucent container + an (initially empty) liquid geom + a target site."""
    zc = z0 + h / 2
    return (
        f'    <geom name="vessel_{slot}" type="cylinder" size="{r:.4f} {h / 2:.4f}" '
        f'pos="{x:.4f} {y:.4f} {zc:.4f}" rgba="{GLASS}" contype="0" conaffinity="0" group="1"/>\n'
        f'    <geom name="liquid_{slot}" type="cylinder" size="{r * 0.85:.4f} 0.0001" '
        f'pos="{x:.4f} {y:.4f} {z0 + 0.0001:.4f}" rgba="{liquid_rgba}" contype="0" conaffinity="0" group="1"/>\n'
        f'    <site name="{slot}" pos="{x:.4f} {y:.4f} {z0 + h + 0.01:.4f}" size="0.003" rgba="1 0 0 0.5" group="4"/>\n'
    )


def build() -> str:
    parts: list[str] = []
    px, py = PLATE_CENTER
    plate_w = COLS * WELL_PITCH + 0.01
    plate_d = len(ROWS) * WELL_PITCH + 0.05
    base_h = 0.004

    # plate: thin base, invisible collision block covering the wells, then wells
    parts.append(
        f'    <geom name="plate_base" type="box" size="{plate_w / 2:.4f} {plate_d / 2:.4f} {base_h / 2:.4f}" '
        f'pos="{px} {py} {base_h / 2:.4f}" rgba="0.95 0.95 0.95 1"/>\n'
        f'    <geom name="plate_collision" type="box" size="{plate_w / 2:.4f} {plate_d / 2:.4f} {WELL_H / 2:.4f}" '
        f'pos="{px} {py} {base_h + WELL_H / 2:.4f}" rgba="0 0 0 0" group="3"/>\n'
    )
    for i, r in enumerate(ROWS):
        for j in range(COLS):
            x = px + (i - (len(ROWS) - 1) / 2) * WELL_PITCH       # rows run along x
            y = py + (j - (COLS - 1) / 2) * WELL_PITCH            # columns run along y
            parts.append(vessel(f"well_{r}{j + 1}", x, y, WELL_R, WELL_H, base_h, "1 1 0.6 0.9"))

    # reservoirs
    for k, (name, rgba) in enumerate(RESERVOIRS.items()):
        x = RES_X0 + k * RES_DX
        parts.append(
            f'    <geom name="collide_reservoir_{name}" type="cylinder" size="{RES_R:.4f} {RES_H / 2:.4f}" '
            f'pos="{x:.4f} {RES_ROW_Y} {RES_H / 2:.4f}" rgba="0 0 0 0" group="3"/>\n'
        )
        parts.append(vessel(f"reservoir_{name}", x, RES_ROW_Y, RES_R, RES_H, 0.0, rgba))

    # tube rack + tubes
    rx, ry = RACK_CENTER
    rack_h = 0.03
    parts.append(
        f'    <geom name="tube_rack" type="box" size="0.02 0.06 {rack_h / 2}" pos="{rx} {ry} {rack_h / 2}" '
        f'rgba="0.3 0.5 0.8 1"/>\n'
    )
    for t in range(4):
        y = ry + (t - 1.5) * TUBE_DY
        parts.append(vessel(f"rack_{t + 1}", rx, y, TUBE_R, TUBE_H, rack_h, "0.95 0.95 0.85 0.8"))

    # stations, pipette, waste
    parts.append(
        '    <geom name="plate_reader" type="box" size="0.10 0.08 0.05" pos="0.68 0.30 0.05" rgba="0.25 0.25 0.28 1"/>\n'
        '    <geom name="reader_slot" type="box" size="0.07 0.05 0.002" pos="0.62 0.30 0.101" rgba="0.1 0.1 0.1 1" contype="0" conaffinity="0"/>\n'
        '    <site name="station_reader" pos="0.62 0.30 0.12" size="0.004" rgba="0 1 0 0.5" group="4"/>\n'
        '    <geom name="incubator" type="box" size="0.08 0.08 0.05" pos="0.74 -0.05 0.05" rgba="0.85 0.55 0.30 1"/>\n'
        '    <site name="station_incubator" pos="0.66 -0.05 0.12" size="0.004" rgba="0 1 0 0.5" group="4"/>\n'
        f'    <site name="station_bench" pos="{px} {py} 0.03" size="0.004" rgba="0 1 0 0.5" group="4"/>\n'
        '    <geom name="pipette_holder" type="box" size="0.02 0.02 0.04" pos="0.28 0.30 0.04" rgba="0.5 0.5 0.5 1"/>\n'
        '    <geom name="pipette" type="cylinder" size="0.008 0.07" pos="0.28 0.30 0.15" rgba="0.92 0.92 0.92 1"/>\n'
        '    <site name="pipette_grip" pos="0.28 0.30 0.18" size="0.004" rgba="0 0 1 0.5" group="4"/>\n'
        '    <geom name="waste_bin" type="cylinder" size="0.05 0.05" pos="0.30 -0.45 0.05" rgba="0.2 0.2 0.2 1"/>\n'
        '    <site name="waste" pos="0.30 -0.45 0.12" size="0.004" rgba="0 1 0 0.5" group="4"/>\n'
    )

    labware = "".join(parts)
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
  </asset>

  <worldbody>
    <light pos="0.4 0 1.5" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="0 0 0.05" pos="0 0 -0.75" material="lab_floor"/>
    <geom name="bench" type="box" size="0.55 0.70 0.375" pos="0.40 0 -0.375" material="bench_top"/>

    <camera name="front" pos="1.60 0 0.55" xyaxes="0 1 0 -0.423 0 0.906"/>
    <camera name="side" pos="0.45 -1.30 0.70" xyaxes="1 0 0 0 0.5 0.866"/>
    <camera name="plate_top" pos="{px} {py} 0.45" xyaxes="0 -1 0 1 0 0"/>

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
          f"{model.nsite} sites, {model.nu} actuators)")
