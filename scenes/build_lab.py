"""Generate scenes/lab.xml: the static lab bench around the Menagerie Panda.

Run from the repo root:  python -m scenes.build_lab
Edit the layout constants below and re-run instead of hand-editing lab.xml.

Site names match the `slot` fields in experiments/lab.py (well_A1, reservoir_enzyme,
rack_1...), so the robot can target any container by name.

Everything here is STATIC (no free joints), so the Panda's "home" keyframe still
matches. When objects need to be grasped or knocked over, give them free joints and
define a new keyframe with the extra qpos entries.
"""

from __future__ import annotations

from pathlib import Path

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
    return f"""<!-- GENERATED by scenes/build_lab.py. Edit that file and re-run, not this one. -->
<mujoco model="agentic_lab">
  <include file="../models/franka_emika_panda/panda.xml"/>

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


def ensure_assets_link() -> None:
    """panda.xml loads meshes from "assets/", resolved next to the MAIN model file,
    so scenes/ needs an assets link pointing at the Panda's asset folder."""
    link = Path(__file__).with_name("assets")
    if not link.exists():
        link.symlink_to(Path("..") / "models" / "franka_emika_panda" / "assets", target_is_directory=True)
        print(f"linked {link} -> ../models/franka_emika_panda/assets")


if __name__ == "__main__":
    ensure_assets_link()
    out = Path(__file__).with_name("lab.xml")
    out.write_text(build())
    print(f"wrote {out}")
