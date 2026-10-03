# Vendored assets — AutoBio

The mesh files under this directory are **visual meshes** vendored from the AutoBio project
and used unmodified (scaled/positioned only in XML, never edited — see licence note).

- **Source:** https://github.com/autobio-bench/AutoBio
- **Commit pinned:** `08504213356cbfe9a82bcd33a43cd282945a45df` (2025-05-28)
- **Paper:** AutoBio: A Simulation and Benchmark for Robotic Automation in Digital Biology
  Laboratory (ICLR 2026), arXiv:2505.14030
- **Licence:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
  (CC BY-NC-SA 4.0), https://creativecommons.org/licenses/by-nc-sa/4.0/

## Licence implications (read before using commercially)

- **NonCommercial:** these assets may be used for research/eval/demo only, not in a
  commercial product.
- **ShareAlike:** if you *modify* a mesh file, the modified file must be shared under
  CC BY-NC-SA 4.0. We therefore **never edit the mesh files** — all scaling and placement
  happens in `scenes/build_lab.py` / the generated XML, which keeps our own code unencumbered.
- **Attribution:** keep this NOTICE with the files.

## Files (round one)

| File | AutoBio source path | Scale | Used for |
| --- | --- | --- | --- |
| `tool/pipette/{body,tube,connector,knob,pusher_mid,pusher_right1,pusher_right2,pusher_right3}_visual.obj` | `assets/tool/pipette/*_visual.obj` | 0.1 | pipette (8 visual parts assembled at a common origin ≈ 199 mm; the single `pipette.obj` is multi-object and MuJoCo only partly loads it) |
| `container/centrifuge_15ml_body.STL` | `assets/container/centrifuge_15ml_screw_vis/centrifuge_tube_15ml_body.STL` | 0.001 | reagent stock tubes (≈ 16.8 mm × 118.6 mm; body only, no cap) |
| `container/centrifuge_1500ul_no_lid.obj` | `assets/container/centrifuge_1500ul_no_lid_vis/visual.obj` | 0.001 | (parked) 1.5 mL tube for later rounds (≈ 13 mm × 40 mm) |
| `rack/centrifuge_10slot/{pillars,lower_plane,upper_plane}.obj` | `assets/rack/centrifuge_10slot_vis/*_visual.obj` | 0.001 | 10-slot tube rack (multi-part) |
| `rack/tip_box_24slot/{up,low}.obj` | `assets/rack/tip_box_24slot_vis/*_visual.obj` | 0.001 | pipette-tip box (multi-part) |
| `rack/pipette_rack_tri/{beam,legs}.obj` | `assets/rack/pipette_rack_tri_vis/*_visual.obj` | 0.001 | pipette holder (optional) |

Collision geometry is **not** vendored: we build our own primitive colliders in
`build_lab.py`, matching AutoBio's own approach (their `.gen.xml` colliders are primitive
boxes too). Visual meshes are placed with `contype=0 conaffinity=0 group=2`.
