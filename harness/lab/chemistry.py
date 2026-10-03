"""Hidden ground truth: an alkaline-phosphatase-like enzyme and a plate reader.

Nothing in this module is visible to the agent. The lab calls it with the
composition each well *actually* received (after pipetting error and spills)
and returns kinetic absorbance reads at 405 nm.

Effects modelled, each tied to a claim in the protocol:
- Michaelis-Menten on pNPP with substrate depletion and product (Pi) inhibition.
- pH bell curve; buffer capacity: outside its range the effective pH drifts.
- DEA and Tris act as phosphate acceptors (transphosphorylation) and raise rate;
  glycine weakly; PBS contributes phosphate, a competitive product inhibitor.
- Tris pKa falls ~0.028 /C, so its delivered pH shifts with temperature.
- Mg2+ is an activator (saturating); Zn2+ is required but inhibits in excess.
- NaCl has a shallow optimum; glycerol stabilises but slows via viscosity.
- Temperature: Arrhenius activation against first-order thermal inactivation.
- Non-enzymatic pNPP hydrolysis grows with pH and temperature (blank signal).
- Edge wells evaporate: concentrated, noisier signal.
- The detector saturates above ~3.5 A and is linear only below 2.5 A.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config as C


@dataclass
class EnzymeParams:
    vmax: float          # uM pNP / min / ug enzyme at optimum
    km_mM: float
    ki_pi_mM: float      # competitive inhibition by phosphate
    ph_opt: float
    ph_width: float
    t_opt_c: float
    ea_scale: float      # Arrhenius steepness
    inact_rate_45: float  # 1/min thermal inactivation at 45 C
    mg_k_mM: float
    zn_opt_mM: float
    zn_width: float      # log10 width of the Zn bell
    nacl_opt_mM: float
    acceptor_boost: dict
    noise_cv: float

    @classmethod
    def sample(cls, rng: np.random.Generator) -> "EnzymeParams":
        return cls(
            vmax=float(rng.uniform(40, 80)),
            km_mM=float(rng.uniform(0.3, 2.5)),
            ki_pi_mM=float(rng.uniform(0.05, 0.3)),
            ph_opt=float(rng.uniform(8.8, 10.2)),
            ph_width=float(rng.uniform(0.8, 1.4)),
            t_opt_c=float(rng.uniform(32, 42)),
            ea_scale=float(rng.uniform(0.05, 0.08)),
            inact_rate_45=float(rng.uniform(0.02, 0.15)),
            mg_k_mM=float(rng.uniform(0.2, 1.5)),
            zn_opt_mM=float(rng.uniform(0.01, 0.06)),
            zn_width=float(rng.uniform(0.5, 0.9)),
            nacl_opt_mM=float(rng.uniform(50, 200)),
            acceptor_boost={"DEA": float(rng.uniform(1.6, 2.4)), "Tris": float(rng.uniform(1.2, 1.6)),
                            "Glycine": float(rng.uniform(1.0, 1.2)), "PBS": 1.0},
            noise_cv=0.06,
        )


@dataclass
class WellContents:
    """What physically ended up in a well (concentrations in the 300 uL well)."""

    buffer: str | None = None
    buffer_ph: float | None = None
    buffer_fraction: float = 0.0   # fraction of the nominal buffer volume delivered
    substrate_mM: float = 0.0
    mg_mM: float = 0.0
    zn_mM: float = 0.0
    nacl_mM: float = 0.0
    glycerol_pct: float = 0.0
    enzyme_ug: float = 0.0
    pnp_mM: float = 0.0            # product spiked in (standards, spike recovery)
    phosphate_mM: float = 0.0      # contamination or PBS
    volume_ul: float = 0.0
    temperature_c: float = 25.0


class EnzymeWorld:
    def __init__(self, params: EnzymeParams, rng: np.random.Generator):
        self.p = params
        self.rng = rng
        self.enzyme_decay_per_hour = 0.03  # bench stability of the enzyme stock

    # ------------------------------------------------------------ physics
    def effective_ph(self, w: WellContents) -> float:
        if w.buffer is None:
            return 7.0
        (lo, hi), pka = C.BUFFERS[w.buffer]
        ph = w.buffer_ph
        if w.buffer == "Tris":
            ph -= 0.028 * (w.temperature_c - 25.0)
        # Weak buffering outside the usable range or when under-dosed: drift
        # halfway back toward the pKa.
        drift = 0.0
        if not lo <= w.buffer_ph <= hi:
            drift += 0.4
        if w.buffer_fraction < 0.8:
            drift += 0.5 * (0.8 - w.buffer_fraction) / 0.8
        return float(ph + min(drift, 0.9) * (pka - ph))

    def rate_constant_factor(self, w: WellContents, t_min: float = 0.0) -> float:
        """Multiplicative activity factor excluding substrate kinetics."""
        p = self.p
        ph = self.effective_ph(w)
        f_ph = np.exp(-0.5 * ((ph - p.ph_opt) / p.ph_width) ** 2)
        f_mg = 0.25 + 0.75 * w.mg_mM / (p.mg_k_mM + w.mg_mM)
        if w.zn_mM <= 0:
            f_zn = 0.05
        else:
            f_zn = np.exp(-0.5 * (np.log10(w.zn_mM / p.zn_opt_mM) / p.zn_width) ** 2)
        f_salt = 1.0 - 0.35 * np.tanh(abs(w.nacl_mM - p.nacl_opt_mM) / 250.0)
        f_visc = 1.0 / (1.0 + 0.025 * w.glycerol_pct)
        stab = 1.0 + 0.02 * w.glycerol_pct  # glycerol slows thermal inactivation
        dt = w.temperature_c - p.t_opt_c
        f_arr = np.exp(p.ea_scale * min(dt, 0.0))
        k_inact = p.inact_rate_45 * np.exp(0.25 * (w.temperature_c - 45.0)) / stab
        f_inact = np.exp(-k_inact * t_min)
        boost = p.acceptor_boost.get(w.buffer, 1.0) if w.buffer else 1.0
        return float(f_ph * f_mg * f_zn * f_salt * f_visc * f_arr * f_inact * boost)

    def phosphate_mM(self, w: WellContents) -> float:
        pi = w.phosphate_mM
        if w.buffer == "PBS":
            pi += 33.0 * w.buffer_fraction  # 100 mM PBS stock diluted 1:3
        return pi

    def progress_curve(self, w: WellContents, times_min, enzyme_age_h: float = 0.0) -> np.ndarray:
        """pNP concentration (mM) over time, integrated with small Euler steps."""
        p = self.p
        times = np.asarray(times_min, float)
        dt = 0.05
        s = w.substrate_mM
        prod = w.pnp_mM
        pi0 = self.phosphate_mM(w)
        enz = w.enzyme_ug * np.exp(-self.enzyme_decay_per_hour * enzyme_age_h)
        out = []
        t = 0.0
        ph = self.effective_ph(w)
        k_hyd = 2e-5 * 10 ** (0.35 * (ph - 9.0)) * np.exp(0.05 * (w.temperature_c - 37.0))
        for target in times:
            while t < target - 1e-9:
                pi = pi0 + (prod - w.pnp_mM)
                km_app = p.km_mM * (1 + pi / p.ki_pi_mM)
                v_enz = p.vmax * enz * self.rate_constant_factor(w, t) * s / (km_app + s) / 1000.0
                v = (v_enz + k_hyd * s) * dt
                v = min(v, s)
                s -= v
                prod += v
                t += dt
            out.append(prod)
        return np.array(out)

    def true_initial_rate(self, cond_contents: WellContents) -> float:
        """Noise-free initial rate (uM/min/ug) used by the scorer."""
        c = self.progress_curve(cond_contents, [0.0, 0.5])
        blank = WellContents(**{**cond_contents.__dict__, "enzyme_ug": 0.0})
        b = self.progress_curve(blank, [0.0, 0.5])
        ug = max(cond_contents.enzyme_ug, 1e-9)
        return float(((c[1] - c[0]) - (b[1] - b[0])) / 0.5 * 1000.0 / ug)

    # -------------------------------------------------------------- reader
    def absorbance(self, w: WellContents, times_min, edge: bool, wavelength_nm: int = 405,
                   enzyme_age_h: float = 0.0) -> np.ndarray:
        evap = 1.0
        activity_cv = self.p.noise_cv   # well-to-well enzyme activity variation
        optical_cv = 0.012
        if edge:
            evap = 1.0 + self.rng.uniform(0.04, 0.12)
            activity_cv *= 1.8
            optical_cv *= 1.8
        jittered = WellContents(**{**w.__dict__,
                                   "enzyme_ug": w.enzyme_ug * max(0.0, 1 + self.rng.normal(0, activity_cv))})
        pnp = self.progress_curve(jittered, times_min, enzyme_age_h) * evap
        scatter = 0.002 * w.glycerol_pct + 0.03  # turbidity, roughly flat across wavelengths
        if wavelength_nm == 405:
            signal = C.PNP_EPSILON_405 * C.PATH_LENGTH_CM * pnp
        else:  # 490 nm reference: pNP barely absorbs
            signal = 0.02 * C.PNP_EPSILON_405 * C.PATH_LENGTH_CM * pnp
        a = signal + scatter
        well_gain = 1.0 + self.rng.normal(0, optical_cv)
        a = a * well_gain + self.rng.normal(0, 0.003, size=a.shape)
        # Detector compresses then clips.
        a = np.where(a > C.DETECTOR_LINEAR_MAX_A,
                     C.DETECTOR_LINEAR_MAX_A + (a - C.DETECTOR_LINEAR_MAX_A) * 0.4, a)
        return np.minimum(a, C.DETECTOR_SATURATION_A)
