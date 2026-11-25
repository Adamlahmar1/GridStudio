# Author: Adam Lahmar <adam.lahmar@rwth-aachen.de>
"""Optional improvements around the unchanged synthesis tool (nothing in powergrid_synth/ is edited).

* library_from_reference : build a cable library from the reference feeder itself (cable types and their share of the length)
* refit_length           : refit the cable-length distribution as a TRUNCATED Cauchy (the sampler cannot draw lengths > 10 m)
* FrequencyPickGenerator : draw the cable type by the `frequency` of the library entries (the comment in the original code
                           describes this, but the original code only looks at the closest current rating)
All three are switched on or off per feeder in config.py. The Italian DSO profile uses none of them, so Italian DSO is unchanged.
"""
from __future__ import annotations

import copy

import numpy as np
from scipy import optimize, stats

from powergrid_synth.distribution.distribution_params import CableLibraryEntry, ModifiedCauchyParams
from powergrid_synth.distribution.distribution_synthesis import SchweetzerFeederGenerator

CAP_KM = 0.01     # the generator draws lengths in (0, max(fitted clip, 0.01 km)]


def assumed_rating_ka(r_ohm_per_km: float) -> float:
    """ASSUMPTION for data without current ratings: rating grows with the cross-section, i.e. with 1/R.
    I_max = 0.10 kA * (1 Ohm/km / R)^0.6 (R = 1.15 -> 0.09 kA, R = 0.45 -> 0.16 kA, R = 0.09 -> 0.43 kA).
    Only used to choose a cable and to apply the tool's threshold, never for the voltage drop."""
    return 0.10 * (1.0 / r_ohm_per_km) ** 0.6


def library_from_reference(ref) -> list:
    """Cable types of the reference line table. Lines are grouped by their name (e.g. a line code);
    the share of the total cable length becomes the `frequency` of the entry."""
    lines = ref.net.line
    key = lines["name"].astype(str)
    entries = []
    total = float(lines["length_km"].sum())
    for name in sorted(key.unique()):
        g = lines[key == name]
        r, x = float(g["r_ohm_per_km"].iloc[0]), float(g["x_ohm_per_km"].iloc[0])
        c = float(g["c_nf_per_km"].iloc[0])
        rating = float(g["max_i_ka"].iloc[0]) if ref.has_ratings else assumed_rating_ka(r)
        entries.append(CableLibraryEntry(name, r, x, c, rating, float(g["length_km"].sum()) / total))
    return entries


def refit_length(real_feeder) -> ModifiedCauchyParams:
    """Maximum-likelihood fit of a Cauchy distribution truncated to (0, CAP_KM] to the real segment lengths.
    Bounded optimisation: an unbounded fit diverged (negative location, width 0) in a first attempt."""
    # only real cables: transformer and switch connections have placeholder lengths (as in the tool's own analysis)
    x = np.array([d["length_km"] for _, _, d in real_feeder.edges(data=True) if d.get("edge_type", "line") == "line"])
    x = x[(x > 0) & (x <= CAP_KM)]

    def nll(theta):
        x0, g = theta
        norm = stats.cauchy.cdf(CAP_KM, loc=x0, scale=g) - stats.cauchy.cdf(0.0, loc=x0, scale=g)
        return -(np.sum(stats.cauchy.logpdf(x, loc=x0, scale=g)) - len(x) * np.log(max(norm, 1e-300)))

    best = None
    for x0 in (0.0002, 0.0005, 0.001, 0.002):
        for g in (0.0002, 0.0005, 0.001, 0.003):
            r = optimize.minimize(nll, x0=[x0, g], method="L-BFGS-B", bounds=[(0.0, CAP_KM), (1e-6, CAP_KM)])
            if best is None or r.fun < best.fun:
                best = r
    return ModifiedCauchyParams(x0=float(best.x[0]), gamma=float(best.x[1]))


class FrequencyPickGenerator(SchweetzerFeederGenerator):
    """Same generator, but the cable type is drawn with the probability `frequency` of the library entries."""

    def _pick_cable_by_capacity(self, i_nom_target, h, library, threshold):
        filtered = library
        if h >= threshold.h_min:
            f = [c for c in library if c.max_i_ka <= threshold.i_nom_max / 1000.0]
            filtered = f or library
        w = np.array([c.frequency for c in filtered], dtype=float)
        return filtered[int(self.rng.choice(len(filtered), p=w / w.sum()))]


def tuned_params(cfg, ref, params, real_feeder):
    """Apply the options of the feeder profile to the fitted parameters (a copy; the fit itself is not changed)."""
    p = copy.deepcopy(params)
    if cfg.cable_library == "reference":
        p.cable_library = library_from_reference(ref)
    if cfg.length_fit == "truncated":
        p.cable_length = refit_length(real_feeder)
    return p


def generator_class(cfg):
    return FrequencyPickGenerator if cfg.cable_choice == "frequency" else SchweetzerFeederGenerator
