"""
Distribution analysis: fit Schweitzer parameters from reference feeders.

Provides routines to extract empirical distributions from a collection of
real feeders and fit the parametric families used in the Schweitzer model
(via MLE / method-of-moments / KL minimization).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from scipy import optimize, stats

from .distribution_params import (
    BetaParams,
    DistributionSynthParams,
    ExponentialClip,
    ExponentialParams,
    MixtureGammaParams,
    MixtureNormalParams,
    MixturePoissonParams,
    ModifiedCauchyParams,
    NegBinomParams,
    NormalParams,
    PowerLawClip,
    TLocationScaleParams,
)


# ------------------------------------------------------------------
# Extraction helpers
# ------------------------------------------------------------------

def _hop_distances(feeders: Sequence[nx.Graph]) -> np.ndarray:
    """Collect all hop distances (excluding source h=0 and root h=1)."""
    hops = []
    for G in feeders:
        for n in G.nodes:
            h = G.nodes[n].get("h", 0)
            if h >= 2:
                hops.append(h - 2)  # shift to 0-based for NegBinom fitting
    return np.array(hops)


def _degrees(feeders: Sequence[nx.Graph]) -> np.ndarray:
    """Collect all node degrees (excluding source/leaf fixed degrees)."""
    degs = []
    for G in feeders:
        max_h = max(G.nodes[n]["h"] for n in G.nodes)
        for n in G.nodes:
            h = G.nodes[n]["h"]
            if h == 0 or h == max_h:
                continue  # source / leaf — deterministic
            degs.append(G.degree(n))
    return np.array(degs, dtype=float)


def _cable_lengths(feeders: Sequence[nx.Graph]) -> np.ndarray:
    """Collect all cable lengths in km.

    Excludes switch/transformer edges (placeholder lengths, not physical
    cable runs) when ``edge_type`` is available; falls back to including
    everything with a positive length otherwise.
    """
    lens = []
    for G in feeders:
        for e in G.edges:
            d = G.edges[e]
            l = d.get("length_km", None)
            if l is None or l <= 0:
                continue
            if d.get("edge_type", "line") != "line":
                continue
            lens.append(l)
    return np.array(lens)


def _intermediate_fractions(feeders: Sequence[nx.Graph]) -> np.ndarray:
    """Fraction of nodes that are intermediate, per feeder."""
    fracs = []
    for G in feeders:
        n = G.number_of_nodes()
        n_int = sum(
            1 for nd in G.nodes if G.nodes[nd].get("node_type") == "intermediate"
        )
        fracs.append(n_int / n if n > 0 else 0.0)
    return np.array(fracs)


def _injection_fractions(feeders: Sequence[nx.Graph]) -> np.ndarray:
    """Fraction of nodes that are injection, per feeder."""
    fracs = []
    for G in feeders:
        n = G.number_of_nodes()
        n_inj = sum(
            1 for nd in G.nodes if G.nodes[nd].get("node_type") == "injection"
        )
        fracs.append(n_inj / n if n > 0 else 0.0)
    return np.array(fracs)


def _load_deviations(feeders: Sequence[nx.Graph]) -> np.ndarray:
    """Per-feeder deviations of load from uniform share."""
    devs = []
    for G in feeders:
        load_nodes = [
            n for n in G.nodes
            if G.nodes[n].get("node_type") == "load"
            and G.nodes[n].get("P_mw", 0) > 0
        ]
        if not load_nodes:
            continue
        total = sum(G.nodes[n]["P_mw"] for n in load_nodes)
        uniform = total / len(load_nodes) if total > 0 else 1.0
        for n in load_nodes:
            devs.append((G.nodes[n]["P_mw"] - uniform) / total if total > 0 else 0.0)
    return np.array(devs)


# ------------------------------------------------------------------
# Fitting routines
# ------------------------------------------------------------------

def fit_neg_binomial(data: np.ndarray) -> NegBinomParams:
    """Fit Negative Binomial to hop-distance data (0-shifted)."""
    mean_x = data.mean()
    var_x = data.var()
    if var_x <= mean_x:
        var_x = mean_x + 0.01  # ensure overdispersion for NB
    p = mean_x / var_x
    r = mean_x * p / (1 - p)
    return NegBinomParams(r=float(r), p=float(p))


def fit_mixture_gamma(data: np.ndarray) -> MixtureGammaParams:
    """Fit a two-component Gamma mixture via EM (simplified).

    Uses a basic EM iteration (50 steps) with Gamma MLE per component.
    Falls back to defaults if the data is insufficient.
    """
    data = data[data > 0]
    if len(data) < 10:
        return MixtureGammaParams()  # fallback to defaults

    # Initialize by splitting at median
    med = np.median(data)
    z = (data <= med).astype(float)
    pi = z.mean()

    a1, b1, a2, b2 = 1.5, 0.65, 4.4, 1.67  # safe defaults

    for _ in range(50):
        d1 = data[z > 0.5]
        d2 = data[z <= 0.5]
        if len(d1) < 2 or len(d2) < 2:
            break

        try:
            a1, _, b1 = stats.gamma.fit(d1, floc=0)
            a2, _, b2 = stats.gamma.fit(d2, floc=0)
        except (ValueError, RuntimeError):
            break

        # E-step
        pdf1 = stats.gamma.pdf(data, a1, scale=b1) * pi
        pdf2 = stats.gamma.pdf(data, a2, scale=b2) * (1 - pi)
        total = pdf1 + pdf2
        total = np.where(total < 1e-300, 1e-300, total)
        z = pdf1 / total
        pi = z.mean()

    return MixtureGammaParams(pi=float(pi), a1=float(a1), b1=float(b1), a2=float(a2), b2=float(b2))


def fit_beta(data: np.ndarray) -> BetaParams:
    """Fit Beta distribution to fraction data in (0, 1)."""
    data = data[(data > 0) & (data < 1)]
    if len(data) < 2:
        return BetaParams()
    a, b, _, _ = stats.beta.fit(data, floc=0, fscale=1)
    return BetaParams(alpha=float(a), beta=float(b))


def _fit_beta_or_default(data: np.ndarray, default: BetaParams) -> BetaParams:
    """Fit a Beta distribution if there's enough informative data in (0, 1),
    otherwise fall back to a calibrated default.

    ``fit_beta`` itself falls back to an uninformative Beta(1,1) = Uniform(0,1)
    when too few points survive its (0, 1) filtering -- e.g. when every real
    feeder we have shows exactly 0% (or 100%) of a fraction. Checking
    ``len(data) > 0`` at the call site doesn't catch that, since a single
    all-zero array still has length 1. Filtering the same way here first
    lets the caller tell "no data" from "data with no informative spread"
    apart, and use its own calibrated prior for the latter instead of
    silently sampling from Uniform(0, 1).
    """
    filtered = data[(data > 0) & (data < 1)]
    return fit_beta(data) if len(filtered) >= 2 else default


def fit_t_location_scale(data: np.ndarray) -> TLocationScaleParams:
    """Fit t-Location-Scale distribution via MLE."""
    if len(data) < 3:
        return TLocationScaleParams()
    nu, mu, sigma = stats.t.fit(data)
    return TLocationScaleParams(mu=float(mu), sigma=float(sigma), nu=float(nu))


def fit_modified_cauchy(data: np.ndarray) -> ModifiedCauchyParams:
    """Fit positive-support Cauchy (x0, gamma) to cable-length data."""
    data = data[data > 0]
    if len(data) < 3:
        return ModifiedCauchyParams()
    loc, scale = stats.cauchy.fit(data)
    return ModifiedCauchyParams(x0=float(loc), gamma=float(scale))


def fit_exponential_clip(
    feeders: Sequence[nx.Graph],
) -> ExponentialClip:
    """Fit exponential clipping g_max(h) = a * exp(b*h) from max lengths per hop."""
    from collections import defaultdict
    max_by_h: Dict[int, float] = defaultdict(float)
    for G in feeders:
        for u, v in G.edges:
            if G.edges[u, v].get("edge_type", "line") != "line":
                continue
            h = max(G.nodes[u]["h"], G.nodes[v]["h"])
            l = G.edges[u, v].get("length_km", 0)
            if l > max_by_h[h]:
                max_by_h[h] = l

    if len(max_by_h) < 2:
        return ExponentialClip()

    hs = np.array(sorted(max_by_h.keys()), dtype=float)
    vals = np.array([max_by_h[int(h)] for h in hs])
    vals = np.where(vals > 0, vals, 1e-6)

    # Linear regression on log(vals) = log(a) + b*h
    log_vals = np.log(vals)
    coeffs = np.polyfit(hs, log_vals, 1)
    b = float(coeffs[0])
    a = float(np.exp(coeffs[1]))
    return ExponentialClip(a=a, b=b)


def fit_power_law_clip(
    feeders: Sequence[nx.Graph],
) -> PowerLawClip:
    """Fit power-law clipping g_dmax(h) = a * h^b from max degrees per hop."""
    from collections import defaultdict
    max_by_h: Dict[int, int] = defaultdict(int)
    for G in feeders:
        for n in G.nodes:
            h = G.nodes[n]["h"]
            d = G.degree(n)
            if d > max_by_h[h]:
                max_by_h[h] = d

    hs_raw = sorted(h for h in max_by_h if h > 0)
    if len(hs_raw) < 2:
        return PowerLawClip()

    hs = np.array(hs_raw, dtype=float)
    vals = np.array([max_by_h[int(h)] for h in hs], dtype=float)
    vals = np.where(vals > 0, vals, 1)

    log_h = np.log(hs)
    log_v = np.log(vals)
    coeffs = np.polyfit(log_h, log_v, 1)
    b = float(coeffs[0])
    a = float(np.exp(coeffs[1]))
    return PowerLawClip(a=a, b=b)


# ------------------------------------------------------------------
# High-level fitting
# ------------------------------------------------------------------

def _measured_beta(fraction: float, concentration: float = 5000.0) -> BetaParams:
    """Beta distribution with the given mean and a narrow spread (a 'measured value' that the generator can sample).
    With 500 the number of load buses of a feeder with some hundred buses varied by about +-4 around the measured value;
    with 5000 it varies by about +-1.5."""
    f = min(max(float(fraction), 0.005), 0.995)
    return BetaParams(alpha=f * concentration, beta=(1.0 - f) * concentration)


def _net_export_fractions(feeders: Sequence[nx.Graph]) -> np.ndarray:
    """Fraction of nodes with negative net power (generation larger than load), per feeder."""
    fracs = []
    for G in feeders:
        n = G.number_of_nodes()
        n_exp = sum(1 for nd in G.nodes if G.nodes[nd].get("P_mw", 0.0) < 0)
        fracs.append(n_exp / n if n > 0 else 0.0)
    return np.array(fracs)


def _injection_deviations(feeders: Sequence[nx.Graph]) -> np.ndarray:
    """Deviations of the generation of each exporting node from a uniform share (same definition as for loads)."""
    devs = []
    for G in feeders:
        exp = [abs(G.nodes[n]["P_mw"]) for n in G.nodes if G.nodes[n].get("P_mw", 0.0) < 0]
        total = sum(exp)
        if not exp or total <= 0:
            continue
        uniform = total / len(exp)
        devs.extend((p - uniform) / total for p in exp)
    return np.array(devs)


def _power_node_positions_and_shares(feeders: Sequence[nx.Graph], sign: float):
    """Positions (hop distance / largest hop distance of the feeder), shares of the total and power factors of the nodes
    whose net power has the given sign (+1 consumers, -1 exporters). Shares and power factors are sorted by share,
    largest first, so that entry i of both lists describes the same node."""
    hops, powers, pfs = [], [], []
    for G in feeders:
        h_max = max((G.nodes[n]["h"] for n in G.nodes), default=0)
        for n in G.nodes:
            p = G.nodes[n].get("P_mw", 0.0)
            if p * sign > 0 and h_max > 0:
                hops.append(G.nodes[n]["h"] / h_max)
                powers.append(abs(p))
                pfs.append(G.nodes[n].get("pf" if sign > 0 else "pf_gen"))      # generation: its own power factor
    total = sum(powers)
    order = sorted(range(len(powers)), key=lambda i: -powers[i]) if total > 0 else []
    if any(f is None for f in pfs):                       # power factor not available for every node: do not use it
        return hops, [powers[i] / total for i in order], []
    return hops, [powers[i] / total for i in order], [float(pfs[i]) for i in order]


def fit_params_from_feeders(
    feeders: Sequence[nx.Graph],
    single_reference: bool = False,
) -> DistributionSynthParams:
    """Fit all Schweitzer parameters from a collection of reference feeders.

    Parameters
    ----------
    feeders : sequence of nx.Graph
        Reference distribution feeder graphs, each annotated with
        ``h``, ``P_mw``, ``node_type``, ``length_km`` attributes.
    single_reference : bool, default False
        Use this when there is only ONE reference feeder. With a single feeder the shares of empty nodes and of
        generation nodes cannot be fitted as distributions and silently fall back to defaults that come from the Dutch
        dataset of the original paper. With ``single_reference=True`` the measured values of the reference are used
        instead: the share of empty nodes, the share of generation nodes (nodes with negative net power), the split of
        the generation between them, and one constant cut-off for the cable length (the longest measured cable).
        It also records WHERE the load and generation nodes sit along the feeder and how large their shares are
        (``load_hops``, ``injection_hops``, ``load_shares``, ``injection_shares``) and their power factors
        (``load_pfs``, loads only): the generator then places them at these positions, splits the totals in these
        proportions and gives the loads these power factors, instead of the defaults of the paper (which put all power
        nodes at the far end, spread the power evenly and use the power factors of the Dutch dataset). The default (False) keeps the previous behaviour.

    Returns
    -------
    DistributionSynthParams
        Fitted parameter set ready for ``SchweetzerFeederGenerator``.
    """
    hops = _hop_distances(feeders)
    degs = _degrees(feeders)
    lengths = _cable_lengths(feeders)
    int_frac = _intermediate_fractions(feeders)
    inj_frac = _injection_fractions(feeders)
    load_dev = _load_deviations(feeders)

    params = DistributionSynthParams(
        hop_dist=fit_neg_binomial(hops) if len(hops) > 0 else NegBinomParams(),
        degree_dist=fit_mixture_gamma(degs) if len(degs) > 0 else MixtureGammaParams(),
        degree_clip=fit_power_law_clip(feeders),
        intermediate_frac=_fit_beta_or_default(int_frac, BetaParams(1.64, 15.77)),
        injection_frac=_fit_beta_or_default(inj_frac, BetaParams(0.92, 20.53)),
        load_deviation=fit_t_location_scale(load_dev) if len(load_dev) > 0 else TLocationScaleParams(),
        cable_length=fit_modified_cauchy(lengths) if len(lengths) > 0 else ModifiedCauchyParams(),
        length_clip=fit_exponential_clip(feeders),
    )
    if single_reference:
        params.intermediate_frac = _measured_beta(float(np.mean(int_frac)))
        params.injection_frac = _measured_beta(float(np.mean(_net_export_fractions(feeders))))
        inj_dev = _injection_deviations(feeders)
        if len(inj_dev) >= 3:
            params.injection_deviation = NormalParams(mu=float(inj_dev.mean()), sigma=float(inj_dev.std()))
        if len(lengths) >= 3:
            # The fitted maximum length per hop is unreliable when most connections are switches (few cables per hop).
            # Use one constant cut-off for all hops instead: the longest measured cable.
            params.length_clip = ExponentialClip(a=float(np.max(lengths)), b=0.0)
        params.load_hops, params.load_shares, params.load_pfs = _power_node_positions_and_shares(feeders, +1.0)
        # the generation uses its own power factor ("pf_gen", from the reactive power of the sgens); the "pf" of a bus belongs
        # to its LOAD and says nothing about the generation of an exporting bus
        params.injection_hops, params.injection_shares, params.injection_pfs = _power_node_positions_and_shares(feeders, -1.0)
        # an empty list (the reference has no such nodes) means "not measured": keep the paper's behaviour for it
        for name in ("load_hops", "load_shares", "load_pfs", "injection_hops", "injection_shares", "injection_pfs"):
            setattr(params, name, getattr(params, name) or None)
    return params
