# Author: Adam Lahmar <adam.lahmar@rwth-aachen.de>
"""Tests for the single-reference fit and the new comparison numbers (distribution_analysis / distribution_validation)."""
import networkx as nx
import numpy as np
import pytest

from powergrid_synth.distribution.distribution_analysis import fit_params_from_feeders
from powergrid_synth.distribution.distribution_synthesis import SchweetzerFeederGenerator
from powergrid_synth.distribution.distribution_validation import compare_cable_lengths, feeder_counts

DEFAULT_INTERMEDIATE = (1.64, 15.77)          # defaults of the original paper (BetaParams in the fit)


def _reference(n_nodes=80, seed=1):
    """A reference feeder with a clearly non-default share of empty nodes: only 10 nodes carry power."""
    G = SchweetzerFeederGenerator(seed=seed).generate_feeder(n_nodes=n_nodes, total_load_mw=2.0, total_gen_mw=0.5, v_nom_kv=20.0)
    powered = [n for n in G.nodes if n not in (0, 1)][:10]
    for n in G.nodes:
        G.nodes[n]["node_type"] = "load" if n in powered else "intermediate"
        G.nodes[n]["P_mw"] = 0.0
    for i, n in enumerate(powered):
        G.nodes[n]["P_mw"] = 0.4 if i < 7 else -0.3          # 7 consumers, 3 exporters
    return G


def _mean(beta):
    return beta.alpha / (beta.alpha + beta.beta)


def test_default_fit_with_one_feeder_uses_the_defaults():
    p = fit_params_from_feeders([_reference()])
    assert (p.intermediate_frac.alpha, p.intermediate_frac.beta) == DEFAULT_INTERMEDIATE


def test_single_reference_uses_the_measured_shares():
    G = _reference()
    p = fit_params_from_feeders([G], single_reference=True)
    n = G.number_of_nodes()
    assert _mean(p.intermediate_frac) == pytest.approx((n - 10) / n, abs=0.01)     # share of empty nodes
    assert _mean(p.injection_frac) == pytest.approx(3 / n, abs=0.01)               # share of exporting nodes


def test_single_reference_sets_one_constant_length_cutoff():
    G = _reference()
    longest = max(d["length_km"] for *_, d in G.edges(data=True))
    p = fit_params_from_feeders([G], single_reference=True)
    assert p.length_clip.a == pytest.approx(longest) and p.length_clip.b == 0.0


def test_generator_reproduces_the_number_of_power_buses():
    G = _reference(n_nodes=120, seed=2)
    p = fit_params_from_feeders([G], single_reference=True)
    counts = []
    for seed in range(5):
        syn = SchweetzerFeederGenerator(params=p, seed=seed).generate_feeder(
            n_nodes=120, total_load_mw=2.0, total_gen_mw=0.5, v_nom_kv=20.0)
        c = feeder_counts(syn)
        counts.append(c["load buses"] + c["generation buses"])
    assert 5 <= np.mean(counts) <= 20          # reference has 10 power buses; the default would give about 100


def test_feeder_counts():
    G = nx.Graph()
    for n, (h, p) in enumerate([(0, 0.0), (1, 0.0), (2, 1.5), (2, -0.5)]):
        G.add_node(n, h=h, P_mw=p)
    G.add_edge(0, 1, edge_type="switch", length_km=0.001)
    G.add_edge(1, 2, edge_type="line", length_km=0.4)
    G.add_edge(1, 3, edge_type="line", length_km=0.2)
    c = feeder_counts(G)
    assert (c["cables"], c["switches"], c["load buses"], c["generation buses"]) == (2, 1, 1, 1)
    assert c["total load (MW)"] == 1.5 and c["total generation (MW)"] == 0.5
    assert c["total cable length (km)"] == pytest.approx(0.6) and c["longest cable (m)"] == pytest.approx(400.0)


def test_compare_cable_lengths_ks_distance():
    def chain(lengths):
        G = nx.path_graph(len(lengths) + 1)
        for i, (u, v) in enumerate(G.edges):
            G.edges[u, v]["length_km"] = lengths[i]
        return G
    a, b = chain([0.1, 0.2, 0.3, 0.4]), chain([0.1, 0.2, 0.3, 0.4])
    far = chain([5.0, 6.0, 7.0, 8.0])
    assert compare_cable_lengths(a, b) == 0.0
    assert compare_cable_lengths(a, far) == 1.0
