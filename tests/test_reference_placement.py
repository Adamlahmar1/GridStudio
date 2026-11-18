# Author: Adam Lahmar <adam.lahmar@rwth-aachen.de>
"""Tests for placing load and generation nodes like the reference (single_reference=True)."""
import math

import networkx as nx
import numpy as np
import pytest

pp = pytest.importorskip("pandapower", reason="pandapower not installed")

from powergrid_synth.distribution.distribution_analysis import fit_params_from_feeders
from powergrid_synth.distribution.distribution_converter import pandapower_to_feeders
from powergrid_synth.distribution.distribution_synthesis import SchweetzerFeederGenerator


def _reference():
    """Chain of 40 nodes (hop 0..39): consumers near the start (one dominant), exporters in the middle."""
    G = nx.path_graph(40)
    for n in G.nodes:
        G.nodes[n].update(h=n, P_mw=0.0, pf=0.95, node_type="intermediate")
    for n, p, pf in ((3, 8.0, 0.80), (5, 0.3, 0.90), (7, 0.2, 0.90), (9, 0.1, 0.95)):
        G.nodes[n].update(P_mw=p, pf=pf, node_type="load")
    for n, p in ((20, -3.0), (22, -0.2), (24, -0.1)):
        G.nodes[n].update(P_mw=p, node_type="injection", pf_gen=0.99)
    for u, v in G.edges:
        G.edges[u, v]["length_km"] = 0.2
    return G


def _generate(params, seed=0):
    return SchweetzerFeederGenerator(params=params, seed=seed).generate_feeder(
        n_nodes=60, total_load_mw=8.6, total_gen_mw=3.3, v_nom_kv=20.0)


def _nodes(G, sign):
    return [n for n in G.nodes if G.nodes[n].get("P_mw", 0.0) * sign > 0]


def test_fit_measures_positions_shares_and_power_factors():
    p = fit_params_from_feeders([_reference()], single_reference=True)
    assert sorted(p.load_hops) == pytest.approx([3 / 39, 5 / 39, 7 / 39, 9 / 39])
    assert p.load_shares[0] == pytest.approx(8.0 / 8.6) and p.load_shares == sorted(p.load_shares, reverse=True)
    assert p.load_pfs[0] == pytest.approx(0.80)                       # power factor of the dominant load
    assert p.injection_shares[0] == pytest.approx(3.0 / 3.3)
    assert len(p.injection_hops) == 3


def test_default_fit_has_no_reference_placement():
    p = fit_params_from_feeders([_reference()])
    assert p.load_hops is None and p.injection_shares is None and p.load_pfs is None


def test_generator_keeps_the_dominant_load_and_totals():
    p = fit_params_from_feeders([_reference()], single_reference=True)
    for seed in range(5):
        G = _generate(p, seed)
        loads = sorted((G.nodes[n]["P_mw"] for n in _nodes(G, +1)), reverse=True)
        gens = sorted((-G.nodes[n]["P_mw"] for n in _nodes(G, -1)), reverse=True)
        assert sum(loads) == pytest.approx(8.6) and sum(gens) == pytest.approx(3.3)
        # reference: 93 % / 91 %; extra nodes take the median share, which dilutes the dominant share a little
        assert loads[0] / sum(loads) > 0.8 and gens[0] / sum(gens) > 0.8


def test_generator_places_power_nodes_near_the_reference_positions():
    p = fit_params_from_feeders([_reference()], single_reference=True)
    G = _generate(p)
    h_max = max(G.nodes[n]["h"] for n in G.nodes)
    mean_pos = np.mean([G.nodes[n]["h"] / h_max for n in _nodes(G, +1)])
    assert mean_pos < 0.4                 # reference loads sit at 0.08-0.23 of the feeder; the paper's rule gives about 0.9


def test_load_power_factor_is_used_for_reactive_power():
    p = fit_params_from_feeders([_reference()], single_reference=True)
    G = _generate(p)
    big = max(_nodes(G, +1), key=lambda n: G.nodes[n]["P_mw"])
    assert G.nodes[big]["Q_mvar"] / G.nodes[big]["P_mw"] == pytest.approx(math.tan(math.acos(0.80)))


def test_source_node_stays_intermediate_and_types_are_consistent():
    G = _generate(fit_params_from_feeders([_reference()], single_reference=True))
    assert G.nodes[0]["node_type"] == "intermediate" and G.nodes[0]["P_mw"] == 0.0
    for n in G.nodes:
        t, p = G.nodes[n]["node_type"], G.nodes[n]["P_mw"]
        assert (t == "intermediate") == (p == 0.0)
        assert (t == "load") == (p > 0.0) and (t == "injection") == (p < 0.0)


def test_paper_algorithm_unchanged_without_reference_placement():
    a = SchweetzerFeederGenerator(seed=4).generate_feeder(n_nodes=60, total_load_mw=2.0, total_gen_mw=1.0, v_nom_kv=20.0)
    b = SchweetzerFeederGenerator(seed=4).generate_feeder(n_nodes=60, total_load_mw=2.0, total_gen_mw=1.0, v_nom_kv=20.0)
    assert [a.nodes[n]["P_mw"] for n in sorted(a)] == [b.nodes[n]["P_mw"] for n in sorted(b)]


def test_fit_records_the_power_factor_of_the_generation():
    p = fit_params_from_feeders([_reference()], single_reference=True)
    assert p.injection_pfs == [0.99, 0.99, 0.99]
    assert p.load_pfs[0] == pytest.approx(0.80)                       # the loads keep their own power factors


def test_generators_use_the_measured_power_factor():
    p = fit_params_from_feeders([_reference()], single_reference=True)
    G = _generate(p)
    for n in _nodes(G, -1):
        assert G.nodes[n]["Q_mvar"] / G.nodes[n]["P_mw"] == pytest.approx(math.tan(math.acos(0.99)))


def test_converter_stores_the_power_factor_of_the_generation_separately():
    net = pp.create_empty_network()
    buses = [pp.create_bus(net, vn_kv=20.0) for _ in range(4)]
    pp.create_ext_grid(net, buses[0])
    for a, b in zip(buses, buses[1:]):
        pp.create_line_from_parameters(net, a, b, 1.0, 0.2, 0.1, 0.0, 0.3)
    pp.create_load(net, buses[2], p_mw=1.0, q_mvar=0.5)
    pp.create_sgen(net, buses[3], p_mw=2.0, q_mvar=0.2)
    G = pandapower_to_feeders(net)[0]
    assert G.nodes[buses[3]]["pf_gen"] == pytest.approx(2.0 / math.sqrt(2.0 ** 2 + 0.2 ** 2))
    assert "pf_gen" not in G.nodes[buses[2]]                           # a bus without generation has none
    assert G.nodes[buses[2]]["pf"] == pytest.approx(1.0 / math.sqrt(1.0 + 0.25))
