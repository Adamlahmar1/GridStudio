# Author: Adam Lahmar <adam.lahmar@rwth-aachen.de>
"""Tests for the cable library of a reference network, the utilisation cap and the capacitance of custom cables."""
import networkx as nx
import numpy as np
import pytest

pp = pytest.importorskip("pandapower", reason="pandapower not installed")

from powergrid_synth.distribution.distribution_equipment import cable_library_from_net, max_current_ratio
from powergrid_synth.distribution.distribution_params import CableLibraryEntry, DistributionSynthParams
from powergrid_synth.distribution.distribution_powerflow import feeder_to_pandapower
from powergrid_synth.distribution.distribution_synthesis import SchweetzerFeederGenerator


def _net():
    net = pp.create_empty_network()
    b = [pp.create_bus(net, vn_kv=20.0) for _ in range(5)]
    spec = [(b[0], b[1], 1.0, 0.2, 0.15, 200.0, 0.33),     # (from, to, km, r, x, c, rating): 1.0 km of type A
            (b[1], b[2], 1.0, 0.2, 0.15, 200.0, 0.33),     # 1.0 km more of type A
            (b[2], b[3], 2.0, 0.9, 0.50, 7.0, 0.14),       # type B
            (b[3], b[4], 0.1, 150.0, 100.0, 4000.0, 0.33)]  # placeholder line, not a cable
    for f, t, km, r, x, c, i in spec:
        pp.create_line_from_parameters(net, f, t, km, r, x, c, i)
    return net


def test_library_groups_lines_by_type_and_weights_by_length():
    lib = {round(c.max_i_ka, 2): c for c in cable_library_from_net(_net())}
    assert set(lib) == {0.14, 0.33}                       # the placeholder line (150 Ohm/km) is left out
    assert lib[0.33].frequency == pytest.approx(0.5) and lib[0.14].frequency == pytest.approx(0.5)
    assert (lib[0.33].r_ohm_per_km, lib[0.33].x_ohm_per_km, lib[0.33].c_nf_per_km) == (0.2, 0.15, 200.0)
    assert [c.max_i_ka for c in cable_library_from_net(_net())] == sorted(c.max_i_ka for c in cable_library_from_net(_net()))


def test_max_current_ratio_of_a_chain():
    net = _net()
    G = nx.path_graph(5)
    for n in G.nodes:
        G.nodes[n].update(h=n, P_mw=0.0)
    G.nodes[4]["P_mw"] = 5.0                              # all load at the end of the chain
    for u, v in G.edges:
        G.edges[u, v]["edge_type"] = "line"
    # design load 6 MW at 20 kV: I = 6 / (sqrt(3) * 20) = 0.1732 kA through every cable; weakest cable is 0.14 kA
    assert max_current_ratio(net, G, 6.0, 20.0) == pytest.approx(0.1732 / 0.14, rel=1e-3)


def test_ratio_cap_limits_the_draw():
    def generate(cap):
        p = DistributionSynthParams(current_ratio_cap=cap)
        return SchweetzerFeederGenerator(params=p, seed=2).generate_feeder(
            n_nodes=150, total_load_mw=6.0, total_gen_mw=0.0, v_nom_kv=20.0)
    free, capped = generate(None), generate(0.1)
    # the cap does not use any random numbers, so both feeders are identical except for the cable choice;
    # a smaller maximum ratio means a larger target current, i.e. the same or a larger cable on every connection
    pairs = [(free.edges[e]["max_i_ka"], capped.edges[e]["max_i_ka"]) for e in free.edges]
    assert all(c >= f for f, c in pairs) and any(c > f for f, c in pairs)


def test_capacitance_of_a_custom_library_reaches_the_network():
    lib = [CableLibraryEntry("custom", 0.3, 0.2, 123.0, 1.0, 1.0)]
    G = SchweetzerFeederGenerator(params=DistributionSynthParams(cable_library=lib), seed=1).generate_feeder(
        n_nodes=30, total_load_mw=1.0, total_gen_mw=0.0, v_nom_kv=20.0)
    net = feeder_to_pandapower(G)
    assert set(net.line.c_nf_per_km) == {123.0}


def test_default_library_capacitance_is_unchanged():
    G = SchweetzerFeederGenerator(seed=1).generate_feeder(n_nodes=30, total_load_mw=1.0, v_nom_kv=20.0)
    net = feeder_to_pandapower(G)
    default = {c.name: c.c_nf_per_km for c in DistributionSynthParams().cable_library}
    assert sorted(net.line.c_nf_per_km) == sorted(default[d["cable_type"]] for *_, d in G.edges(data=True))
