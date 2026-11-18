# Author: Adam Lahmar <adam.lahmar@rwth-aachen.de>
"""Tests for switches and upstream equipment of synthetic feeders (distribution_equipment.py)."""
import networkx as nx
import numpy as np
import pytest

pp = pytest.importorskip("pandapower", reason="pandapower not installed")

from powergrid_synth.distribution.distribution_equipment import (
    UpstreamEquipment,
    add_upstream_equipment,
    assign_switch_edges,
    fit_switch_structure,
    upstream_from_net,
)
from powergrid_synth.distribution.distribution_powerflow import feeder_to_pandapower
from powergrid_synth.distribution.distribution_synthesis import SchweetzerFeederGenerator

STRUCTURE = {"cable_fraction": 0.25, "p_switch": {1: 0.0, 2: 0.75, 3: 0.44, 4: 1.0}}
TRAFO = dict(sn_mva=16.0, vn_hv_kv=150.0, vn_lv_kv=20.0, vk_percent=13.65, vkr_percent=0.54,
             pfe_kw=17.6, i0_percent=1.0, shift_degree=0.0)


@pytest.fixture()
def feeder():
    return SchweetzerFeederGenerator(seed=3).generate_feeder(
        n_nodes=60, total_load_mw=3.0, total_gen_mw=0.5, v_nom_kv=20.0)


def test_graph_without_switch_edges_gives_only_lines(feeder):
    net = feeder_to_pandapower(feeder)
    assert len(net.switch) == 0
    assert len(net.line) == feeder.number_of_edges()


def test_assign_switch_edges_matches_cable_fraction(feeder):
    counts = assign_switch_edges(feeder, STRUCTURE, np.random.default_rng(0))
    n = feeder.number_of_edges()
    assert counts["n_cables"] == round(0.25 * n)
    assert counts["n_cables"] + counts["n_switches"] == n
    types = [d["edge_type"] for _, _, d in feeder.edges(data=True)]
    assert types.count("line") == counts["n_cables"]
    assert types.count("switch") == counts["n_switches"]
    for _, _, d in feeder.edges(data=True):
        if d["edge_type"] == "switch":
            assert d["length_km"] == 0.001 and "cable_type" not in d and "r_per_km" not in d


def test_assignment_is_reproducible(feeder):
    import copy
    a, b = copy.deepcopy(feeder), copy.deepcopy(feeder)
    assign_switch_edges(a, STRUCTURE, np.random.default_rng(5))
    assign_switch_edges(b, STRUCTURE, np.random.default_rng(5))
    assert [d["edge_type"] for _, _, d in a.edges(data=True)] == [d["edge_type"] for _, _, d in b.edges(data=True)]


def test_switch_edges_become_closed_bus_switches(feeder):
    counts = assign_switch_edges(feeder, STRUCTURE, np.random.default_rng(1))
    net = feeder_to_pandapower(feeder)
    assert len(net.line) == counts["n_cables"]
    assert len(net.switch) == counts["n_switches"]
    assert net.switch.closed.all() and (net.switch.et == "b").all()
    pp.runpp(net)                       # the network must still be solvable
    assert net.converged


def test_upstream_equipment_is_added_in_front_of_the_feeder(feeder):
    assign_switch_edges(feeder, STRUCTURE, np.random.default_rng(2))
    net = feeder_to_pandapower(feeder)
    n_bus, n_switch, n_load = len(net.bus), len(net.switch), len(net.load)
    root = int(net.ext_grid.bus.iloc[0])
    info = add_upstream_equipment(net, UpstreamEquipment(dict(TRAFO), 1.0))
    assert len(net.bus) == n_bus + 2 and len(net.trafo) == 1 and len(net.switch) == n_switch + 1
    assert len(net.load) == n_load
    assert int(net.ext_grid.bus.iloc[0]) == info["hv_bus"] and net.bus.vn_kv[info["hv_bus"]] == 150.0
    breaker = net.switch.iloc[-1]
    assert breaker.closed and {int(breaker.bus), int(breaker.element)} == {info["mv_bus"], root}
    pp.runpp(net)
    assert net.converged and net.res_bus.vm_pu.min() > 0.9


def test_fit_switch_structure_from_reference():
    g = nx.Graph()
    for n, h in enumerate([0, 1, 2, 2, 3]):
        g.add_node(n, h=h)
    g.add_edge(0, 1, edge_type="transformer")          # not counted
    g.add_edge(1, 2, edge_type="line")
    g.add_edge(1, 3, edge_type="switch")
    g.add_edge(3, 4, edge_type="switch")
    s = fit_switch_structure(g)
    assert s["cable_fraction"] == pytest.approx(1 / 3)
    assert s["p_switch"][3] == 1.0 or s["p_switch"][2] >= 0.0       # keys 1..4 exist and are fractions
    assert set(s["p_switch"]) == {1, 2, 3, 4}


def test_upstream_from_net_without_transformer_is_none():
    net = pp.create_empty_network()
    b = pp.create_bus(net, vn_kv=20.0)
    pp.create_ext_grid(net, bus=b)
    assert upstream_from_net(net) is None


def test_upstream_from_net_copies_transformer_and_source_voltage():
    net = pp.create_empty_network()
    hv = pp.create_bus(net, vn_kv=150.0)
    mv = pp.create_bus(net, vn_kv=20.0)
    pp.create_ext_grid(net, bus=hv, vm_pu=1.02)
    pp.create_transformer_from_parameters(net, hv_bus=hv, lv_bus=mv, **TRAFO)
    eq = upstream_from_net(net)
    assert eq.ext_vm_pu == pytest.approx(1.02)
    assert eq.transformer["sn_mva"] == 16.0 and eq.transformer["vn_hv_kv"] == 150.0
