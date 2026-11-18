# Author: Adam Lahmar <adam.lahmar@rwth-aachen.de>
"""Tests for sizing the cables for the design load (generate_feeder(design_load_mw=...)) and the design load itself."""
import numpy as np
import pandas as pd
import pytest

pp = pytest.importorskip("pandapower", reason="pandapower not installed")

from powergrid_synth.distribution.distribution_synthesis import SchweetzerFeederGenerator
from powergrid_synth.distribution.distribution_timeseries import design_load_mw


def _generate(**kw):
    return SchweetzerFeederGenerator(seed=3).generate_feeder(
        n_nodes=120, total_load_mw=4.0, total_gen_mw=2.0, v_nom_kv=20.0, **kw)


def _capacity(G):
    return sum(d["max_i_ka"] for *_, d in G.edges(data=True))


def test_default_is_unchanged():
    a, b = _generate(), _generate(design_load_mw=None)
    assert [d["cable_type"] for *_, d in a.edges(data=True)] == [d["cable_type"] for *_, d in b.edges(data=True)]


def test_design_load_gives_stronger_cables():
    assert _capacity(_generate(design_load_mw=10.0)) > _capacity(_generate())


def test_design_load_does_not_change_the_nodes():
    a, b = _generate(), _generate(design_load_mw=10.0)
    assert [a.nodes[n]["P_mw"] for n in sorted(a.nodes)] == [b.nodes[n]["P_mw"] for n in sorted(b.nodes)]


def test_design_current_estimate_at_the_root():
    """The estimated current of the cable at the start is design load / (sqrt(3) * voltage)."""
    G = _generate(design_load_mw=10.0)
    root_edges = [d for u, v, d in G.edges(data=True) if min(G.nodes[u]["h"], G.nodes[v]["h"]) == 0]
    assert max(d["I_est_ka"] for d in root_edges) == pytest.approx(10.0 / (np.sqrt(3) * 20.0), rel=1e-6)


def test_design_load_quantile_ignores_one_outlier(tmp_path):
    n = 1000
    t = np.full(n, 2000.0)           # kW
    t[10] = 40000.0                  # one wrong reading
    pd.DataFrame({"time": range(n), "P1_A1": t, "P1_R1": 0.0}).to_csv(tmp_path / "load_timeseries_5min.csv", index=False)
    pd.DataFrame({"time": range(n), "T1_A": 0.5, "T1_R": 0.0}).to_csv(tmp_path / "transformer_timeseries_5min.csv", index=False)
    net = pp.create_empty_network()
    b = pp.create_bus(net, vn_kv=20.0)
    pp.create_load(net, b, p_mw=0.0, agui_cod="P1", agui_type="pod")
    pp.create_load(net, b, p_mw=0.0, agui_cod="T1", agui_type="transformer")
    assert design_load_mw(net, tmp_path) == pytest.approx(2.5)
    assert design_load_mw(net, tmp_path, quantile=1.0) == pytest.approx(40.5)


def test_design_generation_gives_stronger_cables_than_load_only():
    load_only = _generate(design_load_mw=4.0)
    both = _generate(design_load_mw=4.0, design_gen_mw=12.0)       # generation much larger than the load case
    assert _capacity(both) > _capacity(load_only)


def test_design_generation_never_makes_cables_smaller():
    load_only = _generate(design_load_mw=10.0)
    both = _generate(design_load_mw=10.0, design_gen_mw=0.5)       # small generation: the load case still decides
    assert [d["cable_type"] for *_, d in load_only.edges(data=True)] == [d["cable_type"] for *_, d in both.edges(data=True)]


def test_design_generation_current_estimate_at_the_root():
    G = _generate(design_load_mw=4.0, design_gen_mw=12.0)
    root_edges = [d for u, v, d in G.edges(data=True) if min(G.nodes[u]["h"], G.nodes[v]["h"]) == 0]
    assert max(d["I_est_ka"] for d in root_edges) == pytest.approx(12.0 / (np.sqrt(3) * 20.0), rel=1e-6)


def test_design_generation_quantile(tmp_path):
    n = 1000
    pd.DataFrame({"time": range(n), "P1_A1": 0.0, "P1_R1": 0.0, "P1_A2": 1500.0, "P1_R2": 0.0}).to_csv(
        tmp_path / "load_timeseries_5min.csv", index=False)
    t = np.full(n, -0.5); t[10] = -30.0                            # transformer: negative = export, one wrong reading
    pd.DataFrame({"time": range(n), "T1_A": t, "T1_R": 0.0}).to_csv(tmp_path / "transformer_timeseries_5min.csv", index=False)
    net = pp.create_empty_network()
    b = pp.create_bus(net, vn_kv=20.0)
    pp.create_sgen(net, b, p_mw=0.0, agui_cod="P1", agui_type="pod")
    pp.create_sgen(net, b, p_mw=0.0, agui_cod="T1", agui_type="transformer")
    from powergrid_synth.distribution.distribution_timeseries import design_generation_mw
    assert design_generation_mw(net, tmp_path) == pytest.approx(2.0)              # 1.5 MW pod + 0.5 MW transformer export
    assert design_generation_mw(net, tmp_path, quantile=1.0) == pytest.approx(31.5)


def test_design_values_count_a_device_without_readings_as_zero(tmp_path):
    n = 1000
    a1 = np.full(n, 2000.0); a1[:600] = np.nan                  # one load has no values for the first 600 steps
    pd.DataFrame({"time": range(n), "P1_A1": 3000.0, "P1_R1": 0.0, "P2_A1": a1, "P2_R1": 0.0}).to_csv(
        tmp_path / "load_timeseries_5min.csv", index=False)
    pd.DataFrame({"time": range(n), "T1_A": 0.0, "T1_R": 0.0}).to_csv(tmp_path / "transformer_timeseries_5min.csv", index=False)
    net = pp.create_empty_network()
    b = pp.create_bus(net, vn_kv=20.0)
    pp.create_load(net, b, p_mw=0.0, agui_cod="P1", agui_type="pod")
    pp.create_load(net, b, p_mw=0.0, agui_cod="P2", agui_type="pod")
    assert design_load_mw(net, tmp_path, quantile=1.0) == pytest.approx(5.0)       # 3 + 2 MW, the late steps
    assert design_load_mw(net, tmp_path, quantile=0.0) == pytest.approx(3.0)       # the first 600 steps are not dropped
