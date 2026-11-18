# Author: Adam Lahmar <adam.lahmar@rwth-aachen.de>
"""Tests for the interactive feeder map (powergrid_synth.distribution.feeder_map) and the peak operating points."""
import json
import re

import numpy as np
import pandas as pd
import pytest

pp = pytest.importorskip("pandapower", reason="pandapower not installed")

from powergrid_synth.distribution.distribution_timeseries import peak_operating_points
from powergrid_synth.distribution.feeder_map import write_feeder_map


def _net():
    """110 kV grid -> transformer -> 20 kV busbar -> breaker -> two cables -> switch; a load and a generator."""
    net = pp.create_empty_network()
    hv = pp.create_bus(net, vn_kv=110.0)
    bb = pp.create_bus(net, vn_kv=20.0)
    b = [pp.create_bus(net, vn_kv=20.0) for _ in range(4)]
    pp.create_ext_grid(net, hv)
    pp.create_transformer_from_parameters(net, hv, bb, sn_mva=16.0, vn_hv_kv=110.0, vn_lv_kv=20.0, vkr_percent=0.5, vk_percent=10.0,
                                          pfe_kw=0.0, i0_percent=0.0)
    pp.create_switch(net, bb, b[0], et="b", closed=True)                       # breaker (at the transformer's busbar)
    pp.create_line_from_parameters(net, b[0], b[1], 0.5, 0.3, 0.1, 10.0, 0.3)
    pp.create_line_from_parameters(net, b[1], b[2], 0.4, 0.3, 0.1, 10.0, 0.3)
    pp.create_switch(net, b[2], b[3], et="b", closed=True)                     # ordinary switch
    pp.create_load(net, b[2], p_mw=1.0, q_mvar=0.3, agui_cod="P1", agui_type="pod")
    pp.create_sgen(net, b[3], p_mw=0.5, q_mvar=0.0, agui_cod="G1", agui_type="pod")
    return net


def _data(path):
    html = path.read_text(encoding="utf-8")
    assert "__DATA_JSON__" not in html
    return json.loads(re.search(r"const DATA = (\{.*?\});\nconst CASES", html, re.S).group(1))


def test_map_contains_both_networks_and_the_equipment(tmp_path):
    out = write_feeder_map(_net(), _net(), tmp_path / "sub" / "map.html", real_name="R", synthetic_name="S")
    d = _data(out)
    assert [s["label"] for s in d["scenarios"]] == ["Annual mean"]              # no time series: only the annual mean
    for key, name in (("real", "R"), ("synthetic", "S")):
        n = d["networks"][key]
        assert n["name"] == name
        st = n["stats"]
        assert (st["buses"], st["cables"], st["switches"], st["breakers"], st["transformers"]) == (6, 2, 1, 1, 1)
        kinds = {x["id"]: x["kind"] for x in n["nodes"]}
        assert kinds[n["source"]] == "source" and list(kinds.values()).count("busbar") == 1
        assert list(kinds.values()).count("load") == 1 and list(kinds.values()).count("generation") == 1
        assert all(v is not None and 0.9 < v <= 1.05 for x in n["nodes"] for v in x["vm"])


def test_map_edges_carry_the_data_of_the_cables_and_the_transformer(tmp_path):
    d = _data(write_feeder_map(_net(), _net(), tmp_path / "map.html"))
    edges = d["networks"]["real"]["edges"]
    line = next(e for e in edges if e["kind"] == "line")
    assert line["length_km"] in (0.5, 0.4) and line["i_max"] == 0.3 and 0 <= line["loading"][0] < 100
    trafo = next(e for e in edges if e["kind"] == "transformer")
    assert (trafo["sn_mva"], trafo["vn_hv"], trafo["vn_lv"]) == (16.0, 110.0, 20.0) and trafo["loading"][0] > 0


def test_map_distance_along_the_cables(tmp_path):
    d = _data(write_feeder_map(_net(), _net(), tmp_path / "map.html"))
    far = max(d["networks"]["real"]["nodes"], key=lambda n: n["dist"])
    assert far["dist"] == pytest.approx(0.9)                                    # 0.5 km + 0.4 km; switches and transformer have no length


def _timeseries(tmp_path, n=3000):
    t = pd.date_range("2024-01-01", periods=n, freq="5min")
    load = np.full(n, 2000.0)
    load[500:525] = 5000.0                                                      # the real peak: 25 steps
    load[1500:1511] = 18000.0                                                   # an hour of doubled readings (clock change)
    gen = np.full(n, 100.0); gen[2000:2010] = 1500.0
    pd.DataFrame({"time": t, "P1_A1": load, "P1_R1": 0.0, "G1_A2": gen, "G1_R2": 0.0}).to_csv(tmp_path / "load_timeseries_5min.csv", index=False)
    pd.DataFrame({"time": t}).to_csv(tmp_path / "transformer_timeseries_5min.csv", index=False)


def test_peak_operating_points_ignore_an_hour_of_doubled_readings(tmp_path):
    _timeseries(tmp_path)
    p = peak_operating_points(_net(), tmp_path)
    assert p["peak_load"]["load_mw"] == pytest.approx(5.0) and p["peak_load"]["time"].startswith("2024-01-02")      # not the 18 MW hour
    assert p["peak_generation"]["gen_mw"] == pytest.approx(1.5)


def test_map_with_time_series_has_three_operating_points(tmp_path):
    _timeseries(tmp_path)
    d = _data(write_feeder_map(_net(), _net(), tmp_path / "map.html", timeseries_dir=tmp_path))
    assert [s["key"] for s in d["scenarios"]] == ["mean", "peak_load", "peak_generation"]
    assert d["scenarios"][1]["label"].startswith("Peak load, 02 Jan 2024") and d["scenarios"][1]["load_mw"] == 5.0
    node = d["networks"]["real"]["nodes"][0]
    assert len(node["vm"]) == 3 and all(v is not None for v in node["vm"])


def test_map_survives_a_missing_time_series(tmp_path):
    d = _data(write_feeder_map(_net(), _net(), tmp_path / "map.html", timeseries_dir=tmp_path / "does_not_exist"))
    assert len(d["scenarios"]) == 1
