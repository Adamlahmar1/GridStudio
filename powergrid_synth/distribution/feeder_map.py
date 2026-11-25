# Author: Adam Lahmar <adam.lahmar@rwth-aachen.de>
"""
Interactive map of a real and a synthetic feeder side by side (one self-contained HTML page).

    write_feeder_map(real_net, synthetic_net, "output/synthetic_italian_dso_map.html", timeseries_dir=...)

The page shows both networks (switch in the page) with the equipment drawn explicitly

    grid connection -> transformer -> busbar -> breaker -> feeder with cables (solid) and switches (dashed)

and, for up to three operating points, the bus voltages and the cable / transformer loadings from an AC power flow:

    annual mean       the network as it is (annual-mean load and generation)
    peak load         the moment of the highest load in the time series (needs ``timeseries_dir``)
    peak generation   the moment of the highest generation in the time series (needs ``timeseries_dir``)

Layout: x = hop distance from the source (BFS) or the distance along the cables in km (switch in the page), y = tidy-tree
order of the BFS spanning tree, so siblings do not overlap. The page contains all its data and needs no server and no
internet connection; it contains the data of the networks you give it, so treat it like the data itself.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Optional

import networkx as nx
import pandapower as pp
import pandas as pd

from .distribution_timeseries import peak_operating_points

TEMPLATE_PATH = Path(__file__).with_name("feeder_map_template.html")


def _tidy_layout(tree: nx.DiGraph, root) -> dict:
    """y-position of every node: leaves get consecutive slots, inner nodes are centred over their children (largest branch first)."""
    children = {n: list(tree.successors(n)) for n in tree.nodes}
    size = {}
    for n in nx.dfs_postorder_nodes(tree, root):
        size[n] = 1 + sum(size[c] for c in children[n])
    for n in children:
        children[n].sort(key=lambda c: -size[c])
    y, slot = {}, [0]

    def assign(n):
        if not children[n]:
            y[n] = float(slot[0])
            slot[0] += 1
            return y[n]
        y[n] = sum(assign(c) for c in children[n]) / len(children[n])
        return y[n]

    assign(root)
    return y


def _solve(net, point):
    """Power-flow results of one operating point: voltage per bus, loading per line and per transformer, cable losses.
    ``None`` entries if the power flow does not converge."""
    n = copy.deepcopy(net)
    if point is not None:
        if len(n.load) and n.load.p_mw.sum() > 0:
            n.load[["p_mw", "q_mvar"]] = n.load[["p_mw", "q_mvar"]] * (point[0] / n.load.p_mw.sum())
        if len(n.sgen) and n.sgen.p_mw.sum() > 0:
            n.sgen[["p_mw", "q_mvar"]] = n.sgen[["p_mw", "q_mvar"]] * (point[1] / n.sgen.p_mw.sum())
    try:
        pp.runpp(n, numba=False)
    except Exception:
        return None
    return (n.res_bus.vm_pu.to_dict(), n.res_line.loading_percent.to_dict(),
            n.res_trafo.loading_percent.to_dict() if len(n.trafo) else {}, float(n.res_line.pl_mw.sum()),
            n.res_line.i_ka.to_dict(), n.res_line.pl_mw.to_dict())


def _build_network(net, name: str, points: list) -> dict:
    """Nodes, edges and statistics of one network for the page."""
    G = nx.Graph()
    G.add_nodes_from(int(b) for b in net.bus.index)
    edges = []
    for i, r in net.line.iterrows():
        edges.append(dict(kind="line", a=int(r.from_bus), b=int(r.to_bus), id=int(i), length_km=float(r.length_km),
                          r=float(r.r_ohm_per_km), x=float(r.x_ohm_per_km), c=float(r.c_nf_per_km), i_max=float(r.max_i_ka)))
    for i, r in net.trafo.iterrows():
        edges.append(dict(kind="transformer", a=int(r.hv_bus), b=int(r.lv_bus), id=int(i), sn_mva=float(r.sn_mva),
                          vn_hv=float(r.vn_hv_kv), vn_lv=float(r.vn_lv_kv), vk=float(r.vk_percent), vkr=float(r.vkr_percent)))
    trafo_lv = {e["b"] for e in edges if e["kind"] == "transformer"}
    for i, r in net.switch.iterrows():
        if r.et != "b" or not bool(r.closed):
            continue
        a, b = int(r.bus), int(r.element)
        kind = "breaker" if (a in trafo_lv or b in trafo_lv) else "switch"     # a switch directly at the transformer = feeder breaker
        edges.append(dict(kind=kind, a=a, b=b, id=int(i)))
    for e in edges:
        G.add_edge(e["a"], e["b"])

    source = int(net.ext_grid.bus.iloc[0])
    tree = nx.bfs_tree(G, source)
    hop = nx.single_source_shortest_path_length(G, source)
    y = _tidy_layout(tree, source)
    seg = {frozenset((e["a"], e["b"])): e.get("length_km", 0.0) for e in edges}
    dist = {source: 0.0}                                    # distance along the cables (switches and the transformer have no length)
    for u, v in nx.bfs_edges(G, source):
        dist[v] = dist[u] + seg[frozenset((u, v))]

    res = [_solve(net, p) for p in points]
    load_at = net.load.groupby("bus").p_mw.sum().to_dict()
    gen_at = net.sgen.groupby("bus").p_mw.sum().to_dict() if len(net.sgen) else {}

    def vm(r, b):
        return None if r is None else round(float(r[0][b]), 6)

    nodes = []
    for b in G.nodes:
        pl, pg = float(load_at.get(b, 0.0)), float(gen_at.get(b, 0.0))
        kind = ("source" if b == source else "busbar" if b in trafo_lv else "prosumer" if pl > 0 and pg > 0
                else "load" if pl > 0 or b in load_at else "generation" if pg > 0 or b in gen_at else "junction")
        nodes.append(dict(id=int(b), hop=int(hop[b]), dist=round(dist[b], 6), y=round(y[b], 3), kind=kind, vn=float(net.bus.vn_kv[b]),
                          deg=int(G.degree[b]), p_load=round(pl, 6), p_gen=round(pg, 6), vm=[vm(r, b) for r in res],
                          parent=(int(next(iter(tree.predecessors(b)))) if b != source else None)))
    for e in edges:
        if e["kind"] == "line":
            e["loading"] = [None if r is None else round(float(r[1][e["id"]]), 3) for r in res]
            e["cur_a"] = [None if r is None else round(float(r[4][e["id"]]) * 1000.0, 3) for r in res]       # current in A
            e["loss_kw"] = [None if r is None else round(float(r[5][e["id"]]) * 1000.0, 4) for r in res]     # losses of the cable in kW
        elif e["kind"] == "transformer":
            e["loading"] = [None if r is None else round(float(r[2][e["id"]]), 3) for r in res]
    cables = [e for e in edges if e["kind"] == "line"]
    stats = dict(buses=len(nodes), cables=len(cables), switches=sum(e["kind"] == "switch" for e in edges),
                 breakers=sum(e["kind"] == "breaker" for e in edges), transformers=sum(e["kind"] == "transformer" for e in edges),
                 loads=sum(n["kind"] == "load" for n in nodes), prosumers=sum(n["kind"] == "prosumer" for n in nodes),
                 generation=sum(n["kind"] == "generation" for n in nodes), cable_km=round(sum(e["length_km"] for e in cables), 5),
                 max_hop=max(hop.values()), load_mw=round(float(net.load.p_mw.sum()), 4), gen_mw=round(float(net.sgen.p_mw.sum()) if len(net.sgen) else 0.0, 4),
                 losses_mw=[None if r is None else round(r[3], 6) for r in res])
    return dict(name=name, source=source, nodes=nodes, edges=edges, stats=stats)


def _scenarios(real_net, timeseries_dir: Optional[Path]):
    """Operating points (labels and total load / generation); the first one is the network as it is."""
    scen = [{"key": "mean", "label": "Annual mean", "detail": "annual-mean load and generation"}]
    points = [None]
    if timeseries_dir is None:
        return scen, points
    try:
        peaks = peak_operating_points(real_net, timeseries_dir)
    except Exception:                                           # missing or unreadable time series: only the annual mean
        return scen, points
    for key, title, detail in (("peak_load", "Peak load", "highest load in the data"), ("peak_generation", "Peak generation", "highest generation in the data")):
        p = peaks[key]
        t = pd.to_datetime(p["time"], errors="coerce")
        when = f"{t:%d %b %Y %H:%M}" if not pd.isna(t) else str(p["time"])
        scen.append({"key": key, "label": f"{title}, {when}", "detail": detail, "load_mw": round(p["load_mw"], 4), "gen_mw": round(p["gen_mw"], 4)})
        points.append((p["load_mw"], p["gen_mw"]))
    return scen, points


def write_feeder_map(real_net, synthetic_net, out_path, timeseries_dir=None, real_name: str = "Real feeder",
                     synthetic_name: str = "Synthetic feeder") -> Path:
    """Write the interactive map page; returns its path. ``real_net`` should carry the annual-mean power (see
    ``apply_real_power_snapshot``); ``timeseries_dir`` adds the peak-load and peak-generation operating points."""
    scen, points = _scenarios(real_net, Path(timeseries_dir) if timeseries_dir is not None else None)
    data = dict(scenarios=scen, networks=dict(real=_build_network(real_net, real_name, points),
                                              synthetic=_build_network(synthetic_net, synthetic_name, points)))
    html = TEMPLATE_PATH.read_text(encoding="utf-8").replace("__DATA_JSON__", json.dumps(data, separators=(",", ":")))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def write_multi_feeder_map(cases, out_path) -> Path:
    """Write ONE interactive page for several feeders: the page has a selector "Feeder" at the top left, and for every feeder the same views as
    ``write_feeder_map`` (real or synthetic network, operating point, colouring, axis).

    ``cases`` is a list of dicts with ``label`` (name in the selector), ``real_net``, ``synthetic_net`` and optionally ``real_name``, ``synthetic_name``,
    ``mean_label`` / ``mean_detail`` (name of the first operating point = the networks as they are) and ``scenarios``: a list of further operating points,
    each a dict with ``label``, ``detail``, ``load_mw`` and ``gen_mw`` (total load and generation of the moment, as returned by ``_scenarios``)."""
    out_cases = []
    for c in cases:
        scen = [{"key": "mean", "label": c.get("mean_label", "Annual mean"),
                 "detail": c.get("mean_detail", "annual-mean load and generation")}]
        points = [None]
        for i, s in enumerate(c.get("scenarios") or []):
            scen.append({"key": s.get("key", f"point{i}"), "label": s["label"], "detail": s.get("detail", ""),
                         "load_mw": round(float(s["load_mw"]), 4), "gen_mw": round(float(s["gen_mw"]), 4)})
            points.append((float(s["load_mw"]), float(s["gen_mw"])))
        out_cases.append(dict(label=c["label"], scenarios=scen, networks=dict(
            real=_build_network(c["real_net"], c.get("real_name", "Real feeder"), points),
            synthetic=_build_network(c["synthetic_net"], c.get("synthetic_name", "Synthetic feeder"), points))))
    html = TEMPLATE_PATH.read_text(encoding="utf-8").replace("__DATA_JSON__", json.dumps(dict(cases=out_cases), separators=(",", ":")))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out
