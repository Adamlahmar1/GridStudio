"""
Convert a Schweitzer-format feeder graph (from ``pandapower_to_feeders`` or
``SchweetzerFeederGenerator.generate_feeder``) into a runnable pandapower
network, and check whether it's electrically feasible.

Topology/statistical validation (``distribution_validation.py``) checks
whether a synthetic feeder *looks* like a real one -- same hop/degree/cable-
length distributions. It says nothing about whether the feeder actually
*works* as a power system: within voltage limits, without overloaded lines,
solvable at all. That's what this module adds.
"""
from __future__ import annotations

from typing import Dict

import networkx as nx
import pandapower as pp

from .distribution_params import DEFAULT_CABLE_LIBRARY

_CABLE_C_NF_PER_KM = {c.name: c.c_nf_per_km for c in DEFAULT_CABLE_LIBRARY}


def feeder_to_pandapower(G: nx.Graph, sn_mva: float = 1.0) -> "pp.pandapowerNet":
    """Build a pandapower network from a Schweitzer-format feeder graph.

    The node at ``h == 0`` becomes the external grid connection (slack).
    Nodes are mapped 1:1 to pandapower bus indices, so results can be
    joined back onto the graph by node id.

    Cable ``c_nf_per_km`` isn't stored on synthetic-feeder edges (only
    r/x/max_i are); it's looked up from ``DEFAULT_CABLE_LIBRARY`` by
    ``cable_type`` name, which is a reasonable assumption for graphs
    produced by this package's own generator/converter, but callers using a
    custom cable library should be aware line charging capacitance will be
    wrong.
    """
    v_nom_kv = G.graph.get("v_nom_kv", 10.0)
    net = pp.create_empty_network(sn_mva=sn_mva)

    for n in G.nodes:
        pp.create_bus(net, vn_kv=v_nom_kv, name=str(n), index=n)

    root = next(n for n in G.nodes if G.nodes[n]["h"] == 0)
    pp.create_ext_grid(net, bus=root, vm_pu=1.0)

    for n, d in G.nodes(data=True):
        node_type = d.get("node_type")
        p_mw = d.get("P_mw", 0.0)
        q_mvar = d.get("Q_mvar", 0.0)
        if node_type == "load" and p_mw > 0:
            pp.create_load(net, bus=n, p_mw=p_mw, q_mvar=q_mvar)
        elif node_type == "injection":
            # Graph convention: injection P_mw/Q_mvar are negative.
            # pandapower sgen convention: positive p_mw = generation output.
            pp.create_sgen(net, bus=n, p_mw=-p_mw, q_mvar=-q_mvar)

    for u, v, d in G.edges(data=True):
        length_km = d.get("length_km", 0.001)
        r_per_km = d.get("r_per_km", 0.0)
        x_per_km = d.get("x_per_km", 0.0)
        c_nf_per_km = _CABLE_C_NF_PER_KM.get(d.get("cable_type"), 0.0)
        max_i_ka = d.get("max_i_ka", 1.0)
        pp.create_line_from_parameters(
            net, from_bus=u, to_bus=v, length_km=length_km,
            r_ohm_per_km=r_per_km, x_ohm_per_km=x_per_km,
            c_nf_per_km=c_nf_per_km, max_i_ka=max_i_ka,
        )

    return net


def check_feasibility(G: nx.Graph, sn_mva: float = 1.0) -> Dict:
    """Run a power flow on a feeder graph and summarize electrical feasibility.

    Returns a dict with ``converged``, and (if converged) ``vm_pu_min``,
    ``vm_pu_max``, ``max_loading_percent``, ``n_overloaded_lines``,
    ``n_undervoltage_buses`` (< 0.9 pu), ``n_overvoltage_buses`` (> 1.1 pu).
    """
    net = feeder_to_pandapower(G, sn_mva=sn_mva)
    try:
        pp.runpp(net)
    except Exception as exc:
        return {"converged": False, "error": f"{type(exc).__name__}: {exc}"}

    vm_pu = net.res_bus["vm_pu"]
    loading = net.res_line["loading_percent"]
    return {
        "converged": True,
        "vm_pu_min": float(vm_pu.min()),
        "vm_pu_max": float(vm_pu.max()),
        "max_loading_percent": float(loading.max()) if len(loading) else 0.0,
        "n_overloaded_lines": int((loading > 100.0).sum()),
        "n_undervoltage_buses": int((vm_pu < 0.9).sum()),
        "n_overvoltage_buses": int((vm_pu > 1.1).sum()),
    }
