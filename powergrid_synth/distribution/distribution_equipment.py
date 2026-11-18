# Author: Adam Lahmar <adam.lahmar@rwth-aachen.de>
"""
Switches and upstream equipment for synthetic feeders (optional; the generation itself is not changed).

The generator creates a tree in which every connection is a cable. Real feeders also contain switches (in a typical
MV feeder most of the connections) and an upstream substation transformer. This module adds both:

* ``fit_switch_structure``   : from a reference feeder, the share of connections that are cables and how the switch share
                               depends on the degree of the parent node (hubs are connected by switches).
* ``assign_switch_edges``    : marks connections of a synthetic feeder as closed switches (``edge_type = "switch"``) so that the
                               number of cables matches the reference. ``feeder_to_pandapower`` turns them into bus-bus switches.
* ``upstream_from_net`` / ``add_upstream_equipment`` : copy the transformer of the reference network and put it, with a
                               circuit breaker, in front of the feeder (HV source bus -> transformer -> MV busbar -> breaker -> feeder).
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Dict, Optional

import networkx as nx
import numpy as np
import pandas as pd
import pandapower as pp

MAX_DEGREE_CLASS = 4          # parent degrees 4 and above are treated as one class

_CABLE_ATTRIBUTES = ("cable_type", "max_i_ka", "c_nf_per_km", "r_per_km", "x_per_km", "r_ohm", "x_ohm", "I_est_ka")
_TRAFO_KEYS = ("sn_mva", "vn_hv_kv", "vn_lv_kv", "vkr_percent", "vk_percent", "pfe_kw", "i0_percent",
               "shift_degree", "tap_side", "tap_neutral", "tap_min", "tap_max", "tap_step_percent",
               "tap_step_degree", "tap_pos", "vector_group", "vk0_percent", "vkr0_percent", "mag0_percent",
               "mag0_rx", "si0_hv_partial")


# ----------------------------------------------------------------------------------------------- cables
def cable_library_from_net(net, max_r_ratio: float = 20.0):
    """Cable library made of the cable types that really occur in a reference network.

    Lines with identical resistance, reactance, capacitance and current rating form one type; its ``frequency`` is its
    share of the total cable length. Types whose resistance is more than ``max_r_ratio`` times the (length-weighted)
    median are left out: they are placeholder lines, not cables (a real network can contain short lines with an unphysical resistance).
    The tool's default library consists of Dutch medium-voltage cables; the cables of a real network can have a clearly higher
    reactance and resistance, which makes the voltage drop of the synthetic feeder too small."""
    from .distribution_params import CableLibraryEntry
    lines = net.line[net.line.in_service] if "in_service" in net.line else net.line
    cols = ["r_ohm_per_km", "x_ohm_per_km", "c_nf_per_km", "max_i_ka"]
    key = lines[cols].round(3)
    groups = lines.groupby([key[c] for c in cols])["length_km"].sum().reset_index()
    order = groups["r_ohm_per_km"].to_numpy().argsort()
    cum = groups["length_km"].to_numpy()[order].cumsum()
    median_r = groups["r_ohm_per_km"].to_numpy()[order][np.searchsorted(cum, cum[-1] / 2.0)]
    groups = groups[groups["r_ohm_per_km"] <= max_r_ratio * median_r].sort_values("max_i_ka")
    total = float(groups["length_km"].sum())
    return [CableLibraryEntry(name=f"ref_{i}_{row.max_i_ka * 1000:.0f}A", r_ohm_per_km=float(row.r_ohm_per_km),
                              x_ohm_per_km=float(row.x_ohm_per_km), c_nf_per_km=float(row.c_nf_per_km),
                              max_i_ka=float(row.max_i_ka), frequency=float(row.length_km) / total)
            for i, row in enumerate(groups.itertuples())]


def max_current_ratio(net, feeder: nx.Graph, design_load_mw: float, v_nom_kv: float,
                      design_gen_mw: Optional[float] = None) -> float:
    """Highest ratio I_est / I_rated over the cables of a reference feeder, where I_est is the current the generator
    would estimate for the cable: the load downstream of it (loads only, scaled to ``design_load_mw``) divided by
    sqrt(3) * v_nom_kv. With ``design_gen_mw`` the generation case (generation only, scaled to ``design_gen_mw``) is
    evaluated as well and the larger current counts, as in ``generate_feeder``. Used as ``current_ratio_cap`` so that no
    synthetic cable is drawn smaller than the smallest margin of the reference."""
    rating = {frozenset((int(r.from_bus), int(r.to_bus))): float(r.max_i_ka) for r in net.line.itertuples()}
    total = sum(max(feeder.nodes[n].get("P_mw", 0.0), 0.0) for n in feeder.nodes)
    scale = design_load_mw / total if total > 0 else 1.0
    total_gen = sum(max(-feeder.nodes[n].get("P_mw", 0.0), 0.0) for n in feeder.nodes)
    scale_gen = design_gen_mw / total_gen if design_gen_mw is not None and total_gen > 0 else None
    down: Dict[int, float] = {}
    down_gen: Dict[int, float] = {}
    for n in sorted(feeder.nodes, key=lambda n: -feeder.nodes[n]["h"]):
        children = [m for m in feeder.neighbors(n) if feeder.nodes[m]["h"] > feeder.nodes[n]["h"]]
        down[n] = max(feeder.nodes[n].get("P_mw", 0.0), 0.0) * scale + sum(down[m] for m in children)
        if scale_gen is not None:
            down_gen[n] = max(-feeder.nodes[n].get("P_mw", 0.0), 0.0) * scale_gen + sum(down_gen[m] for m in children)
    ratios = []
    for u, v, d in feeder.edges(data=True):
        key = frozenset((u, v))
        if d.get("edge_type", "line") == "line" and key in rating and rating[key] > 0:
            child = v if feeder.nodes[v]["h"] > feeder.nodes[u]["h"] else u
            power = max(down[child], down_gen.get(child, 0.0))
            ratios.append(power / (np.sqrt(3) * v_nom_kv) / rating[key])
    return float(max(ratios)) if ratios else 1.0


# ----------------------------------------------------------------------------------------------- switches
def _parent_degree_class(G: nx.Graph, u: int, v: int) -> int:
    parent = u if G.nodes[u]["h"] <= G.nodes[v]["h"] else v
    return min(G.degree[parent], MAX_DEGREE_CLASS)


def fit_switch_structure(reference: nx.Graph) -> Dict:
    """Share of cables among the cable and switch connections of a reference feeder, and the switch share per
    degree class of the parent node. The transformer connection is not counted."""
    n_cable = n_switch = 0
    counts = {k: [0, 0] for k in range(1, MAX_DEGREE_CLASS + 1)}          # [switches, connections]
    for u, v, d in reference.edges(data=True):
        kind = d.get("edge_type", "line")
        if kind == "transformer":
            continue
        k = _parent_degree_class(reference, u, v)
        counts[k][1] += 1
        if kind == "switch":
            counts[k][0] += 1
            n_switch += 1
        else:
            n_cable += 1
    total = n_cable + n_switch
    return {"cable_fraction": n_cable / total if total else 1.0,
            "p_switch": {k: (s / t if t else 0.0) for k, (s, t) in counts.items()}}


def assign_switch_edges(G: nx.Graph, structure: Dict, rng: np.random.Generator) -> Dict[str, int]:
    """Turn some connections of a synthetic feeder into closed switches (in place).

    The number of cables is ``cable_fraction`` of the connections. Which connections stay cables is drawn without
    replacement with the weight ``1 - p_switch(parent degree class)``, so connections at hubs become switches, as in the
    reference. A switch gets the placeholder length of 1 m that the converter also uses for real switches, and loses its
    cable data.
    """
    edges = list(G.edges)
    n_cables = int(round(structure["cable_fraction"] * len(edges)))
    w = np.array([max(1.0 - structure["p_switch"][_parent_degree_class(G, u, v)], 0.0) for u, v in edges]) + 1e-3
    keep = set(int(i) for i in rng.choice(len(edges), size=n_cables, replace=False, p=w / w.sum()))
    for i, (u, v) in enumerate(edges):
        d = G.edges[u, v]
        if i in keep:
            d["edge_type"] = "line"
        else:
            d["edge_type"] = "switch"
            d["length_km"] = 0.001
            for key in _CABLE_ATTRIBUTES:
                d.pop(key, None)
    return {"n_cables": n_cables, "n_switches": len(edges) - n_cables}


# ----------------------------------------------------------------------------------------------- upstream equipment
@dataclass
class UpstreamEquipment:
    transformer: dict = field(default_factory=dict)      # keyword arguments of pp.create_transformer_from_parameters
    ext_vm_pu: float = 1.0                                # voltage of the external grid at the high-voltage bus


def _plain(value):
    return value.item() if hasattr(value, "item") else value


def upstream_from_net(net) -> Optional[UpstreamEquipment]:
    """Copy the (first) transformer and the external grid voltage of a reference network; None if it has no transformer."""
    if len(net.trafo) == 0:
        return None
    row = net.trafo.iloc[0]
    accepted = set(inspect.signature(pp.create_transformer_from_parameters).parameters)
    kwargs = {k: _plain(row[k]) for k in _TRAFO_KEYS
              if k in accepted and k in row.index and not pd.isna(row[k])}
    vm = float(net.ext_grid.vm_pu.iloc[0]) if len(net.ext_grid) else 1.0
    return UpstreamEquipment(kwargs, vm)


def add_upstream_equipment(net, equipment: UpstreamEquipment) -> Dict[str, int]:
    """Put the substation in front of the feeder: HV source bus -> transformer -> MV busbar -> circuit breaker -> feeder root.
    The external grid is moved from the feeder root to the new high-voltage bus."""
    ext = net.ext_grid.index[0]
    root = int(net.ext_grid.at[ext, "bus"])
    tr = equipment.transformer
    hv = pp.create_bus(net, vn_kv=tr["vn_hv_kv"], name="HV source")
    mv = pp.create_bus(net, vn_kv=tr["vn_lv_kv"], name="MV busbar")
    pp.create_transformer_from_parameters(net, hv_bus=hv, lv_bus=mv, name="substation transformer", **tr)
    pp.create_switch(net, bus=mv, element=root, et="b", closed=True, type="CB", name="feeder circuit breaker")
    net.ext_grid.at[ext, "bus"] = hv
    net.ext_grid.at[ext, "vm_pu"] = equipment.ext_vm_pu
    return {"hv_bus": hv, "mv_bus": mv, "feeder_root": root}
