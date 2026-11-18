# Author: Adam Lahmar <adam.lahmar@rwth-aachen.de>
"""Real ENEL feeder against the synthetic feeders, with the same electrical test for all of them.

    python compare_real_fake.py --seeds 10 --out report_rows.json

For every seed the script generates two synthetic feeders and measures structure, cables, similarity to the real feeder, the
annual-mean power flow and one real day (288 steps of 5 minutes, loads and generation scaled to the measured totals):
  * "original fake": the version of pipeline.py with all new options switched off (--feeder-only --default-fit --mean-sizing --default-cables)
  * "new fake"     : the default of pipeline.py (switches + transformer, measured positions/sizes/power factors, cable types of the
                     real network, design sizing for the busiest and the sunniest moment, utilisation cap)
and prints a short table. The numbers in REPORT_enel_real_vs_fake.md come from this script (10 seeds); it needs the ENEL data
(--enel-data-dir or the environment variable ENEL_DATA_DIR).
"""
import argparse
import os
import copy
import json
import warnings
from pathlib import Path

import numpy as np
import pandapower as pp

warnings.filterwarnings("ignore")
from feeder_pipeline.loaders import load_enel
from pipeline import DEFAULT_ENEL_DATA_DIR
from powergrid_synth.distribution.distribution_analysis import fit_params_from_feeders
from powergrid_synth.distribution.distribution_converter import pandapower_to_feeders
from powergrid_synth.distribution.distribution_equipment import (
    add_upstream_equipment, assign_switch_edges, cable_library_from_net, fit_switch_structure, max_current_ratio,
    upstream_from_net)
from powergrid_synth.distribution.distribution_powerflow import feeder_to_pandapower
from powergrid_synth.distribution.distribution_synthesis import SchweetzerFeederGenerator
from powergrid_synth.distribution.distribution_timeseries import design_generation_mw, design_load_mw
from powergrid_synth.distribution.distribution_validation import (
    compare_cable_lengths, compare_feeders, compute_emergent_properties, feeder_counts)

ap = argparse.ArgumentParser()
ap.add_argument("--enel-data-dir", default=os.environ.get("ENEL_DATA_DIR", DEFAULT_ENEL_DATA_DIR))
ap.add_argument("--seeds", type=int, default=10)
ap.add_argument("--out", default="report_rows.json", help="per-seed numbers as JSON")
_args = ap.parse_args()
DEFAULT_ENEL_DATA_DIR = _args.enel_data_dir
N_SEEDS = _args.seeds
OUT = Path(_args.out)

ref = load_enel(DEFAULT_ENEL_DATA_DIR)
real_net = ref.net
feeders = pandapower_to_feeders(real_net)
real = feeders[0]
em = compute_emergent_properties(real)
profile = ref.day
design = design_load_mw(real_net, Path(DEFAULT_ENEL_DATA_DIR) / "timeseries_data")
design_gen = design_generation_mw(real_net, Path(DEFAULT_ENEL_DATA_DIR) / "timeseries_data")
equipment = upstream_from_net(real_net)


def feeder_part(net, v=20.0):
    return [int(b) for b in net.bus.index if abs(net.bus.vn_kv[b] - v) / v <= 0.05]


def concentration(G):
    P = {n: G.nodes[n].get("P_mw", 0.0) for n in G}
    H = max(G.nodes[n]["h"] for n in G)
    L = sorted(((p, n) for n, p in P.items() if p > 0), reverse=True)
    Gn = sorted(((-p, n) for n, p in P.items() if p < 0), reverse=True)
    tl, tg = sum(p for p, _ in L), sum(p for p, _ in Gn)
    pos = lambda X: float(np.mean([G.nodes[n]["h"] / H for _, n in X])) if X else float("nan")
    q = sum(G.nodes[n].get("Q_mvar", 0.0) for _, n in L)
    return {"largest load, share of load": L[0][0] / tl, "largest generator, share of generation": Gn[0][0] / tg if Gn else float("nan"),
            "mean position of load buses (0=source, 1=far end)": pos(L), "mean position of generation buses": pos(Gn),
            "highest hop distance": float(H)}


def electrics(net):
    l = net.line
    km = l.length_km
    ok = l.r_ohm_per_km < 20 * float(np.median(l.r_ohm_per_km))                       # ignore the 2 placeholder lines of the real network
    w = lambda c: float((l[c][ok] * km[ok]).sum() / km[ok].sum())
    return {"cable resistance (Ohm/km, length-weighted)": w("r_ohm_per_km"), "cable reactance (Ohm/km)": w("x_ohm_per_km"),
            "cable rating (kA, length-weighted)": w("max_i_ka")}


def snapshot(net, head, buses):
    pp.runpp(net, numba=False)
    vm = net.res_bus.vm_pu
    return {"annual mean: lowest voltage (pu)": float(vm[buses].min()),
            "annual mean: voltage drop (%)": 100 * float(vm[head] - vm[buses].min()),
            "annual mean: highest cable loading (%)": float(net.res_line.loading_percent.max()),
            "annual mean: cable losses (MW)": float(net.res_line.pl_mw.sum()),
            "annual mean: transformer loading (%)": float(net.res_trafo.loading_percent.max()) if len(net.trafo) else float("nan")}


def day(net, head, buses):
    bl, bg = net.load[["p_mw", "q_mvar"]].copy(), net.sgen[["p_mw", "q_mvar"]].copy()
    p_l, p_g = float(bl.p_mw.sum()), float(bg.p_mw.sum())
    out = []
    for _, l, g in profile:
        net.load[["p_mw", "q_mvar"]] = bl * (l / p_l)
        net.sgen[["p_mw", "q_mvar"]] = bg * (g / p_g)
        try:
            pp.runpp(net, numba=False)
        except Exception:
            continue
        vm = net.res_bus.vm_pu
        out.append((100 * float(vm[head] - vm[buses].min()), float(vm[buses].min()), float(net.res_line.loading_percent.max()),
                    float(net.res_trafo.loading_percent.max()) if len(net.trafo) else float("nan"), float(net.res_line.pl_mw.sum())))
    net.load[["p_mw", "q_mvar"]], net.sgen[["p_mw", "q_mvar"]] = bl, bg
    a = np.array(out)
    return {"day: converged steps (of 288)": float(len(a)), "day: max voltage drop (%)": a[:, 0].max(), "day: lowest voltage (pu)": a[:, 1].min(),
            "day: highest cable loading (%)": a[:, 2].max(), "day: steps with a cable above 100 %": float((a[:, 2] > 100).sum()),
            "day: highest transformer loading (%)": float(np.nanmax(a[:, 3])) if len(net.trafo) else float("nan"),
            "day: highest cable losses (MW)": a[:, 4].max()}


def evaluate(G, net, head, buses, syn=True):
    row = {}
    row.update({k: float(v) for k, v in feeder_counts(G).items()})
    if syn:
        kl = compare_feeders(G, real)
        row.update({f"KL {k}": float(v) for k, v in kl.items()})
        row["KS cable length"] = float(compare_cable_lengths(G, real))
    row.update(concentration(G))
    row.update(electrics(net))
    row.update(snapshot(copy.deepcopy(net), head, buses))
    row.update(day(copy.deepcopy(net), head, buses))
    return row


results = {"real": [], "original fake": [], "new fake": []}
hop = {n: d["h"] for n, d in real.nodes(data=True)}
r_buses = feeder_part(real_net)
r_head = min(r_buses, key=lambda b: hop.get(b, 10 ** 9))
results["real"].append(evaluate(real, real_net, r_head, r_buses, syn=False))
print("real done", flush=True)

p_orig = fit_params_from_feeders(feeders, single_reference=False)
p_new = fit_params_from_feeders(feeders, single_reference=True)
p_new.cable_library = cable_library_from_net(real_net)
p_new.current_ratio_cap = max_current_ratio(real_net, real, design, 20.0, design_gen)
structure = fit_switch_structure(real)

for s in range(N_SEEDS):
    g = SchweetzerFeederGenerator(params=p_orig, seed=s).generate_feeder(
        n_nodes=real.number_of_nodes(), total_load_mw=em["total_load_mw"], total_gen_mw=em["total_gen_mw"], v_nom_kv=20.0)
    net = feeder_to_pandapower(g)
    head = min(g.nodes, key=lambda n: g.nodes[n]["h"])
    results["original fake"].append(evaluate(g, net, head, feeder_part(net)))

    g = SchweetzerFeederGenerator(params=p_new, seed=s).generate_feeder(
        n_nodes=real.number_of_nodes(), total_load_mw=em["total_load_mw"], total_gen_mw=em["total_gen_mw"], v_nom_kv=20.0,
        design_load_mw=design, design_gen_mw=design_gen)
    assign_switch_edges(g, structure, np.random.default_rng(s + 1))
    net = feeder_to_pandapower(g)
    add_upstream_equipment(net, equipment)
    head = int(net.bus.index[net.bus.name == "MV busbar"][0])
    results["new fake"].append(evaluate(g, net, head, feeder_part(net)))
    print("seed", s, "done", flush=True)
    OUT.write_text(json.dumps(results))
OUT.write_text(json.dumps(results))
print("finished")

# ----------------------------------------------------------------------------------------------- short table
SUMMARY = ["cables", "switches", "load buses", "generation buses", "total cable length (km)", "median cable (m)", "longest cable (m)",
           "largest load, share of load", "largest generator, share of generation",
           "day: highest cable loading (%)", "day: steps with a cable above 100 %", "day: max voltage drop (%)",
           "day: lowest voltage (pu)", "day: highest transformer loading (%)"]


def _ms(rows, key):
    v = np.array([r.get(key, np.nan) for r in rows], dtype=float)
    if np.all(np.isnan(v)):
        return "-"
    return f"{np.nanmean(v):.2f}" + (f" +- {np.nanstd(v, ddof=1):.2f}" if len(v) > 1 else "")


print(f"\n{'':44s}{'real':>12s}{'original fake':>20s}{'new fake':>20s}   ({N_SEEDS} seeds)")
for key in SUMMARY:
    print(f"{key:44s}{_ms(results['real'], key):>12s}{_ms(results['original fake'], key):>20s}{_ms(results['new fake'], key):>20s}")
