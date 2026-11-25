# Author: Adam Lahmar <adam.lahmar@rwth-aachen.de>
"""Generic pipeline for any registered feeder.

    python -m feeder_pipeline.run italian_dso --data-dir <Italian DSO folder>
    python -m feeder_pipeline.run italian_dso --data-dir <Italian DSO folder> --seeds 10
    python -m feeder_pipeline.run italian_dso --data-dir <Italian DSO folder> --day

Steps (same as pipeline.py): load reference -> convert to feeder graph -> fit parameters -> generate synthetic feeder ->
statistical validation -> electrical validation -> save; optionally a comparison over one real day (--day).
"""
from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path

import numpy as np
import pandapower as pp

from powergrid_synth.distribution.distribution_analysis import fit_params_from_feeders
from powergrid_synth.distribution.distribution_converter import pandapower_to_feeders
from powergrid_synth.distribution.distribution_powerflow import feeder_to_pandapower
from powergrid_synth.distribution.distribution_validation import (
    compare_feeders, compute_emergent_properties, validate_tree)

from .config import FEEDERS
from .synthesis import generator_class, tuned_params

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------------------------------- helpers
def _feeder_buses(net, v_nom_kv, hop):
    """Buses at the voltage of the feeder itself, and the 'head' = the one closest to the source."""
    cand = [int(b) for b in net.bus.index if abs(net.bus.vn_kv[b] - v_nom_kv) / v_nom_kv <= 0.05]
    head = min(cand, key=lambda b: hop.get(b, 10 ** 9))
    return head, cand


def _drop_and_loading(net, head, buses, with_loading):
    vm = net.res_bus.vm_pu
    drop = 100.0 * float(vm[head] - vm[buses].min())
    loading = float(net.res_line.loading_percent.max()) if with_loading else float("nan")
    return drop, loading


def _solve(net):
    try:
        pp.runpp(net, numba=False)
        return True
    except Exception:
        return False


def _scale(net, base_l, base_g, k_l, k_g):
    net.load[["p_mw", "q_mvar"]] = base_l * k_l
    if base_g is not None:
        net.sgen[["p_mw", "q_mvar"]] = base_g * k_g


class _Scaler:
    """Scales a network to a measured load/generation total and solves the power flow."""

    def __init__(self, net, head, buses, with_loading):
        self.net, self.head, self.buses, self.with_loading = net, head, buses, with_loading
        self.base_l = net.load[["p_mw", "q_mvar"]].copy()
        self.base_g = net.sgen[["p_mw", "q_mvar"]].copy() if len(net.sgen) else None
        self.p_l0 = float(self.base_l.p_mw.sum())
        self.p_g0 = float(self.base_g.p_mw.sum()) if self.base_g is not None else 0.0

    def step(self, load_mw, gen_mw):
        _scale(self.net, self.base_l, self.base_g, load_mw / self.p_l0 if self.p_l0 > 0 else 0.0,
               gen_mw / self.p_g0 if self.p_g0 > 0 else 0.0)
        if not _solve(self.net):
            return None
        return _drop_and_loading(self.net, self.head, self.buses, self.with_loading)

    def restore(self):
        """Put the original power values back, so the network can be used again (e.g. by the next seed)."""
        _scale(self.net, self.base_l, self.base_g, 1.0, 1.0)


# ----------------------------------------------------------------------------------------------- the pipeline
def run_feeder(cfg, data_dir=None, seed=42, out_dir=None, with_day=False, day_step=None, verbose=True, ref=None):
    say = print if verbose else (lambda *a, **k: None)
    ref = ref or cfg.loader(cfg.data_dir(data_dir))
    say(f"[1/7] {cfg.name}: {len(ref.net.bus)} buses, {len(ref.net.line)} lines, {len(ref.net.load)} loads, "
        f"{len(ref.net.sgen)} generators, nominal {ref.v_nom_kv} kV")

    feeders = pandapower_to_feeders(ref.net)                                      # step 2
    real = feeders[0]
    hop = {n: d["h"] for n, d in real.nodes(data=True)}
    r_head, r_buses = _feeder_buses(ref.net, ref.v_nom_kv, hop)
    say(f"[2/7] feeder graph: {real.number_of_nodes()} nodes, {real.number_of_edges()} edges")

    params = tuned_params(cfg, ref, fit_params_from_feeders(feeders, single_reference=cfg.single_reference), real)   # step 3 (+ options of the profile)
    say(f"[3/7] fitted parameters (library={cfg.cable_library}, cable choice={cfg.cable_choice}, length fit={cfg.length_fit})")

    em = compute_emergent_properties(real)                                         # step 4
    syn = generator_class(cfg)(params=params, seed=seed).generate_feeder(
        n_nodes=real.number_of_nodes(), total_load_mw=em["total_load_mw"], total_gen_mw=em["total_gen_mw"],
        v_nom_kv=ref.v_nom_kv)
    say(f"[4/7] synthetic feeder: {syn.number_of_nodes()} nodes, {syn.number_of_edges()} edges (seed {seed})")

    issues = validate_tree(syn)                                                    # step 5
    kl = compare_feeders(syn, real)
    say(f"[5/7] tree check: {'OK' if not issues else issues}; KL divergence {{ {', '.join(f'{k}: {v:.3f}' for k, v in kl.items())} }}")

    syn_net = feeder_to_pandapower(syn)                                            # step 6
    saved_path = None
    if out_dir is not None:       # save BEFORE solving: a power flow would add result tables to the file
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        saved_path = Path(out_dir) / f"synthetic_{cfg.name}_feeder.json"
        pp.to_json(syn_net, str(saved_path))
    s_head = min(syn.nodes, key=lambda n: syn.nodes[n]["h"])
    s_buses = [int(b) for b in syn_net.bus.index]
    ok_r, ok_s = _solve(ref.net), _solve(syn_net)
    d_r, l_r = _drop_and_loading(ref.net, r_head, r_buses, ref.has_ratings) if ok_r else (float("nan"),) * 2
    d_s, l_s = _drop_and_loading(syn_net, s_head, s_buses, True) if ok_s else (float("nan"),) * 2
    say(f"[6/7] snapshot: voltage drop real {d_r:.2f} % / synthetic {d_s:.2f} %; max line loading real "
        f"{'n/a' if not ref.has_ratings else f'{l_r:.1f} %'} / synthetic {l_s:.1f} %")

    e = list(syn.edges(data=True))
    km = np.array([d["length_km"] for _, _, d in e]); rr = np.array([d["r_per_km"] for _, _, d in e])
    r_len = ref.net.line["length_km"].to_numpy(); r_res = ref.net.line["r_ohm_per_km"].to_numpy()
    n_power = np.array([syn.nodes[n].get("P_mw", 0.0) for n in syn.nodes])
    row = dict(feeder=cfg.name, seed=seed, nodes=syn.number_of_nodes(), edges=syn.number_of_edges(),
               load_buses=int((n_power > 0).sum()), real_load_buses=int(sum(1 for n in real if real.nodes[n].get("P_mw", 0.0) > 0)),
               kl_hop=kl["hop_distance_kl"], kl_deg=kl["degree_kl"], kl_len=kl["cable_length_kl"],
               real_km=float(r_len.sum()), syn_km=float(km.sum()),
               real_R=float(np.average(r_res, weights=r_len)), syn_R=float(np.average(rr, weights=km)),
               real_drop=d_r, syn_drop=d_s, real_loading=l_r, syn_loading=l_s)

    if saved_path is not None:                                                     # step 7
        say(f"[7/7] saved {saved_path}")

    if with_day and ref.day:                                                       # optional: one real day
        step = day_step or cfg.day_step
        prof = ref.day[::step]
        cache = getattr(ref, "_real_day_cache", {})            # the real feeder does not change between seeds: compute once
        if step not in cache:
            sc_r = _Scaler(ref.net, r_head, r_buses, ref.has_ratings)
            cache[step] = [x for x in (sc_r.step(l, g) for _, l, g in prof) if x]
            sc_r.restore()
            ref._real_day_cache = cache
        res_r = cache[step]
        sc_s = _Scaler(syn_net, s_head, s_buses, True)
        res_s = [x for x in (sc_s.step(l, g) for _, l, g in prof) if x]
        row.update(day_steps=len(prof),
                   day_real_drop=max(d for d, _ in res_r), day_syn_drop=max(d for d, _ in res_s),
                   day_real_loading=max(l for _, l in res_r) if ref.has_ratings else float("nan"),
                   day_syn_loading=max(l for _, l in res_s),
                   day_real_over=sum(1 for _, l in res_r if l > 100) if ref.has_ratings else -1,
                   day_syn_over=sum(1 for _, l in res_s if l > 100))
        say(f"      day ({len(prof)} steps): max voltage drop real {row['day_real_drop']:.2f} % / synthetic {row['day_syn_drop']:.2f} %; "
            f"max loading real {row['day_real_loading']:.1f} % / synthetic {row['day_syn_loading']:.1f} %; "
            f"steps above 100 %: real {row['day_real_over']} / synthetic {row['day_syn_over']}")
    return row


def main():
    ap = argparse.ArgumentParser(description="Generic synthetic-feeder pipeline")
    ap.add_argument("feeder", choices=sorted(FEEDERS), help="registered feeder profile")
    ap.add_argument("--data-dir", default=None, help="data folder of the feeder (or environment variable of the profile)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, default=0, help="study mode: run seeds 0..N-1 and write a CSV summary")
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--day", action="store_true", help="also compare real and synthetic feeder over one real day")
    ap.add_argument("--day-step", type=int, default=None, help="use every n-th step of the day")
    a = ap.parse_args()

    cfg = FEEDERS[a.feeder]
    ref = cfg.loader(cfg.data_dir(a.data_dir))
    if not a.seeds:
        run_feeder(cfg, seed=a.seed, out_dir=a.output_dir, with_day=a.day, day_step=a.day_step, ref=ref)
        return
    rows = [run_feeder(cfg, seed=s, with_day=a.day, day_step=a.day_step, verbose=False, ref=ref) for s in range(a.seeds)]
    path = Path(a.output_dir) / f"study_{cfg.name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"{cfg.name}: {len(rows)} seeds, summary saved to {path}")
    for k in ("kl_hop", "kl_deg", "kl_len", "syn_km", "syn_R", "syn_drop", "day_syn_drop"):
        v = [r[k] for r in rows if k in r]
        if v:
            print(f"  {k:12s} {np.mean(v):9.4f} +- {np.std(v, ddof=1):.4f}")
    print(f"  real: km {rows[0]['real_km']:.3f}, R {rows[0]['real_R']:.3f}, snapshot drop {rows[0]['real_drop']:.2f} %"
          + (f", day drop {rows[0]['day_real_drop']:.2f} %" if "day_real_drop" in rows[0] else ""))


if __name__ == "__main__":
    main()
