# Author: Adam Lahmar <adam.lahmar@rwth-aachen.de>
import argparse
import copy
import os
from pathlib import Path

import numpy as np
import pandapower as pp

from powergrid_synth.distribution.distribution_converter import (
    pandapower_to_feeders,
    feeder_summary,
)
from powergrid_synth.distribution.distribution_analysis import fit_params_from_feeders
from powergrid_synth.distribution.distribution_synthesis import SchweetzerFeederGenerator
from powergrid_synth.distribution.distribution_powerflow import check_feasibility, feeder_to_pandapower
from powergrid_synth.distribution.distribution_equipment import (
    add_upstream_equipment,
    assign_switch_edges,
    cable_library_from_net,
    fit_switch_structure,
    max_current_ratio,
    upstream_from_net,
)
from powergrid_synth.distribution.distribution_timeseries import (
    apply_real_power_snapshot,
    design_generation_mw,
    design_load_mw,
)
from powergrid_synth.distribution.feeder_map import write_feeder_map
from powergrid_synth.distribution.distribution_validation import (
    compare_cable_lengths,
    compare_feeders,
    compute_emergent_properties,
    feeder_counts,
    validate_tree,
)

DEFAULT_ITALIAN_DSO_DATA_DIR = "Italian_DSO_grid_data"
NET_SUBPATH = Path("ppnet_DW00.1.382280-VE_202412") / "ppnet_DW00.1.382280-VE_202412.json"


def parse_args():
    # Paths and seed are configurable so the same script runs locally and inside
    # the Docker container (where the data is mounted at /data, output at /output).
    parser = argparse.ArgumentParser(
        description="Italian DSO distribution feeder: reference-based synthesis and validation")
    parser.add_argument(
        "--italian-dso-data-dir", default=os.environ.get("ITALIAN_DSO_DATA_DIR", DEFAULT_ITALIAN_DSO_DATA_DIR),
        help="folder containing ppnet_DW00.1.382280-VE_202412/ and timeseries_data/ "
             "(environment variable: ITALIAN_DSO_DATA_DIR)")
    parser.add_argument(
        "--output-dir", default=os.environ.get("OUTPUT_DIR", "output"),
        help="folder the generated feeder is saved to (environment variable: OUTPUT_DIR)")
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed of the synthetic feeder (default: 42)")
    parser.add_argument(
        "--feeder-only", action="store_true",
        help="old behaviour: every connection is a cable and the saved network has no transformer and no switch")
    parser.add_argument(
        "--default-fit", action="store_true",
        help="old behaviour: fit as for several reference feeders (shares of empty and generation nodes and the "
             "maximum cable length fall back to defaults of the original paper)")
    parser.add_argument(
        "--mean-sizing", action="store_true",
        help="old behaviour: size the cables for the net power (load minus generation) at the annual-average load "
             "instead of for the busiest moment (design load, no generation)")
    parser.add_argument(
        "--default-cables", action="store_true",
        help="old behaviour: draw the cables from the tool's default library (Dutch medium-voltage cables) instead of "
             "the cable types of the real network")
    parser.add_argument(
        "--no-map", action="store_true",
        help="do not write the interactive map (synthetic_italian_dso_map.html: the real and the synthetic feeder side by side)")
    return parser.parse_args()


def main(args):
    data_dir = Path(args.italian_dso_data_dir)
    real_net_path = data_dir / NET_SUBPATH
    timeseries_dir = data_dir / "timeseries_data"

    if not real_net_path.exists():
        raise SystemExit(
            f"Italian DSO network file not found: {real_net_path}\n"
            "The Italian DSO data is not part of this repository or the Docker image. Point the pipeline at the folder "
            "that contains ppnet_DW00.1.382280-VE_202412/ and timeseries_data/ with --italian-dso-data-dir "
            "(or mount it at /data when using Docker: -v <folder>:/data:ro).")

    print("==================================================")
    print(" Italian DSO Distribution Feeder: Reference-Based Synthesis")
    print("==================================================")

    # --- STEP 1: LOAD THE REAL Italian DSO GRID ---
    print("\n[Step 1/8] Loading real Italian DSO pandapower network...")
    net = pp.from_json(str(real_net_path))
    print(f" Loaded: {len(net.bus)} buses, {len(net.line)} lines, "
          f"{len(net.switch)} switches, {len(net.trafo)} transformer(s), "
          f"{len(net.load)} loads, {len(net.sgen)} sgens.")

    # The static snapshot ships with every load/sgen p_mw == 0 -- the real
    # magnitudes live in the separate 5-minute time-series CSVs. Wire in each
    # device's annual-mean power before converting to feeder graphs, so
    # topology AND magnitude-based stats (total_load_mw, load_deviation, ...)
    # both reflect the real network.
    apply_real_power_snapshot(net, timeseries_dir)
    print(f" Wired in real power snapshot: "
          f"{net.load.p_mw.sum():.3f} MW total load, "
          f"{net.sgen.p_mw.sum():.3f} MW total generation (annual means).")

    # --- STEP 2: CONVERT TO SCHWEITZER FEEDER GRAPH(S) ---
    print("\n[Step 2/8] Converting to feeder graph(s) (hop distance, node types)...")
    real_feeders = pandapower_to_feeders(net)
    for i, summ in enumerate(feeder_summary(real_feeders)):
        print(f" Feeder {i}: {summ}")

    # --- STEP 3: FIT SYNTHESIS PARAMETERS FROM THE REAL FEEDER(S) ---
    print("\n[Step 3/8] Fitting Schweitzer distribution parameters from real data...")
    # One reference feeder: use its own measured shares and cable cut-off instead of defaults (see --default-fit).
    fitted_params = fit_params_from_feeders(real_feeders, single_reference=not args.default_fit)
    print(f" Fitted hop_dist:   {fitted_params.hop_dist}")
    print(f" Fitted degree_dist:{fitted_params.degree_dist}")
    print(f" Fitted cable_length:{fitted_params.cable_length}")

    # --- STEP 4: GENERATE A SYNTHETIC (FAKE) FEEDER FROM THE FITTED PARAMETERS ---
    print("\n[Step 4/8] Generating synthetic feeder from fitted parameters...")
    real_feeder = real_feeders[0]
    n_nodes_target = real_feeder.number_of_nodes()

    real_emergent = compute_emergent_properties(real_feeder)
    total_load_mw = real_emergent["total_load_mw"]
    total_gen_mw = real_emergent["total_gen_mw"]
    print(f" Real feeder totals: {total_load_mw:.3f} MW load, {total_gen_mw:.3f} MW generation.")

    # Real feeders are built for their two worst cases, not for the annual average (see --mean-sizing): the busiest
    # moment (high load, no sun) and the sunniest moment (peak generation).
    design_mw = None if args.mean_sizing else design_load_mw(net, timeseries_dir)
    design_gen = None if args.mean_sizing else design_generation_mw(net, timeseries_dir)
    if design_mw is not None:
        print(f" Cables are sized for the design load of {design_mw:.2f} MW (annual mean: {total_load_mw:.2f} MW) "
              f"and the design generation of {design_gen:.2f} MW (annual mean: {total_gen_mw:.2f} MW).")
        # No real cable carries more than its rating in these two cases: do not draw smaller cables than that either.
        fitted_params.current_ratio_cap = max_current_ratio(net, real_feeder, design_mw, 20.0, design_gen)
        print(f" Cables are never drawn below {1 / fitted_params.current_ratio_cap:.2f} times the current they carry "
              f"(highest utilisation of a real cable: {fitted_params.current_ratio_cap:.2f}).")
    if not args.default_cables:
        fitted_params.cable_library = cable_library_from_net(net)
        print(f" Cable library: {len(fitted_params.cable_library)} cable types of the real network.")

    gen = SchweetzerFeederGenerator(params=fitted_params, seed=args.seed)
    synthetic_feeder = gen.generate_feeder(
        n_nodes=n_nodes_target,
        total_load_mw=total_load_mw,
        total_gen_mw=total_gen_mw,
        v_nom_kv=20.0,  # nominal voltage of the real feeder (its dominant MV level)
        design_load_mw=design_mw,
        design_gen_mw=design_gen,
    )
    print(f" Generated synthetic feeder: {synthetic_feeder.number_of_nodes()} nodes, "
          f"{synthetic_feeder.number_of_edges()} edges.")

    if not args.feeder_only:
        # The generator makes every connection a cable. The real feeder has few cables and many switches, so a share of
        # the connections is turned into closed switches (own random generator: the generation above is not affected).
        structure = fit_switch_structure(real_feeder)
        counts = assign_switch_edges(synthetic_feeder, structure, np.random.default_rng(args.seed + 1))
        print(f" Switch structure: {counts['n_cables']} cables and {counts['n_switches']} switches "
              f"(real feeder: {sum(1 for *_, d in real_feeder.edges(data=True) if d.get('edge_type') == 'line')} cables, "
              f"{sum(1 for *_, d in real_feeder.edges(data=True) if d.get('edge_type') == 'switch')} switches).")

    # --- STEP 5: VALIDATE THE FAKE FEEDER AGAINST THE REAL ONE ---
    print("\n[Step 5/8] Validating synthetic feeder against real Italian DSO feeder...")
    issues = validate_tree(synthetic_feeder)
    if issues:
        print(f" Structural issues: {issues}")
    else:
        print(" Structural check: OK (connected tree).")

    kl = compare_feeders(synthetic_feeder, real_feeder)
    print(" KL divergence (synthetic vs. real):")
    for k, v in kl.items():
        print(f"   {k}: {v:.4f}")

    print(f"   cable_length_ks (0 = identical, 1 = completely different): "
          f"{compare_cable_lengths(synthetic_feeder, real_feeder):.3f}")

    print("\n Emergent properties:")
    print(f"   Real:      {compute_emergent_properties(real_feeder)}")
    print(f"   Synthetic: {compute_emergent_properties(synthetic_feeder)}")

    print("\n Real feeder compared with the synthetic feeder:")
    real_counts, syn_counts = feeder_counts(real_feeder), feeder_counts(synthetic_feeder)
    print(f"   {'':28s}{'real':>12s}{'synthetic':>12s}")
    for key in real_counts:
        print(f"   {key:28s}{real_counts[key]:12.2f}{syn_counts[key]:12.2f}")

    # --- STEP 6: ELECTRICAL FEASIBILITY (power flow) ---
    print("\n[Step 6/8] Checking electrical feasibility (power flow)...")
    try:
        pp.runpp(net)
        real_pf = {
            "converged": True,
            "vm_pu_min": float(net.res_bus["vm_pu"].min()),
            "vm_pu_max": float(net.res_bus["vm_pu"].max()),
            "max_loading_percent": float(net.res_line["loading_percent"].max()),
            "n_overloaded_lines": int((net.res_line["loading_percent"] > 100.0).sum()),
        }
    except Exception as exc:
        real_pf = {"converged": False, "error": f"{type(exc).__name__}: {exc}"}
    synthetic_pf = check_feasibility(synthetic_feeder)
    print(f"   Real:      {real_pf}")
    print(f"   Synthetic: {synthetic_pf}")

    # --- STEP 7: SAVE THE GENERATED SYNTHETIC GRID ---
    print("\n[Step 7/8] Saving synthetic feeder...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "synthetic_italian_dso_feeder.json"
    synthetic_net = feeder_to_pandapower(synthetic_feeder)
    if not args.feeder_only:
        equipment = upstream_from_net(net)       # transformer and source voltage of the real network
        if equipment is not None:
            add_upstream_equipment(synthetic_net, equipment)
            check = copy.deepcopy(synthetic_net)  # solve a copy: a power flow would add result tables to the saved file
            pp.runpp(check)
            print(f" Complete network: {len(synthetic_net.bus)} buses, {len(synthetic_net.line)} lines, "
                  f"{len(synthetic_net.switch)} switches, {len(synthetic_net.trafo)} transformer, "
                  f"{len(synthetic_net.load)} loads, {len(synthetic_net.sgen)} sgens; source at "
                  f"{synthetic_net.bus.vn_kv[synthetic_net.ext_grid.bus.iloc[0]]:.0f} kV.")
            print(f" Power flow of the complete network: converged, voltage {check.res_bus.vm_pu.min():.4f}"
                  f"-{check.res_bus.vm_pu.max():.4f} pu, max line loading {check.res_line.loading_percent.max():.1f} %, "
                  f"transformer loading {check.res_trafo.loading_percent.max():.1f} %.")
    pp.to_json(synthetic_net, str(output_path))
    print(f" Saved to {output_path.resolve()}")

    # --- STEP 8: INTERACTIVE MAP OF THE REAL AND THE SYNTHETIC FEEDER ---
    if not args.no_map:
        print("\n[Step 8/8] Writing the interactive map (real and synthetic feeder side by side)...")
        try:
            map_path = write_feeder_map(
                net, synthetic_net, output_dir / "synthetic_italian_dso_map.html", timeseries_dir=timeseries_dir,
                real_name=f"Real feeder ({real_net_path.stem})", synthetic_name=f"Synthetic feeder (seed {args.seed})")
            print(f" Saved to {map_path.resolve()} (one self-contained page: open it in a web browser; it contains your feeder data)")
        except Exception as exc:                      # the map is an extra: never let it stop the pipeline
            print(f" The map could not be written ({type(exc).__name__}: {exc}); the synthetic feeder above is saved.")

    print("\n[Done]")


if __name__ == "__main__":
    main(parse_args())
