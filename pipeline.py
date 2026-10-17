import argparse
import os
from pathlib import Path

import pandapower as pp

from powergrid_synth.distribution.distribution_converter import (
    pandapower_to_feeders,
    feeder_summary,
)
from powergrid_synth.distribution.distribution_analysis import fit_params_from_feeders
from powergrid_synth.distribution.distribution_synthesis import SchweetzerFeederGenerator
from powergrid_synth.distribution.distribution_powerflow import check_feasibility, feeder_to_pandapower
from powergrid_synth.distribution.distribution_timeseries import apply_real_power_snapshot
from powergrid_synth.distribution.distribution_validation import (
    compare_feeders,
    compute_emergent_properties,
    validate_tree,
)

DEFAULT_ENEL_DATA_DIR = "ENEL_grid_data"   # folder next to this script; override with --enel-data-dir or ENEL_DATA_DIR
NET_SUBPATH = Path("ppnet_DW00.1.382280-VE_202412") / "ppnet_DW00.1.382280-VE_202412.json"


def parse_args():
    # Paths and seed are configurable so the same script runs locally and inside
    # the Docker container (where the data is mounted at /data, output at /output).
    parser = argparse.ArgumentParser(
        description="ENEL distribution feeder: reference-based synthesis and validation")
    parser.add_argument(
        "--enel-data-dir", default=os.environ.get("ENEL_DATA_DIR", DEFAULT_ENEL_DATA_DIR),
        help="folder containing ppnet_DW00.1.382280-VE_202412/ and timeseries_data/ "
             "(environment variable: ENEL_DATA_DIR)")
    parser.add_argument(
        "--output-dir", default=os.environ.get("OUTPUT_DIR", "output"),
        help="folder the generated feeder is saved to (environment variable: OUTPUT_DIR)")
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed of the synthetic feeder (default: 42)")
    return parser.parse_args()


def main(args):
    data_dir = Path(args.enel_data_dir)
    real_net_path = data_dir / NET_SUBPATH
    timeseries_dir = data_dir / "timeseries_data"

    if not real_net_path.exists():
        raise SystemExit(
            f"ENEL network file not found: {real_net_path}\n"
            "The ENEL data is not part of this repository or the Docker image. Point the pipeline at the folder "
            "that contains ppnet_DW00.1.382280-VE_202412/ and timeseries_data/ with --enel-data-dir "
            "(or mount it at /data when using Docker: -v <folder>:/data:ro).")

    print("==================================================")
    print(" ENEL Distribution Feeder: Reference-Based Synthesis")
    print("==================================================")

    # --- STEP 1: LOAD THE REAL ENEL GRID ---
    print("\n[Step 1/7] Loading real ENEL pandapower network...")
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
    print("\n[Step 2/7] Converting to feeder graph(s) (hop distance, node types)...")
    real_feeders = pandapower_to_feeders(net)
    for i, summ in enumerate(feeder_summary(real_feeders)):
        print(f" Feeder {i}: {summ}")

    # --- STEP 3: FIT SYNTHESIS PARAMETERS FROM THE REAL FEEDER(S) ---
    print("\n[Step 3/7] Fitting Schweitzer distribution parameters from real data...")
    fitted_params = fit_params_from_feeders(real_feeders)
    print(f" Fitted hop_dist:   {fitted_params.hop_dist}")
    print(f" Fitted degree_dist:{fitted_params.degree_dist}")
    print(f" Fitted cable_length:{fitted_params.cable_length}")

    # --- STEP 4: GENERATE A SYNTHETIC (FAKE) FEEDER FROM THE FITTED PARAMETERS ---
    print("\n[Step 4/7] Generating synthetic feeder from fitted parameters...")
    real_feeder = real_feeders[0]
    n_nodes_target = real_feeder.number_of_nodes()

    real_emergent = compute_emergent_properties(real_feeder)
    total_load_mw = real_emergent["total_load_mw"]
    total_gen_mw = real_emergent["total_gen_mw"]
    print(f" Real feeder totals: {total_load_mw:.3f} MW load, {total_gen_mw:.3f} MW generation.")

    gen = SchweetzerFeederGenerator(params=fitted_params, seed=args.seed)
    synthetic_feeder = gen.generate_feeder(
        n_nodes=n_nodes_target,
        total_load_mw=total_load_mw,
        total_gen_mw=total_gen_mw,
        v_nom_kv=20.0,  # matches the real feeder's dominant MV level (252/254 buses)
    )
    print(f" Generated synthetic feeder: {synthetic_feeder.number_of_nodes()} nodes, "
          f"{synthetic_feeder.number_of_edges()} edges.")

    # --- STEP 5: VALIDATE THE FAKE FEEDER AGAINST THE REAL ONE ---
    print("\n[Step 5/7] Validating synthetic feeder against real ENEL feeder...")
    issues = validate_tree(synthetic_feeder)
    if issues:
        print(f" Structural issues: {issues}")
    else:
        print(" Structural check: OK (connected tree).")

    kl = compare_feeders(synthetic_feeder, real_feeder)
    print(" KL divergence (synthetic vs. real):")
    for k, v in kl.items():
        print(f"   {k}: {v:.4f}")

    print("\n Emergent properties:")
    print(f"   Real:      {compute_emergent_properties(real_feeder)}")
    print(f"   Synthetic: {compute_emergent_properties(synthetic_feeder)}")

    # --- STEP 6: ELECTRICAL FEASIBILITY (power flow) ---
    print("\n[Step 6/7] Checking electrical feasibility (power flow)...")
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
    print("\n[Step 7/7] Saving synthetic feeder...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "synthetic_enel_feeder.json"
    pp.to_json(feeder_to_pandapower(synthetic_feeder), str(output_path))
    print(f" Saved to {output_path.resolve()}")

    print("\n[Done]")


if __name__ == "__main__":
    main(parse_args())
