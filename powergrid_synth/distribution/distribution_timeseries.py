"""
Wire real, measured power magnitudes from the ENEL 5-minute time-series
exports into a static pandapower snapshot's load/sgen tables.

The static ENEL network export has every load/sgen ``p_mw`` at 0 -- the
real magnitudes live in separate CSVs, keyed by each device's
``agui_cod`` (its pod or transformer identifier), not by pandapower row
index. Two source files carry the numbers:

* ``load_timeseries_5min.csv`` -- per-pod meters, one active/reactive
  column pair per direction (``<cod>_A1``/``_R1`` = import/consumption,
  ``<cod>_A2``/``_R2`` = export/production), in kW/kvar.
* ``transformer_timeseries_5min.csv`` -- per-LV-transformer *net* flow,
  one active/reactive column pair (``<cod>_A``/``_R``), in MW/Mvar,
  signed (positive = importing from MV / net consumption, negative =
  net export / local generation exceeding local demand).

A bus behind an LV transformer can have *both* a load row and an sgen
row in pandapower (its way of expressing "there is local generation
somewhere behind this transformer"). Since the transformer file only
reports the net flow, it is split by sign: the positive part becomes
the load row's value, the flipped negative part becomes the sgen row's
value.

Each row is reduced to its annual mean -- a representative "typical
operating point" for the static snapshot, consistent with how
``_load_deviations`` in ``distribution_analysis.py`` measures a node's
share of a feeder's *total* load rather than an instantaneous reading.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def _clip_mean(series: pd.Series, keep_positive: bool) -> float:
    s = series if keep_positive else -series
    return float(s.clip(lower=0.0).mean())


def apply_real_power_snapshot(net, timeseries_dir: Path) -> None:
    """Overwrite ``net.load``/``net.sgen`` ``p_mw``/``q_mvar`` in place with
    annual-mean real values from the ENEL time-series exports."""
    timeseries_dir = Path(timeseries_dir)
    load_wide = pd.read_csv(timeseries_dir / "load_timeseries_5min.csv")
    trafo_wide = pd.read_csv(timeseries_dir / "transformer_timeseries_5min.csv")

    for idx, row in net.load.iterrows():
        cod, kind = row["agui_cod"], row["agui_type"]
        if kind == "pod":
            net.load.at[idx, "p_mw"] = load_wide[f"{cod}_A1"].mean() / 1000.0
            net.load.at[idx, "q_mvar"] = load_wide[f"{cod}_R1"].mean() / 1000.0
        else:  # transformer: net flow, keep the import (positive) side
            net.load.at[idx, "p_mw"] = _clip_mean(trafo_wide[f"{cod}_A"], keep_positive=True)
            net.load.at[idx, "q_mvar"] = _clip_mean(trafo_wide[f"{cod}_R"], keep_positive=True)

    for idx, row in net.sgen.iterrows():
        cod, kind = row["agui_cod"], row["agui_type"]
        if kind == "transformer":
            net.sgen.at[idx, "p_mw"] = _clip_mean(trafo_wide[f"{cod}_A"], keep_positive=False)
            net.sgen.at[idx, "q_mvar"] = _clip_mean(trafo_wide[f"{cod}_R"], keep_positive=False)
        else:  # pod's own generation (export) side
            net.sgen.at[idx, "p_mw"] = load_wide[f"{cod}_A2"].mean() / 1000.0
            net.sgen.at[idx, "q_mvar"] = load_wide[f"{cod}_R2"].mean() / 1000.0
