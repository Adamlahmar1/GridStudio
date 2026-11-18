# Author: Adam Lahmar <adam.lahmar@rwth-aachen.de>
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


def design_load_mw(net, timeseries_dir: Path, quantile: float = 0.99) -> float:
    """Total load (MW) that the feeder has to be able to carry: the ``quantile`` of the summed load over the whole time
    series, with the generation left out (the busiest moment is typically one without sun).

    The load is summed the same way as in ``apply_real_power_snapshot`` (import side only). A quantile is used instead
    of the maximum because of artefacts in the data (for example the clock change in autumn can duplicate an hour, which gives
    about an hour of doubled readings). A device without a reading counts as zero: if one small load has no values for some
    months, without this every step of these months would be unknown (NaN) and ignored."""
    timeseries_dir = Path(timeseries_dir)
    load_wide = pd.read_csv(timeseries_dir / "load_timeseries_5min.csv")
    trafo_wide = pd.read_csv(timeseries_dir / "transformer_timeseries_5min.csv")
    total = pd.Series(0.0, index=load_wide.index)
    for _, row in net.load.iterrows():
        cod = row["agui_cod"]
        if row["agui_type"] == "pod":
            total += (load_wide[f"{cod}_A1"] / 1000.0).fillna(0.0)
        else:
            total += trafo_wide[f"{cod}_A"].clip(lower=0.0).fillna(0.0)
    return float(total.quantile(quantile))


def design_generation_mw(net, timeseries_dir: Path, quantile: float = 0.99) -> float:
    """Total generation (MW) that the feeder has to be able to export: the ``quantile`` of the summed generation over the
    whole time series (the sunniest moments), counted the same way as in ``apply_real_power_snapshot``.

    Together with ``design_load_mw`` this gives the two cases a feeder is built for: peak load without generation, and
    peak generation. Cables sized for the first case only are overloaded on sunny days."""
    timeseries_dir = Path(timeseries_dir)
    load_wide = pd.read_csv(timeseries_dir / "load_timeseries_5min.csv")
    trafo_wide = pd.read_csv(timeseries_dir / "transformer_timeseries_5min.csv")
    total = pd.Series(0.0, index=load_wide.index)
    for _, row in net.sgen.iterrows():
        cod = row["agui_cod"]
        if row["agui_type"] == "transformer":
            total += (-trafo_wide[f"{cod}_A"]).clip(lower=0.0).fillna(0.0)
        else:
            total += (load_wide[f"{cod}_A2"] / 1000.0).fillna(0.0)
    return float(total.quantile(quantile))


def peak_operating_points(net, timeseries_dir: Path, outlier_factor: float = 1.5) -> dict:
    """The moments of the highest load and of the highest generation in the time series.

    Returns ``{"peak_load": {...}, "peak_generation": {...}}``; each entry has ``time`` (as in the files), ``load_mw`` and
    ``gen_mw``: the summed measured load and generation of THAT time step (a device without a reading counts as zero).
    Load readings above ``outlier_factor`` times the 99th percentile are treated as errors and cannot be the peak (the clock
    change in autumn can duplicate an hour, which gives about an hour of doubled readings, well above any real peak). The moment is found on the median of three consecutive
    readings, so that a single wrong reading is not taken for the peak either; the values reported are the readings of the
    chosen step. The rule is applied to the load only: generation peaks are short (a few hours a day at most), so the 99th
    percentile can be far below a genuine peak."""
    timeseries_dir = Path(timeseries_dir)
    load_wide = pd.read_csv(timeseries_dir / "load_timeseries_5min.csv")
    trafo_wide = pd.read_csv(timeseries_dir / "transformer_timeseries_5min.csv")
    load = pd.Series(0.0, index=load_wide.index)
    gen = pd.Series(0.0, index=load_wide.index)
    for _, row in net.load.iterrows():
        cod = row["agui_cod"]
        load += ((load_wide[f"{cod}_A1"] / 1000.0) if row["agui_type"] == "pod" else trafo_wide[f"{cod}_A"].clip(lower=0.0)).fillna(0.0)
    for _, row in net.sgen.iterrows():
        cod = row["agui_cod"]
        gen += ((-trafo_wide[f"{cod}_A"]).clip(lower=0.0) if row["agui_type"] == "transformer" else load_wide[f"{cod}_A2"] / 1000.0).fillna(0.0)
    times = load_wide["time"] if "time" in load_wide.columns else pd.Series(load_wide.index.astype(str))

    def at(series, drop_outliers):
        valid = series.where(series <= outlier_factor * series.quantile(0.99)) if drop_outliers else series      # NaN = treated as an error
        i = int(valid.rolling(3, center=True, min_periods=1).median().idxmax())
        return {"time": str(times.iloc[i]), "load_mw": float(load.iloc[i]), "gen_mw": float(gen.iloc[i])}

    return {"peak_load": at(load, True), "peak_generation": at(gen, False)}


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
