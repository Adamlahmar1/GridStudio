# Author: Adam Lahmar <adam.lahmar@rwth-aachen.de>
"""One loader per feeder. A loader reads the data folder and returns a `Reference` (see reference.py)."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandapower as pp
import pandas as pd

from powergrid_synth.distribution.distribution_timeseries import apply_real_power_snapshot

from .reference import Reference

# ----------------------------------------------------------------------------------------------- ENEL
ENEL_NET = Path("ppnet_DW00.1.382280-VE_202412") / "ppnet_DW00.1.382280-VE_202412.json"
ENEL_DAY = "2024-07-15"          # mid-summer day: solar generation clearly visible


def _clip(series: pd.Series, keep_positive: bool) -> pd.Series:
    return (series if keep_positive else -series).clip(lower=0.0)


def _enel_day(net, ts_dir: Path, day: str):
    """Total load and generation of one real day, 5-minute steps (sum over all devices of the load and generation columns)."""
    load_w = pd.read_csv(ts_dir / "load_timeseries_5min.csv", parse_dates=["time"])
    trafo_w = pd.read_csv(ts_dir / "transformer_timeseries_5min.csv", parse_dates=["time"])
    start = pd.Timestamp(day)
    sel = lambda df: df[(df["time"] >= start) & (df["time"] < start + pd.Timedelta(days=1))].reset_index(drop=True)
    L, T = sel(load_w), sel(trafo_w)
    load = pd.Series(0.0, index=L.index)
    gen = pd.Series(0.0, index=L.index)
    for _, r in net.load.iterrows():
        load += ((L[f"{r['agui_cod']}_A1"] / 1000.0) if r["agui_type"] == "pod" else _clip(T[f"{r['agui_cod']}_A"], True)).fillna(0.0)
    for _, r in net.sgen.iterrows():
        gen += (_clip(T[f"{r['agui_cod']}_A"], False) if r["agui_type"] == "transformer" else L[f"{r['agui_cod']}_A2"] / 1000.0).fillna(0.0)
    epochs = [int(t.to_pydatetime().replace(tzinfo=dt.timezone.utc).timestamp()) for t in L["time"]]
    return list(zip(epochs, load.tolist(), gen.tolist()))


def load_enel(data_dir) -> Reference:
    d = Path(data_dir)
    net = pp.from_json(str(d / ENEL_NET))
    apply_real_power_snapshot(net, d / "timeseries_data")      # the file has p_mw = 0, real values come from the time series
    return Reference("enel", net, v_nom_kv=20.0, day=_enel_day(net, d / "timeseries_data", ENEL_DAY), has_ratings=True)
