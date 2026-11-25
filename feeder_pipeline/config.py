# Author: Adam Lahmar <adam.lahmar@rwth-aachen.de>
"""Feeder profiles: everything that is special about a feeder in one place.

To add a new feeder:
  1. write a loader in loaders.py that returns a Reference,
  2. add one FeederConfig below.
Nothing else has to change.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .loaders import load_italian_dso


@dataclass(frozen=True)
class FeederConfig:
    name: str
    loader: Callable                  # loader(data_dir) -> Reference
    data_env: str                     # environment variable that names the data folder
    default_dir: str                  # used if neither --data-dir nor the environment variable is given
    cable_library: str = "tool"       # "tool" = default (medium-voltage) library | "reference" = cable types of the reference feeder
    cable_choice: str = "closest"     # "closest" = tool rule (closest current rating) | "frequency" = draw by share of length
    length_fit: str = "tool"          # "tool" = fit of the tool | "truncated" = truncated-Cauchy refit
    day_step: int = 1                 # use every n-th step of the real day (speeds up long days)
    single_reference: bool = False    # True = use the measured shares/cut-off of the reference (see fit_params_from_feeders)

    def data_dir(self, override=None) -> Path:
        return Path(override or os.environ.get(self.data_env, self.default_dir))


FEEDERS = {
    # Italian DSO: all options at the tool's defaults -> identical to pipeline.py
    "italian_dso": FeederConfig("italian_dso", load_italian_dso, "ITALIAN_DSO_DATA_DIR", "data/italian_dso"),
}
