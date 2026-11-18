# Author: Adam Lahmar <adam.lahmar@rwth-aachen.de>
"""What every feeder loader has to deliver: a `Reference`.

A reference is the REAL feeder in the form the pipeline needs. To support a new feeder, write one loader function
that returns a Reference and register it in config.py; the rest of the pipeline does not change.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandapower as pp


@dataclass
class Reference:
    name: str
    net: "pp.pandapowerNet"          # real network; loads/generators already carry real power values (snapshot)
    v_nom_kv: float                  # nominal voltage of the feeder itself (for example 20 kV)
    day: Optional[list] = None       # one real day: [(epoch_seconds, total_load_mw, total_gen_mw), ...]
    has_ratings: bool = True         # does the data contain cable current ratings (max_i_ka)?
