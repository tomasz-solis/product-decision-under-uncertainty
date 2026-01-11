# simulator/config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import copy
import yaml

DistName = Literal["tri", "uniform"]


@dataclass(frozen=True)
class ParamSpec:
    dist: DistName
    low: float
    mode: Optional[float] = None
    high: Optional[float] = None


def load_config(config_path: str | Path) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a mapping/dict.")
    return cfg


def get_seed(cfg: Dict[str, Any]) -> int:
    proj = cfg.get("project", {})
    if isinstance(proj, dict) and "seed" in proj:
        return int(proj["seed"])
    return 7


def get_simulation_settings(cfg: Dict[str, Any]) -> Dict[str, Any]:
    sim = cfg.get("simulation", {})
    if not isinstance(sim, dict):
        raise ValueError("'simulation' must be a mapping/dict if provided.")
    n_worlds = int(sim.get("n_worlds", 20000))
    volume = int(sim.get("volume", 100000))
    scenario = str(sim.get("scenario", "base"))
    return {"n_worlds": n_worlds, "volume": volume, "scenario": scenario}


def apply_scenario(cfg: Dict[str, Any], scenario_name: str) -> Dict[str, Any]:
    """
    Return a new config with scenario overrides applied.

    Overrides are expected under:
      cfg["scenarios"][scenario_name][param_name] = {low, mode, high}
    """
    cfg2 = copy.deepcopy(cfg)

    scenarios = cfg2.get("scenarios", {})
    if not isinstance(scenarios, dict) or not scenarios:
        # No scenarios defined: treat as no-op
        return cfg2

    if scenario_name not in scenarios:
        raise ValueError(f"Scenario '{scenario_name}' not found in config.yaml (scenarios).")

    overrides = scenarios[scenario_name]
    if overrides is None:
        return cfg2
    if not isinstance(overrides, dict):
        raise ValueError(f"Scenario '{scenario_name}' must be a mapping/dict.")

    params = cfg2.get("params")
    if not isinstance(params, dict) or not params:
        raise ValueError("Config must contain a non-empty 'params' mapping.")

    for param_name, ov in overrides.items():
        if param_name not in params:
            raise ValueError(f"Scenario override references unknown param '{param_name}'.")
        if not isinstance(ov, dict):
            raise ValueError(f"Scenario override for '{param_name}' must be a mapping/dict.")

        # Only override fields that exist in the distribution.
        dist = params[param_name].get("dist")
        if dist not in ("tri", "uniform"):
            raise ValueError(f"Param '{param_name}': unsupported dist '{dist}'.")

        # Apply low/high always. Apply mode only for tri.
        for k in ("low", "high"):
            if k in ov:
                params[param_name][k] = float(ov[k])

        if dist == "tri" and "mode" in ov:
            params[param_name]["mode"] = float(ov["mode"])

    return cfg2


def parse_param_specs(cfg: Dict[str, Any]) -> Dict[str, ParamSpec]:
    params = cfg.get("params")
    if not isinstance(params, dict) or not params:
        raise ValueError("Config must contain a non-empty 'params' mapping.")

    out: Dict[str, ParamSpec] = {}
    for name, spec in params.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Param '{name}' must be a mapping.")
        dist = spec.get("dist")
        if dist not in ("tri", "uniform"):
            raise ValueError(f"Param '{name}': unsupported dist '{dist}' (use tri|uniform).")

        low = float(spec["low"])
        if dist == "uniform":
            high = float(spec["high"])
            out[name] = ParamSpec(dist="uniform", low=low, high=high)
        else:
            mode = float(spec["mode"])
            high = float(spec["high"])
            out[name] = ParamSpec(dist="tri", low=low, mode=mode, high=high)

    return out
