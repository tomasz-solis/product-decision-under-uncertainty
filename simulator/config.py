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


def apply_scenario(cfg: dict, scenario_name: str) -> dict:
    """
    Apply a scenario by overriding entries in cfg["params"].

    Supports both YAML shapes:
    1) scenarios.<name>.overrides.<param> = {...}
    2) scenarios.<name>.<param> = {...}

    This function is pure:
    - does not mutate input cfg
    - returns a new dict
    """
    new_cfg = copy.deepcopy(cfg)

    scenarios = new_cfg.get("scenarios", {})
    if scenario_name not in scenarios:
        raise KeyError(f"Unknown scenario '{scenario_name}'. Available: {list(scenarios.keys())}")

    scenario_entry = scenarios.get(scenario_name) or {}

    # Support both: {"overrides": {...}} and direct mapping {...}
    overrides = scenario_entry.get("overrides", scenario_entry)

    if overrides is None:
        overrides = {}

    if "params" not in new_cfg or not isinstance(new_cfg["params"], dict) or not new_cfg["params"]:
        raise ValueError("Config must contain a non-empty 'params' mapping.")

    if not isinstance(overrides, dict):
        raise TypeError(f"Scenario '{scenario_name}' must be a dict (or contain an 'overrides' dict).")

    for param_name, param_override in overrides.items():
        if param_name not in new_cfg["params"]:
            raise KeyError(f"Scenario override for unknown param '{param_name}'.")

        if not isinstance(param_override, dict):
            raise TypeError(f"Override for param '{param_name}' must be a dict (e.g. low/mode/high).")

        # Merge: scenario only overrides the fields it provides
        new_cfg["params"][param_name] = {**new_cfg["params"][param_name], **param_override}

    return new_cfg


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
