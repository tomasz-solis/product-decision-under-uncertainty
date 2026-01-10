# Decision Simulator

This folder contains the simulation code used by the Decision Quality Framework.

The simulator exists to compare decision options under uncertainty using the
same set of assumptions and random draws.

It is not a forecasting tool and it is not intended to be used in isolation
from the surrounding decision framing.

---

## What this code does

The simulator:

- samples uncertainty from explicit distributions defined in `config.yaml`
- evaluates multiple decision options under identical assumptions
- produces comparative outputs:
  - expected value
  - downside risk
  - regret
  - win rate

All outputs are meant to be read comparatively, not as predictions.

---

## What this code does not do

The simulator does not:

- predict real-world outcomes
- estimate company-specific metrics
- recommend a decision
- optimize parameters
- validate assumptions

It exists to support reasoning about trade-offs, not to automate decisions.

---

## Files overview

- `mvp_simulator.py`  
  Core simulation logic and evaluation outputs.

- `config.yaml`  
  Single source of truth for assumptions, distributions, and scenarios.

- `config.py`  
  Configuration loading and scenario override logic.

- `model_spec.md`  
  High-level description of what the simulator represents and what it does not.

- `assumptions.md`  
  Plain-language documentation of all parameters and their ranges.

- `data_sources.md`  
  Public sources used to anchor assumption ranges.

---

## How to run

From the repository root:

```bash
python -m simulator.mvp_simulator
```

Outputs are printed to stdout for inspection.
The simulator is designed to be read and modified, not deployed.