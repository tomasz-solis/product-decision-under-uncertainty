# Case Study: Platform Reliability Investment

This is an illustrative case study. It shows how to structure a decision analysis using this framework. Parameter ranges and outcomes are synthetic but realistic.

**Decision:** How should we invest ~2M EUR in platform improvements given increasing failure rates and competitive pressure?

---

## Options

### Do Nothing (Status Quo)

Maintain current system. Baseline failure rate ~5% (range: 2-10%). Annual drift cost ~50K EUR. No upfront investment, but failure rate may increase and competitive disadvantage grows.

### Stabilize Core

Invest ~800K EUR in checkout refactoring. Expected failure rate reduction: 60% (range: 40-80%). Regression probability during rollout: 15% (range: 5-25%). Addresses root cause but no new features.

### Feature Extension (Express Checkout)

Invest ~600K EUR in optional express checkout. Expected adoption: 30% (range: 10-50%). Bypasses legacy flow for adopters, reducing their failure exposure. New revenue stream but doesn't fix underlying system.

### New Capability (AI Fraud Prevention)

Invest ~1.2M EUR. Revenue uplift ~8% (range: 5-15%). Regression risk multiplier: 2.5x. Highest cost and risk, but potential competitive moat.

---

## Parameter Sources

| Parameter | Source | Range |
|-----------|--------|-------|
| Baseline failure rate | Internal metrics | 2% / 5% / 10% |
| Churn impact | Cohort analysis | 10% / 15% / 25% |
| Revenue per success | Transaction data | 28 / 35 / 45 EUR |
| Failure cost | Support tickets | 20 / 35 / 55 EUR |
| Stabilize failure reduction | Engineering estimate | 40% / 60% / 80% |
| Express checkout adoption | Historical feature launches | 10% / 30% / 50% |
| Regression probability | Historical release data | 5% / 15% / 25% |

Ranges came from an engineering workshop (technical parameters), product workshop (adoption/usage), and finance review (cost assumptions).

---

## Results (Base Scenario, 20K worlds)

| Option | EV | P05 | P50 | P95 | Win Rate |
|--------|-----|-----|-----|-----|----------|
| Do Nothing | -185K | -520K | -165K | -45K | 0% |
| Stabilize Core | +420K | -80K | +380K | +850K | 38% |
| Feature Extension | +365K | +50K | +340K | +720K | 47% |
| New Capability | +280K | -350K | +310K | +1.2M | 15% |

### Regret

| Option | Mean Regret | P95 Regret |
|--------|-------------|-----------|
| Do Nothing | 605K | 1.05M |
| Stabilize Core | 115K | 340K |
| Feature Extension | 89K | 285K |
| New Capability | 175K | 520K |

Feature Extension minimizes both average and tail regret. In the worst 5% of scenarios, it's only 285K worse than the best alternative.

### Top Sensitivity Drivers

| Parameter | Spearman Correlation |
|-----------|---------------------|
| Extension adoption rate | +0.72 |
| Baseline failure rate | -0.65 |
| Failure-to-churn rate | -0.58 |

Outcome depends heavily on adoption rate, which is unobservable until post-launch.

---

## Scenario Robustness

Feature Extension has positive EV in all three scenarios (base, conservative, aggressive). Stabilize Core wins more often under conservative assumptions but Feature Extension dominates in the aggressive scenario.

The choice depends on risk appetite:
- Risk-neutral: Stabilize Core (+420K EV)
- Risk-averse: Feature Extension (P05 = +50K, lowest regret)

---

## What the Framework Showed

1. Do Nothing is never optimal. Negative EV in all scenarios.
2. New Capability carries too much regret for a one-shot decision. Better as Phase 2.
3. The decision between Stabilize Core and Feature Extension is a genuine trade-off, not an obvious call. It depends on how much the decision-maker values downside protection vs. expected value.
4. Adoption rate is the key uncertainty. Worth investing in early measurement.

---

**Related:** [simulator/assumptions.md](simulator/assumptions.md), [simulator/model_spec.md](simulator/model_spec.md)
