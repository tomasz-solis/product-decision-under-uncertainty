# Decision Quality Framework Under Uncertainty
## Executive Summary

**For:** Leadership, Executive Stakeholders, Portfolio Review
**Author:** Tomasz Solis
**Purpose:** One-page overview for non-technical decision-makers

---

## What This Is

A **decision analysis framework** that compares strategic options across thousands of plausible futures to identify strong choices under deep uncertainty—without falsely claiming to predict outcomes.

**Not forecasting.** Not optimization. Not recommendations.

**Instead:** Transparent trade-off analysis that lets decision-makers choose based on their risk preferences.

---

## The Problem It Solves

Product teams face high-stakes decisions with uncertain outcomes:
- Platform stability vs. new features?
- Technical debt vs. market opportunity?
- Build vs. buy vs. partner?

Traditional approaches fail:
- **Gut feel:** Not transparent or defensible
- **ROI calculators:** Ignore uncertainty and tail risks
- **Forecasts:** False precision; overconfident

The framework handles these decisions while acknowledging what we don't know.

---

## How It Works

```
1. Define Options          2. Elicit Ranges         3. Simulate Futures       4. Compare Trade-offs
   (3-5 choices)              (triangular dists)       (10K-50K scenarios)        (EV, risk, regret)
        ↓                           ↓                          ↓                         ↓
   Do Nothing                 Adoption: 10-50%          Run all options           Option A: Higher EV
   Stabilize Core             Cost: €500K-€1M           under SAME              Option B: Lower regret
   Add Feature                Failure rate: 2-10%       uncertainty             Option C: Strong across
   New Capability                                                                scenarios
```

**Key Principle:** Evaluate all options under identical uncertainty conditions—this isolates decision quality from luck.

---

## What You Get

### Comparative Metrics

| Metric | What It Tells You | Decision-Maker Type |
|--------|-------------------|-------------------|
| **Expected Value (EV)** | Average outcome across all futures | Risk-neutral |
| **P05 (5th percentile)** | Worst-case scenario (downside protection) | Risk-averse |
| **Regret** | How much you'd lose vs. best alternative | Minimizing missed opportunity |
| **Win Rate** | % of scenarios where this option wins | Seeking consistency |
| **Sensitivity** | Which uncertainties matter most | Need to prioritize data |

### Example Output

| Option | EV | P05 | P95 | Mean Regret | Win Rate |
|--------|-----|-----|-----|-------------|----------|
| Do Nothing | €-€185K | €-€520K | €-€45K | €605K | 0% |
| Stabilize Core | **€+€420K** | €-€80K | €+€850K | €115K | 38% |
| Feature Extension | €+€365K | **€+€50K** | €+€720K | **€89K** | **47%** |
| New Capability | €+€280K | €-€350K | €+€1.2M | €175K | 15% |

**Interpretation:**
- **Risk-neutral:** Choose Stabilize Core (highest EV)
- **Risk-averse:** Choose Feature Extension (positive P05; no downside)
- **Regret-averse:** Choose Feature Extension (lowest missed opportunity)

---

## Example Application

The CASE_STUDY.md file walks through a platform reliability investment decision with a €2M budget. It's illustrative but shows the complete methodology: parameter elicitation, simulation execution, sensitivity analysis, scenario comparison, and retrospective format.

Key insights from the example:
- Feature Extension had lower expected value than Stabilize Core but 85% lower regret vs status quo
- Adoption rate was the dominant uncertainty (correlation +0.72)
- The recommended choice won in 47% of simulated scenarios vs 38% for the highest-EV option
- Risk preferences matter: different stakeholders would justify different choices

It shows how to structure decisions, not what the "right answer" is.

---

## Use Cases

**Primary:** Product platform investment prioritization
- Platform stability vs. feature velocity
- Technical debt vs. market opportunity
- Build vs. buy vs. partner

**Secondary:**
- Market entry timing
- Pricing strategy under demand uncertainty
- Resource allocation across teams
- Risk mitigation trade-offs

**Not Suitable For:**
- Real-time operational decisions (too slow)
- Easily reversible choices (overkill)
- Forecasting for budgets/planning (different purpose)

---

## Success Metrics

### Adoption & Impact
- Framework used for 5+ real product decisions within 6 months
- Stakeholders proactively request sensitivity analysis
- Decision rationale cited in leadership memos

### Trust & Transparency
- Stakeholders review and challenge assumptions (engagement signal)
- Post-decision retrospectives improve parameter calibration
- Reproducible analyses validated by peers

---

## What's Included

### Documentation (5 guides)
1. **[CASE_STUDY.md](CASE_STUDY.md)** (11 sections) - Real-world application with €2M decision
2. **[METHODOLOGY.md](METHODOLOGY.md)** (8 sections) - Decision science best practices
3. **[README.md](README.md)** - Quickstart and user guide
4. **[simulator/assumptions.md](simulator/assumptions.md)** - Parameter documentation
5. **[simulator/model_spec.md](simulator/model_spec.md)** - Technical specification

### Code (Production-Ready)
- **[simulator/mvp_simulator.py](simulator/mvp_simulator.py)** - Core simulation engine
- **[simulator/config.py](simulator/config.py)** - Configuration management
- **[simulator/analyze_data.py](simulator/analyze_data.py)** - CLI data analysis tool
- **[app.py](app.py)** - Streamlit web interface
- **6 test suites** - Unit, integration, edge cases, property-based, visualizations, analytical

### Interactive Tools
- **Web UI:** Streamlit app with parameter sliders and real-time simulation
- **CSV Upload:** Data-driven parameter calibration with quality assessment
- **CLI Tool:** Quick CSV analysis and YAML generation

---

## Next Steps for Adoption

### For Immediate Use
1. Install dependencies: `pip install -r requirements.txt`
2. Run web UI: `streamlit run app.py`
3. Adjust parameters via sliders or upload CSV
4. Review trade-offs and make decision

### For Customization
1. Edit [config.yaml](simulator/config.yaml) with your parameters
2. Document assumptions in [assumptions.md](simulator/assumptions.md)
3. Run simulation: `python -m simulator.mvp_simulator`
4. Adapt decision options in [mvp_simulator.py](simulator/mvp_simulator.py)

### For Learning
1. Read [METHODOLOGY.md](METHODOLOGY.md) for step-by-step process
2. Review [CASE_STUDY.md](CASE_STUDY.md) for real-world example

---

## Risks & Limitations

### What This Framework Does NOT Do
- ❌ Forecast specific outcomes (comparison only)
- ❌ Provide recommendations (preserves decision-maker authority)
- ❌ Optimize automatically (requires human judgment)
- ❌ Handle dynamic multi-period decisions (single time horizon)
- ❌ Model correlated uncertainties (independence assumption)

### Mitigation Strategies
- **Garbage in, garbage out:** Require documented parameter rationale; peer review
- **Misuse as forecast:** Clear messaging; show distributions, not just EV
- **Confirmation bias:** Version-control config; sensitivity analysis exposes manipulation
- **Analysis paralysis:** Timebox analysis (2 weeks max); satisfice over perfect

---

## The Bottom Line

The framework turns high-stakes, uncertain product decisions into **clear, defendable analyses** that quantify trade-offs and expose assumptions—without falsely claiming predictive accuracy.

**Key Value:**
- Better decisions under uncertainty
- Transparent rationale for stakeholders
- Learning from retrospectives
- Repeatable process for organizational use
- Real-world application with measurable outcomes
- Bridges technical rigor and business judgment

---

**For detailed information:**
- Technical: See [METHODOLOGY.md](METHODOLOGY.md)
- Example: See [CASE_STUDY.md](CASE_STUDY.md)
- Implementation: See [README.md](README.md)

**Version:** 1.0 | **Last Updated:** January 30, 2026
