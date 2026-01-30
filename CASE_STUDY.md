# Case Study: E-Commerce Platform Reliability Investment Decision

Note: This case study is illustrative. It shows how to structure a decision analysis but does not represent an actual business decision. Parameter ranges and outcomes are realistic but synthetic.

**Decision Date:** Q4 2025
**Decision Owner:** VP of Product, Product Director
**Analyst:** Tomasz Solis
**Framework:** Decision Quality Under Uncertainty v1.0

---

## Executive Summary

**Decision:** How should we invest €2M in platform improvements given increasing failure rates and competitive pressure?

**Recommendation:** **Feature Extension** (optional checkout acceleration) minimizes regret (€89K vs. €605K for Status Quo) while maintaining 47% win rate across scenarios. Stabilize Core has higher EV but 30% higher regret.

**Key Trade-off:** Accept €55K lower EV in exchange for 85% regret reduction vs status quo (23% lower than Stabilize Core).

**Decision Made:** Proceed with Feature Extension; monitor adoption closely (key uncertainty).

---

## 1. Business Context

### The Challenge

Our e-commerce platform processes €50M annual GMV with a baseline failure rate of 5% (transactions fail during checkout). Each failure costs €35 in lost revenue + support costs. Competitors are improving reliability and adding features, increasing churn pressure.

Engineering proposes three investment options:
1. **Stabilize Core:** Refactor legacy checkout system to reduce failures
2. **Feature Extension:** Add optional "express checkout" to reduce customer exposure to failures
3. **New Capability:** Build AI-powered fraud prevention (new revenue stream but high regression risk)

### Stakeholder Concerns

- **CFO:** What's the ROI? Need certainty on payback period.
- **CTO:** Technical debt is mounting; we need stability.
- **CPO:** Competitors are adding features; we're losing market share.
- **CEO:** Can't afford to be wrong; this is 40% of annual product budget.

### Why This Framework?

Traditional ROI analysis assumes point estimates and ignores:
- **Uncertainty in adoption rates:** Will customers use express checkout?
- **Regression risk:** Any release could break critical flows
- **Competitive dynamics:** If competitors move faster, we lose regardless of choice
- **Tail risk:** 95% confidence = 5% chance of disaster

Decision required transparent comparison of options under plausible uncertainty.

---

## 2. Decision Options Modeled

### Option 1: Do Nothing (Status Quo)

**Description:** Maintain current system with existing failure rate and drift.

**Assumptions:**
- Baseline failure rate: 5% (range: 2-10%)
- Annual drift cost: €50K (maintenance, firefighting)
- Churn impact: 15% of failed transactions result in customer churn (range: 10-25%)

**Pros:** No upfront investment
**Cons:** Failure rate may increase; competitive disadvantage

### Option 2: Stabilize Core

**Description:** Invest €800K in refactoring legacy checkout system.

**Assumptions:**
- Failure rate reduction: 60% (range: 40-80%)
- Regression probability during rollout: 15% (range: 5-25%)
- Regression cost if it occurs: €150K (range: €50K-€300K)

**Pros:** Addresses root cause; long-term stability
**Cons:** High upfront cost; regression risk; no new features

### Option 3: Feature Extension (Express Checkout)

**Description:** Invest €600K in optional express checkout feature.

**Assumptions:**
- Adoption rate: 30% (range: 10-50%)
- Exposure reduction for adopters: 70% (bypass legacy flow)
- Revenue per adopter: €12 (premium feature, annual)
- Feature loss rate: 8% annually (customers disable it)

**Pros:** New revenue stream; reduced risk exposure; competitive differentiation
**Cons:** Requires adoption; doesn't fix underlying system

### Option 4: New Capability (AI Fraud Prevention)

**Description:** Invest €1.2M in AI-powered fraud detection.

**Assumptions:**
- Revenue uplift from fraud reduction: 8% (range: 5-15%)
- Regression risk multiplier: 2.5x (new system, higher complexity)
- Competitive advantage window: 18 months before competitors catch up

**Pros:** Major differentiation; new competitive moat
**Cons:** Highest cost and risk; diverts resources from reliability

---

## 3. Parameter Calibration

### Data Sources

| Parameter | Source | Range (P05/P50/P95) |
|-----------|--------|---------------------|
| Baseline failure rate | Internal metrics (Q1-Q3 2025) | 2% / 5% / 10% |
| Churn impact | Customer cohort analysis | 10% / 15% / 25% |
| Revenue per success | Transaction database | €28 / €35 / €45 |
| Failure cost | Support ticket analysis | €20 / €35 / €55 |
| Stabilize failure reduction | Engineering estimate + vendor benchmarks | 40% / 60% / 80% |
| Express checkout adoption | Comparable feature launches (internal + competitor data) | 10% / 30% / 50% |
| Regression probability | Historical release failure rate | 5% / 15% / 25% |

### Elicitation Process

1. **Engineering Workshop:** CTO + 3 senior engineers estimated technical parameters (failure reduction, regression risk)
2. **Product Workshop:** CPO + product managers estimated adoption and usage rates based on historical feature launches
3. **Finance Review:** CFO validated cost assumptions and revenue impact calculations
4. **Data Validation:** Analyst team anchored estimates to internal metrics and external benchmarks

**Key Insight:** Wide ranges reflect genuine uncertainty, not lack of rigor. Accepting this uncertainty is critical to decision quality.

---

## 4. Simulation Results

### Configuration

- **Simulation runs:** 20,000 worlds
- **Volume:** 100,000 annual transactions
- **Seed:** 42 (reproducible)
- **Scenarios:** Base, Conservative (pessimistic adoption + high regression), Aggressive (optimistic adoption + low regression)

### Base Scenario Results

| Option | Expected Value | P05 (Downside) | P50 (Median) | P95 (Upside) | Win Rate |
|--------|---------------|----------------|--------------|--------------|----------|
| **Do Nothing** | €-€185K | €-€520K | €-€165K | €-€45K | 0% |
| **Stabilize Core** | **€+€420K** | €-€80K | €+€380K | €+€850K | 38% |
| **Feature Extension** | €+€365K | **€+€50K** | €+€340K | €+€720K | **47%** |
| **New Capability** | €+€280K | €-€350K | €+€310K | €+€1.2M | 15% |

### Regret Analysis

**Regret = (Best option outcome in this world) - (Chosen option outcome in this world)**

| Option | Mean Regret | P95 Regret | Max Regret |
|--------|-------------|-----------|------------|
| Do Nothing | €605K | €1.05M | €1.65M |
| Stabilize Core | €115K | €340K | €680K |
| **Feature Extension** | **€89K** | **€285K** | €520K |
| New Capability | €175K | €520K | €1.1M |

**Interpretation:**
- **Feature Extension** minimizes average and worst-case regret
- In the worst 5% of scenarios, Feature Extension is only €285K worse than the best alternative
- Do Nothing has catastrophic regret in optimistic scenarios (missed €1.65M upside)

### Sensitivity Analysis

**Top 5 Parameters Driving Outcome (Spearman Correlation):**

| Parameter | Correlation with Net Value |
|-----------|---------------------------|
| Extension adoption rate | +0.72 |
| Baseline failure rate | -0.65 |
| Failure-to-churn rate | -0.58 |
| Revenue per success | +0.51 |
| Stabilize failure reduction | +0.44 |

**Key Insight:** Decision success heavily depends on adoption rate (unknown until post-launch). Recommendation: Design early adoption incentives and monitor closely.

---

## 5. Scenario Comparison

### Conservative Scenario (Pessimistic)

**Adjustments:**
- Extension adoption: 10-20% (vs. 10-50% base)
- Regression probability: 20-30% (vs. 5-25% base)
- Competitive churn multiplier: 1.5x

**Results:**

| Option | EV | Win Rate |
|--------|-----|----------|
| Do Nothing | €-€310K | 0% |
| Stabilize Core | **€+€280K** | **52%** |
| **Feature Extension** | €+€195K | 35% |
| New Capability | €-€50K | 13% |

**Interpretation:** In pessimistic scenarios, Stabilize Core wins more often BUT Feature Extension still positive EV and lower regret.

### Aggressive Scenario (Optimistic)

**Adjustments:**
- Extension adoption: 30-60% (vs. 10-50% base)
- Regression probability: 2-10% (vs. 5-25% base)
- Revenue uplift multiplier: 1.3x

**Results:**

| Option | EV | Win Rate |
|--------|-----|----------|
| Do Nothing | €-€95K | 0% |
| Stabilize Core | €+€580K | 28% |
| **Feature Extension** | **€+€650K** | **58%** |
| New Capability | €+€520K | 14% |

**Interpretation:** Feature Extension dominates in optimistic scenarios due to higher adoption upside.

---

## 6. Decision Rationale

### Final Recommendation: Feature Extension

**Why?**
1. **Strong across scenarios:** Positive EV in all three scenarios (base, conservative, aggressive)
2. **Lowest regret:** Even if wrong, maximum regret is €285K vs. €340K for Stabilize Core
3. **Fastest learning:** Adoption metrics observable within 30 days; can pivot if low
4. **Competitive differentiation:** Adds customer value, not just fixes internal tech debt
5. **Lower downside:** P05 = +€50K (still profitable in worst 5% of cases)

**Trade-offs Accepted:**
- €55K lower EV than Stabilize Core in base scenario
- Requires marketing investment to drive adoption (not modeled)
- Doesn't address root cause failure rate

### Alternative Justified Decisions

**If stakeholders prioritize:**
- **Maximum EV:** Choose **Stabilize Core** (+€420K EV)
- **Minimum worst-case:** Choose **Feature Extension** (P05 = +€50K)
- **Major upside potential:** Choose **New Capability** (P95 = +€1.2M), accepting high downside risk

### What This Framework Revealed

1. **Do Nothing is never optimal:** Even in conservative scenarios, EV is negative
2. **New Capability too risky:** High regret despite upside; better as Phase 2 after stabilization
3. **Key uncertainty = adoption rate:** Spearman correlation +0.72 means outcome hinges on customer behavior
4. **Stabilize Core vs. Feature Extension depends on risk tolerance:**
   - Risk-neutral stakeholder: Stabilize Core
   - Risk-averse stakeholder: Feature Extension

---

## 7. Decision & Execution

### Decision Made

**Chosen Option:** Feature Extension (Express Checkout)

**Decision Authority:** VP of Product, Director of Product (with CEO approval)

**Rationale Documented:**
- Board presentation included regret analysis and scenario comparison
- CFO accepted €55K EV trade-off for 85% regret reduction vs status quo
- CTO agreed to monitor adoption metrics weekly for pivot signal

### Success Metrics

| Metric | Target (30 days) | Target (90 days) | Pivot Trigger |
|--------|------------------|------------------|---------------|
| Adoption rate | ≥15% | ≥25% | <10% at 60 days |
| Failure rate for adopters | -50% vs. baseline | -60% vs. baseline | <30% reduction |
| Regression incidents | 0 critical | 0 critical | Any critical bug |
| Churn rate | Stable or ↓ | -10% vs. baseline | ↑15% |

**Pivot Plan:** If adoption <10% at 60 days, shift investment to Stabilize Core in Q1 2026.

---

## 8. Retrospective (90 Days Post-Launch)

### Actual Results

| Metric | Projected (P50) | Actual | Variance |
|--------|-----------------|--------|----------|
| Adoption rate | 30% | 34% | +13% |
| Failure rate reduction | 70% | 65% | -7% |
| Regression incidents | 15% probability | 1 minor (fixed in 2 days) | Better than expected |
| Net value (90 days) | €91K | €105K | +15% |

### Parameter Calibration Lessons

**What We Got Right:**
- Adoption range (10-50%): Actual 34% fell within our range
- Regression probability (5-25%): 1 minor incident ≈ 10% of expected severity
- Revenue per adopter: €12 projected, €13 actual

**What We Underestimated:**
- **Competitive response:** Two competitors launched similar features within 60 days (not modeled)
- **Support cost savings:** Express checkout users filed 40% fewer support tickets (unexpected benefit)

**What We Overestimated:**
- **Feature loss rate:** Projected 8% annual churn, actual 3% (stickier than expected)

### Decision Quality vs. Outcome Quality

**Decision Quality: A+**
- Structured process with transparent assumptions
- Multiple stakeholder perspectives incorporated
- Analysis covers multiple scenarios
- Clear pivot criteria pre-defined

**Outcome Quality: A**
- Actual results exceeded P50 projection
- No catastrophic downside risk realized
- Pivot not required

**Key Insight:** Good decision ≠ guaranteed good outcome. This decision was high-quality because it thoroughly addressed uncertainty, even though we got "lucky" with adoption exceeding expectations.

### Process Improvements for Next Decision

1. **Model competitive dynamics:** Explicitly include competitor response scenarios
2. **Capture secondary benefits:** Support cost savings, NPS impact, brand perception
3. **Shorten calibration cycles:** Update parameter ranges quarterly with actuals
4. **Expand scenario set:** Add "fast follower" scenario for competitive moves

---

## 9. Lessons for Decision Scientists

### What Worked

1. **Frame as comparison, not forecast:** Stakeholders trusted results because we didn't claim to predict the future
2. **Regret analysis resonated:** Non-technical executives intuitively understand "How wrong could we be?"
3. **Interactive sliders built buy-in:** Letting CFO adjust parameters himself increased trust
4. **Scenario analysis exposed worldview differences:** Conservative vs. aggressive scenarios made implicit assumptions explicit

### What Was Challenging

1. **Resisting pressure to recommend:** Stakeholders wanted "the answer"; we insisted on presenting trade-offs
2. **Avoiding false precision:** Temptation to narrow ranges to reduce uncertainty perception
3. **Balancing rigor and speed:** Initial analysis took 2 weeks; stakeholders wanted decision in 3 days
4. **Communicating uncertainty:** "Wide ranges = high uncertainty" was uncomfortable for risk-averse stakeholders

### Skills Applied

- **Technical:** Monte Carlo simulation, sensitivity analysis, parameter elicitation
- **Statistical:** Distributional thinking, hypothesis testing, correlation analysis
- **Communication:** Executive storytelling, visual decision aids, explaining trade-offs
- **Judgment:** Balancing data with expert intuition, knowing when to stop refining

---

## 10. Reusability & Adaptability

### How This Framework Generalizes

**Directly applicable to:**
- Build vs. buy vs. partner decisions
- Technical debt prioritization
- Market entry timing
- Pricing strategy optimization
- Resource allocation across teams

**Requires adaptation for:**
- Multi-period decisions (need dynamic time steps)
- Correlated uncertainties (need copulas or conditional sampling)
- Real options value (need decision trees + flexibility value)

### Template for Future Decisions

1. **Define options:** 3-5 mutually exclusive choices + status quo
2. **Identify parameters:** 10-20 uncertain inputs per option
3. **Elicit ranges:** Triangular distributions from expert judgment + data
4. **Run simulation:** 10K-50K worlds with identical parameter draws
5. **Analyze results:** EV, percentiles, win rates, regret, sensitivity
6. **Present trade-offs:** Let decision-maker choose based on their risk preferences
7. **Define success metrics:** Observable indicators + pivot triggers
8. **Retrospective:** Compare projections to actuals; update calibration

---

## Conclusion

This case study shows how the Decision Quality Framework handles high-stakes, uncertain product decisions through structured, transparent analysis. By quantifying trade-offs across plausible futures rather than forecasting a single outcome, the framework enables confident decisions despite deep uncertainty.

**Impact:**
- €2M investment decision made with clear rationale and stakeholder alignment
- 85% regret reduction vs. status quo choice
- Actual results exceeded P50 projection by 15%
- Framework adopted for all major product investments moving forward

**For Decision Scientists:** This case shows the gap between descriptive analytics (what happened) and decision science (what to do). Decision science means modeling uncertainty, comparing options, and explaining trade-offs—core skills for the role.

---

**Document Version:** 1.0
**Last Updated:** January 30, 2026
**Related Documents:** [METHODOLOGY.md](METHODOLOGY.md), [simulator/assumptions.md](simulator/assumptions.md)
