# Decision Science Methodology Guide

**Purpose:** Complete guide to decision analysis under uncertainty using this framework
**Audience:** Decision scientists, senior data analysts, product leaders
**Version:** 1.0
**Last Updated:** January 30, 2026

---

## Table of Contents

1. [Foundational Concepts](#1-foundational-concepts)
2. [When to Use This Framework](#2-when-to-use-this-framework)
3. [Step-by-Step Process](#3-step-by-step-process)
4. [Parameter Elicitation Best Practices](#4-parameter-elicitation-best-practices)
5. [Model Validation & Calibration](#5-model-validation--calibration)
6. [Interpreting Results](#6-interpreting-results)
7. [Common Pitfalls & Solutions](#7-common-pitfalls--solutions)
8. [Advanced Techniques](#8-advanced-techniques)

---

## 1. Foundational Concepts

### 1.1 Decision Quality vs. Outcome Quality

**Decision Quality** = Likelihood of good outcomes given what was knowable at decision time

**Outcome Quality** = Actual results (influenced by luck and unknowable factors)

**Key Principle:** Good decisions can have bad outcomes; bad decisions can have good outcomes. This framework evaluates **decision quality**, not outcome prediction.

### 1.2 Why Comparison Beats Forecasting

| Forecasting Approach | Comparative Approach (This Framework) |
|---------------------|--------------------------------------|
| "Option A will generate €500K" | "Option A performs better than B in 65% of plausible futures" |
| Falsely precise | Transparently uncertain |
| Sensitive to single-point estimates | Works across distributions |
| Encourages confirmation bias | Forces multi-criteria evaluation |
| Brittle under uncertainty | Resilient under uncertainty |

### 1.3 Core Metrics Explained

#### Expected Value (EV)
- **Definition:** Mean outcome across all simulated worlds
- **Use:** Risk-neutral decision-makers; long-run average
- **Limitation:** Ignores distribution shape (tail risks)

#### Percentiles (P05, P50, P95)
- **P05 (5th percentile):** Worst-case scenario (95% of outcomes are better)
- **P50 (Median):** Typical outcome (half better, half worse)
- **P95 (95th percentile):** Best-case scenario (95% of outcomes are worse)
- **Use:** Understand downside risk (P05) and upside potential (P95)

#### Regret
- **Definition:** (Best option outcome in this world) - (Chosen option outcome in same world)
- **Mean Regret:** Average missed opportunity cost
- **P95 Regret:** Worst-case missed opportunity (what's the max we'd regret this choice?)
- **Use:** Avoid decisions with catastrophic opportunity cost

#### Win Rate
- **Definition:** % of simulated worlds where this option has highest value
- **Use:** Frequency-based confidence; consistency measure

#### Sensitivity Analysis
- **Definition:** Spearman rank correlation between input parameters and outcome
- **Use:** Identify which uncertainties matter most; prioritize data collection

---

## 2. When to Use This Framework

### ✅ Ideal Use Cases

1. **One-time, high-stakes decisions** (not ongoing forecasting)
   - Example: Platform architecture choice, market entry, M&A

2. **Multiple options with uncertain outcomes**
   - Need at least 2 alternatives to compare (3-5 optimal)

3. **Deep uncertainty** (wide ranges, unknowable factors)
   - If outcomes are nearly certain, simple math suffices

4. **Stakeholder alignment required**
   - Transparent assumptions enable debate and buy-in

5. **Trade-offs between criteria** (EV vs. risk vs. regret)
   - No single "correct" answer; preferences matter

### ❌ Not Appropriate For

1. **Real-time operational decisions** (too slow)
2. **Easily reversible choices** (overkill; just try and iterate)
3. **Pure forecasting needs** (budgeting, capacity planning)
4. **Deterministic problems** (no uncertainty to model)
5. **Political decisions** (analysis won't change minds)

---

## 3. Step-by-Step Process

### Phase 1: Problem Structuring (2-4 hours)

**Goal:** Define decision options and success criteria

**Steps:**
1. **State the decision:** Write a one-sentence decision statement
   - Good: "How should we invest €2M to improve platform reliability?"
   - Bad: "Should we fix bugs?" (too vague)

2. **Identify options:** 3-5 mutually exclusive choices + status quo
   - Ensure options span strategic space (incremental vs. transformative)
   - Include "Do Nothing" for baseline comparison

3. **Define success metrics:** What constitutes a "good outcome"?
   - Typically: Net value (revenue - costs over 12-24 months)
   - Could include: NPS, market share, time-to-market

4. **Clarify time horizon:** Over what period are we evaluating?
   - Short-term (3-6 months): Tactical decisions
   - Medium-term (1-2 years): Strategic investments
   - Long-term (3+ years): Requires dynamic modeling (advanced)

**Deliverable:** Problem statement doc with options, success metric, time horizon

### Phase 2: Parameter Identification (3-6 hours)

**Goal:** List all uncertain inputs that affect outcomes

**Steps:**
1. **Decompose each option into components:**
   - Revenue drivers: Adoption rate, uplift %, retention
   - Cost drivers: Dev cost, operational cost, support cost
   - Risk drivers: Regression probability, competitive response, churn impact

2. **Distinguish parameters from assumptions:**
   - **Parameter:** Uncertain input (modeled as distribution) - e.g., adoption rate 10-50%
   - **Assumption:** Fixed simplification (not modeled) - e.g., "assume no currency fluctuations"

3. **Aim for 10-20 parameters per decision:**
   - Too few: Underestimates uncertainty
   - Too many: Noise dominates signal; hard to elicit

4. **Group parameters by category:**
   - Base rates (apply to all options)
   - Option-specific (unique to each choice)
   - Shared uncertainties (affect multiple options)

**Deliverable:** Parameter list with definitions and categories

### Phase 3: Range Elicitation (4-8 hours)

**Goal:** Estimate plausible ranges for each parameter

**Process:** See Section 4 (Parameter Elicitation Best Practices)

**Deliverable:** Completed [config.yaml](simulator/config.yaml) with documented ranges

### Phase 4: Simulation Execution (1-2 hours)

**Goal:** Run comparative simulation across all options

**Steps:**
1. **Configure simulation settings:**
   - n_worlds: 10,000 (quick) to 50,000 (thorough)
   - seed: Fixed integer for reproducibility
   - volume: Transaction count or customer base size

2. **Run base scenario:**
   ```bash
   python -m simulator.mvp_simulator
   ```

3. **Run scenario variations:**
   - Conservative (pessimistic assumptions)
   - Aggressive (optimistic assumptions)
   - Custom scenarios based on stakeholder worldviews

4. **Validate outputs:**
   - Check for outliers or illogical results
   - Verify expected value ordering makes intuitive sense
   - Confirm reproducibility (same seed → same results)

**Deliverable:** Simulation results CSV + summary statistics

### Phase 5: Analysis & Interpretation (2-4 hours)

**Goal:** Extract insights and explain trade-offs

**Steps:**
1. **Compare options on multiple criteria:**
   - Which maximizes EV? (risk-neutral choice)
   - Which minimizes P05? (risk-averse choice)
   - Which minimizes regret? (avoids worst mistakes)

2. **Examine scenario consistency:**
   - Does same option win across scenarios?
   - Which options flip order under different worldviews?

3. **Analyze sensitivity:**
   - Which parameters drive outcomes most?
   - Where should we invest in better data?

4. **Identify decision rules:**
   - "If [parameter] > [threshold], choose Option A"
   - "Option B works well unless adoption < 15%"

**Deliverable:** Analysis memo with trade-offs and decision criteria

### Phase 6: Stakeholder Communication (2-3 hours)

**Goal:** Present trade-offs; support decision

**Format:**
1. **Executive summary slide:**
   - Decision statement
   - Recommendation (if requested)
   - Key trade-off (1 sentence)
   - Risk/regret summary

2. **Options comparison table:**
   - EV, P05, P50, P95, Win Rate, Mean Regret

3. **Scenario comparison:**
   - Show performance across worldviews

4. **Sensitivity chart:**
   - Top 5 parameters driving outcomes

5. **Decision criteria:**
   - "Choose A if you prioritize [X]; choose B if you prioritize [Y]"

**Anti-pattern:** Don't bury decision-maker in details. Most stakeholders need <10 slides.

### Phase 7: Decision & Monitoring (1 hour + ongoing)

**Goal:** Document decision; define success metrics

**Steps:**
1. **Record decision rationale:**
   - Which option was chosen?
   - What trade-offs were accepted?
   - What assumptions were critical?

2. **Define observable metrics:**
   - What can we measure post-decision?
   - What thresholds trigger re-evaluation?

3. **Set calibration review:**
   - 30/60/90 days: Compare projections to actuals
   - Update parameter ranges for future decisions

**Deliverable:** Decision memo + monitoring dashboard

---

## 4. Parameter Elicitation Best Practices

### 4.1 Three-Source Approach

**Combine:**
1. **Expert Judgment:** Domain expertise (CTO, product managers, engineers)
2. **Historical Data:** Internal metrics, A/B tests, cohort analysis
3. **External Benchmarks:** Industry reports, academic research, competitor analysis

**Hierarchy:**
- If data exists: Use P05/P50/P95 from historical distribution
- If no data but similar proxy: Adjust proxy data with expert multipliers
- If no data or proxy: Pure expert judgment (triangular distribution)

### 4.2 Elicitation Techniques

#### Technique 1: Triangular Distribution (Low/Mode/High)

**Best for:** Expert judgment when no data exists

**Process:**
1. **Mode (most likely):** "What's your best estimate?"
2. **Low (pessimistic):** "In the worst 10% of cases, what's the lower bound?"
3. **High (optimistic):** "In the best 10% of cases, what's the upper bound?"

**Example:**
- Question: "What % of users will adopt the new feature?"
- Expert response:
  - Mode: 30% (typical adoption for features like this)
  - Low: 10% (if messaging is poor and competition launches similar)
  - High: 50% (if we nail onboarding and get press coverage)

#### Technique 2: Historical Data Percentiles

**Best for:** Parameters with measurable history

**Process:**
1. Extract raw data (transactions, conversions, costs)
2. Calculate P05, P50, P95 from empirical distribution
3. Adjust if future expected to differ from past (expert multiplier)

**Example:**
- Parameter: Baseline failure rate
- Historical data: 1.8%, 5.2%, 9.7% (P05/P50/P95 from Q1-Q3 2025)
- Adjustment: None (system unchanged)
- Final range: 2% / 5% / 10% (rounded)

#### Technique 3: Decomposition

**Best for:** Complex parameters with multiple drivers

**Process:**
1. Break parameter into components
2. Elicit ranges for each component
3. Combine via formula

**Example:**
- Parameter: Revenue per user
- Components:
  - Avg order value: €45 ± €10
  - Orders per year: 2.5 ± 0.5
  - Conversion rate: 4% ± 1%
- Combined: €45 × 2.5 × 0.04 = €4.50/user (calculate for each sampled component)

### 4.3 Calibration & Validation

#### Red Flags

| Symptom | Likely Issue | Fix |
|---------|--------------|-----|
| Range too narrow (High/Low < 1.5x) | Overconfidence | Challenge expert: "What could go wrong?" |
| Mode far from historical median | Misaligned judgment | Show data; ask expert to justify difference |
| All parameters independent | Ignoring correlations | Identify key dependencies (e.g., adoption ↔ revenue) |
| P05 outcome implausible | Bad tail assumptions | Simulate extreme cases; validate plausibility |

#### Cross-Validation

1. **Consistency check:** Do parameters imply reasonable overall outcome?
   - E.g., if adoption = 50%, uplift = 20%, does total revenue make sense?

2. **Extreme scenario test:** Sample P05 for all params; is combined outcome plausible?

3. **Expert review:** Show parameter summary to independent expert; do ranges pass smell test?

### 4.4 Documentation Requirements

For each parameter, document in [assumptions.md](simulator/assumptions.md):

1. **Definition:** What does this parameter represent?
2. **Baseline/Mode:** Central estimate
3. **Range (Low/High):** Plausible bounds
4. **Rationale:** Why this range? (data source, expert logic, benchmark)
5. **Sensitivity:** How much does outcome change if this varies?

**Example:**
```markdown
### extension_uptake

**Definition:** % of customers who opt-in to express checkout feature within 90 days

**Baseline:** 30%

**Range:** 10% (pessimistic) to 50% (optimistic)

**Rationale:**
- Historical data: Prior optional features averaged 25% adoption (range: 12-42%)
- Expert input: Product team estimates 30% based on user research
- External benchmark: Competitor feature adoption reported as 20-35% (TechCrunch)

**Sensitivity:** High (Spearman correlation +0.72 in base scenario)
```

---

## 5. Model Validation & Calibration

### 5.1 What Validation Means (and Doesn't)

**This is NOT forecasting**, so we cannot:
- Backtest against historical decisions (one-time choices have no history)
- Calculate prediction accuracy (not predicting a single outcome)
- Optimize for forecast error minimization

**Instead, we validate:**
- Face validity: Do results align with expert intuition?
- Plausibility: Are extreme outcomes still possible?
- Stability: Do results hold across seeds/scenarios?
- Utility: Does analysis improve decision quality?

### 5.2 Pre-Decision Validation

#### 1. Face Validity Check

**Process:** Show results to domain experts (CTO, product leads) and ask:
- "Does the ranking of options surprise you?"
- "Are any outcomes impossible or absurd?"
- "Do sensitivity results match your intuition about key drivers?"

**Pass criteria:** Experts say "This makes sense" or "Interesting, I didn't expect that but I see why"

**Fail criteria:** "This is clearly wrong" → investigate parameter error

#### 2. Extreme Scenario Testing

**Process:** Manually set all parameters to P05 or P95; check combined outcome

**Example:**
- All pessimistic: Baseline failure 10%, churn 25%, adoption 10%
- Combined outcome: €-500K
- Question: Is this plausible? (Yes, if everything goes wrong, we'd lose this much)

**Pass criteria:** Extreme outcomes are plausible (unlikely but possible)

**Fail criteria:** Extreme outcomes violate physical/business constraints

#### 3. Stability Testing

**Process:** Run simulation with 5 different seeds; compare results

**Example:**
| Seed | Option A EV | Option B EV | Winner |
|------|-------------|-------------|--------|
| 1 | €420K | €365K | A |
| 2 | €425K | €368K | A |
| 3 | €418K | €362K | A |
| 4 | €422K | €370K | A |
| 5 | €419K | €364K | A |

**Pass criteria:** EV varies <5%; winner consistent across seeds

**Fail criteria:** Different seeds produce different winners → need more worlds (increase n_worlds)

#### 4. Sensitivity Validation

**Process:** Examine top 5 correlated parameters; confirm they align with domain knowledge

**Example:**
- Expected: Adoption rate should be highly correlated with outcome (if new feature)
- Observed: Spearman correlation = +0.72
- Conclusion: Validates model structure

**Fail criteria:** Low-importance parameters show high correlation → model mis-specified

### 5.3 Post-Decision Calibration

#### 1. Monitor Actuals vs. Projections

**Process:** After 30/60/90 days, compare real metrics to simulation projections

**Example:**
| Parameter | Projected (P50) | Actual | Assessment |
|-----------|-----------------|--------|------------|
| Adoption rate | 30% | 34% | Within range (10-50%) ✅ |
| Failure reduction | 70% | 65% | Within range (40-80%) ✅ |
| Regression events | 15% prob | 1 minor | Better than expected |

**Calibration actions:**
- If actual within projected range: Ranges well-calibrated
- If actual outside range: Update ranges for next decision
- If actual consistently at edge: Expert bias (over-optimistic or pessimistic)

#### 2. Retrospective Analysis

**Questions:**
1. Which parameters did we estimate accurately?
2. Which did we miss entirely? (e.g., competitive response)
3. Did decision-maker choose based on correct criterion? (EV vs. regret)
4. Would we make same decision knowing what we know now?

**Outcome:** Updated parameter library + process improvements

---

## 6. Interpreting Results

### 6.1 Decision-Maker Profiles

Different stakeholders prioritize different criteria:

| Profile | Prioritizes | Recommended Metric |
|---------|-------------|-------------------|
| **Risk-Neutral** | Expected return | Expected Value (EV) |
| **Risk-Averse** | Avoid disaster | P05 (downside protection) |
| **Regret-Averse** | Minimize missed opportunity | Mean Regret or P95 Regret |
| **Opportunistic** | Capture upside | P95 (best-case scenarios) |
| **Consensus-Builder** | Consistency | Win Rate across scenarios |

### 6.2 Interpretation Patterns

#### Pattern 1: Clear Winner

**Scenario:** Option A dominates on EV, P05, and regret

**Interpretation:** Clear winner; recommend confidently

**Example:**
| Option | EV | P05 | Mean Regret |
|--------|-----|-----|-------------|
| A | €500K | €100K | €50K |
| B | €300K | -€50K | €220K |

**Conclusion:** Choose A (dominates on all criteria)

#### Pattern 2: EV vs. Risk Trade-off

**Scenario:** Option A has higher EV; Option B has better P05

**Interpretation:** Risk appetite determines choice

**Example:**
| Option | EV | P05 | Mean Regret |
|--------|-----|-----|-------------|
| A | €500K | -€100K | €80K |
| B | €400K | €50K | €120K |

**Conclusion:**
- Risk-neutral: Choose A (higher EV)
- Risk-averse: Choose B (positive P05, no downside)

#### Pattern 3: Scenario-Dependent

**Scenario:** Winner changes across conservative/base/aggressive scenarios

**Interpretation:** Worldview drives decision; expose implicit assumptions

**Example:**
| Option | Base EV | Conservative EV | Aggressive EV |
|--------|---------|-----------------|---------------|
| A | €400K | €200K | €600K |
| B | €420K | €380K | €450K |

**Conclusion:**
- If optimistic: Choose A (higher upside)
- If pessimistic: Choose B (consistent across scenarios)
- Force stakeholders to declare their worldview

### 6.3 Sensitivity Insights

#### High-Correlation Parameters = Data Collection Priorities

**Example:**
| Parameter | Correlation | Implication |
|-----------|-------------|-------------|
| Adoption rate | +0.72 | Critical uncertainty; invest in user research |
| Baseline failure rate | -0.65 | Need better instrumentation/monitoring |
| Revenue per user | +0.51 | Moderate impact; existing data likely sufficient |

**Action:** Run A/B test or pilot to narrow adoption rate uncertainty before full launch

#### Low-Correlation Parameters = Don't Overthink

**Example:**
| Parameter | Correlation | Implication |
|-----------|-------------|-------------|
| Regression cost | +0.12 | Low impact; use rough estimate |

**Action:** Don't spend time refining this parameter; outcome insensitive to it

---

## 7. Common Pitfalls & Solutions

### Pitfall 1: False Precision

**Symptom:** Ranges too narrow; P05 and P95 very close to P50

**Why it's bad:** Underestimates uncertainty; overconfident decision

**Solution:**
- Challenge experts: "What could go wrong? What could surprise us?"
- Require minimum range width (e.g., High ≥ 2x Low)

### Pitfall 2: Analysis Paralysis

**Symptom:** Weeks spent refining parameters; never reach decision

**Why it's bad:** Delay costs; diminishing returns on precision

**Solution:**
- Timeboxing: 2 weeks max for analysis
- "Satisficing" mindset: Good enough > perfect
- Sensitivity analysis: Stop refining low-correlation parameters

### Pitfall 3: Confirmation Bias

**Symptom:** Parameters tuned to favor preferred option

**Why it's bad:** Defeats purpose of objective analysis

**Solution:**
- Version-control config.yaml (audit trail)
- Peer review parameter ranges before simulation
- Blind elicitation (don't show results until ranges locked)

### Pitfall 4: Ignoring Regret

**Symptom:** Decision based on EV alone

**Why it's bad:** Leads to choices with high opportunity cost

**Solution:**
- Always present regret alongside EV
- Frame decision: "If we're wrong, how wrong could we be?"

### Pitfall 5: Treating as Forecast

**Symptom:** Stakeholders ask "What will actually happen?"

**Why it's bad:** Misuse of framework; erodes trust when reality differs

**Solution:**
- Clear messaging: "This compares options under uncertainty, not predicts outcomes"
- Show full distributions, not just EV
- Emphasize decision quality over outcome quality

### Pitfall 6: No Post-Decision Learning

**Symptom:** Never compare projections to actuals

**Why it's bad:** No calibration improvement; repeated mistakes

**Solution:**
- Mandatory 90-day retrospective
- Update parameter library with actuals
- Document what was missed (e.g., competitive response)

---

## 8. Advanced Techniques

Note: These techniques are not yet implemented. They represent potential enhancements for future versions.

### 8.1 Correlation Structures

**When:** Parameters are not independent (e.g., high adoption → higher revenue)

**Approach:**
1. Identify correlated pairs (domain knowledge or historical data)
2. Use rank correlation (Spearman) to preserve marginal distributions
3. Sample from copula (e.g., Gaussian, Clayton)

**Implementation:** Requires scipy.stats copula functions (not in MVP)

### 8.2 Value of Information (VoI) Analysis

**When:** Deciding whether to invest in reducing uncertainty (e.g., run pilot study)

**Approach:**
1. Calculate regret with current uncertainty
2. Simulate decision with perfect information about key parameter
3. VoI = Expected regret reduction

**Example:**
- Current mean regret with uncertain adoption: €120K
- Mean regret if we knew adoption rate: €40K
- VoI = €80K → Justified to spend up to €80K on adoption study

### 8.3 Multi-Period Dynamics

**When:** Decisions affect future options (real options value)

**Approach:**
1. Model decision as sequence of stages (e.g., pilot → full launch)
2. At each stage, evaluate options conditional on prior outcomes
3. Calculate value of flexibility (option to pivot)

**Example:**
- Stage 1: Small pilot (€100K)
- If pilot succeeds: Full launch (€1M)
- If pilot fails: Kill project (€0 additional spend)
- Value of staging = Reduced downside risk

### 8.4 Optimization (Use with Caution)

**When:** Continuous decision variable (e.g., pricing, budget allocation)

**Approach:**
1. Define objective function (e.g., maximize EV, minimize regret)
2. Sample parameter space
3. Find optimal value via grid search or gradient descent

**Warning:** Optimization removes human judgment; use only when objective is unambiguous

---

## Summary:

By following this methodology, you apply:

1. **Structuring decisions:** Framing options, metrics, time horizons
2. **Uncertainty quantification:** Distributional thinking, parameter elicitation
3. **Comparative analysis:** Multi-criteria evaluation (EV, risk, regret)
4. **Sensitivity analysis:** Identifying key uncertainties
5. **Stakeholder communication:** Explaining trade-offs, visual decision aids
6. **Model validation:** Face validity, stability, calibration
7. **Process design:** Repeatable framework for organizational use
8. **Continuous learning:** Retrospectives and parameter updating

---

**Related Documents:**
- [CASE_STUDY.md](CASE_STUDY.md) - Real-world application example
- [simulator/assumptions.md](simulator/assumptions.md) - Parameter documentation
- [simulator/model_spec.md](simulator/model_spec.md) - Technical model specification

**Document Version:** 1.0
**Last Updated:** January 30, 2026
