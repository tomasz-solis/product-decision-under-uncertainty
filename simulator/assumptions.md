# Assumptions and Parameterization

This simulator relies on explicit assumptions.
Each assumption is documented with a definition, baseline value, plausible range,
and rationale.

The intent is to make uncertainty visible and adjustable rather than hidden.

---

## Baseline failure rate

**Definition**  
Probability that a user experiences a failure in the core product flow.

**Baseline value**  
5%

**Plausible range**  
2% – 10%

**Rationale**  
Anchored in publicly reported payment failure benchmarks and industry discussions.
The range reflects variation across geographies, user segments, and integration
maturity.

**Notes**  
Sampled as a distribution rather than a fixed value.

---

## Failure-to-churn impact

**Definition**  
Incremental increase in churn probability following a failed core interaction.

**Baseline value**  
15% relative increase

**Plausible range**  
5% – 30%

**Rationale**  
Based on studies of checkout friction, payment failures, and trust erosion in
financial products.

**Notes**  
This parameter captures behavioral impact, not immediate churn alone.

---

## Cost per failure

**Definition**  
Average operational cost associated with a failure, including support,
retries, refunds, and manual intervention.

**Baseline value**  
€8 per failure

**Plausible range**  
€3 – €15

**Rationale**  
Anchored in support cost benchmarks, chargeback fees, and incident handling
estimates from public sources.

---

## Feature extension uptake rate

**Definition**  
Share of eligible users who adopt or use the feature extension.

**Baseline value**  
20%

**Plausible range**  
10% – 35%

**Rationale**  
Anchored in reported adoption rates for optional payment or credit-like features.

---

## Extension loss rate

**Definition**  
Expected loss rate associated with the extension (e.g. defaults, disputes,
or indirect regulatory cost proxies).

**Baseline value**  
3%

**Plausible range**  
1% – 6%

**Rationale**  
Anchored in aggregated BNPL and consumer credit delinquency statistics.

---

## Delivery regression risk

**Definition**  
Probability and impact of introducing regressions or incidents during delivery.

**Baseline value**  
Low probability, high impact

**Plausible range**  
Modeled as a heavy-tailed distribution

**Rationale**  
Based on incident postmortems and software reliability literature showing that
most changes are safe, but rare failures are costly.

---

## Important note

All values are intentionally approximate.
The model is designed to test decision robustness across uncertainty,
not to rely on precise estimates.
