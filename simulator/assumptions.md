# Assumptions

This simulator is built on a small set of explicit assumptions about how the product behaves under different decisions.

These are not facts and not predictions.
They are working beliefs that make trade-offs visible and comparable under uncertainty.

These assumptions are written down to make clear what has to be true for a decision to work.

Disagreement with these assumptions is expected and healthy.

---

## Decrease in friction point rate (stabilize_core)

### What it represents

The expected reduction in how often users encounter friction in the product’s most common, end-to-end usage flows.

This assumption is about improving the quality of what already exists.
It does not assume increased adoption or new demand.
It reflects fewer errors, fewer breakdowns, and a smoother experience for users who are already trying to complete core workflows.

### Why this range exists

The lower bound assumes fixes are applied to parts of the flow that fewer users reach, or to variants that are not the default path. Friction is reduced where the fix applies, but the overall impact is limited because many users never encounter that step.

The upper bound assumes fixes target earlier or widely used steps in the core flow, or address several steps together. Reducing friction earlier increases the chance that users reach later steps, so the effect can compound across the flow. When multiple steps are improved, the impact is larger because key friction points are removed at several points.

### What would break this assumption

This assumption breaks if we are not seeing all the friction that actually matters. If important issues are missing from our tracking, the reduction we model will not reflect reality.

It also breaks if users do not use the product the way we think they do. If real usage patterns differ from the intended flow, fixing the “right” steps may have little impact.

Another failure mode is user error. If people make mistakes and the system does not guide them toward the correct path, or prevent incorrect execution, reducing technical friction alone will not be enough.

The assumption also weakens if added flexibility creates confusion. Offering more options can make the product harder to use, even if individual steps are improved.

Finally, this does not hold if the main sources of friction sit outside our control, such as third-party services or providers that dominate key parts of the flow.

---

## Observed friction rate

### Definition

Share of attempts in core, end-to-end usage flows where the user hits a blocking or clearly degrading issue.

Here, friction shows up as cases where the flow does not work as intended due to errors, breakdowns, or system-level issues.

It does not cover all types of friction. It focuses on problems that are visible, measurable, and clearly disruptive.

### Baseline value

5%

### Plausible range

2% – 10%

### Rationale

The baseline reflects a rough estimate of how often core flows break down in ways that interrupt user progress.

The lower bound assumes most issues are edge cases or are handled reasonably well.
The upper bound assumes breakdowns are more common, especially around integrations, less-tested paths, or high-volume usage.

The range is wide to reflect uncertainty in both what we currently observe and how friction shows up across different users and situations.

### Notes

This represents only one category of friction.
Other forms, such as confusion, hesitation, or workarounds, are not captured here.

---

## Downstream impact of friction

### Definition

What tends to happen after a user hits friction in a core usage flow.

This describes how friction can lead to negative outcomes such as higher churn risk, reduced engagement, loss of trust, or slower progress toward value.

Churn is one possible outcome, but not the only one.

### Baseline value

15% relative increase

### Plausible range

5% – 30%

### Rationale

The baseline reflects the belief that friction in core flows usually has consequences, even when users do not churn immediately.

The lower bound assumes many users recover, tolerate the issue, or have strong reasons to continue.
The upper bound assumes friction meaningfully damages trust or confidence, increasing the chance of abandonment or disengagement over time.

User reactions vary widely, and the effects of friction are often delayed, so this range is kept broad.

### Notes

This is an aggregate effect.
It does not assume immediate churn or a single, fixed path from friction to outcome.

---

## Operational cost of friction

### Definition

Average operational cost created when friction occurs in a core usage flow.

This includes support handling, retries, refunds, manual work, and other operational effort needed to unblock or recover from the issue.

It focuses on costs that scale with friction, not fixed team or platform costs.

### Baseline value

€8 per friction event

### Plausible range

€3 – €15

### Rationale

The baseline reflects a rough average of the operational effort typically required when users run into problems in core flows.

The lower bound assumes most issues are resolved quickly or at low cost, with limited manual intervention.
The upper bound assumes more complex cases that require repeated handling, escalation, or external coordination.

We keep this range wide because the cost of friction varies significantly by issue type, user context, and how early problems are detected.

### Notes

This captures only direct operational costs.
Indirect effects such as reputational damage or long-term customer impact are handled elsewhere in the model.

---

## Feature extension uptake rate

### Definition

Share of eligible users who choose to adopt or actively use the feature extension once it is available.

This reflects voluntary uptake by existing users.
It does not assume the feature is required, default, or universally relevant.

### Baseline value

20%

### Plausible range

10% – 35%

### Rationale

The baseline reflects the belief that optional extensions are adopted by a minority of users, even when they provide clear additional value.

The lower bound assumes the feature appeals only to a narrow segment or requires behavioral change that many users avoid.
The upper bound assumes the extension is well integrated into existing workflows and addresses a common, clearly felt need.

The range is wide because uptake depends heavily on how visible the feature is, how much effort it requires, and how closely it aligns with existing user goals.

### Notes

This is treated as an assumption, not a forecast.
If the feature, target users, or rollout approach differ materially, this range should be revisited.

---

## Extension loss rate

### Definition

Expected loss rate associated with the feature extension.

This includes direct losses (e.g. defaults or disputes) and indirect cost proxies where losses are not directly observable.

### Baseline value

3%

### Plausible range

1% – 6%

### Rationale

The baseline reflects the belief that extensions introducing financial exposure or additional risk tend to generate some level of loss, even under conservative assumptions.

The lower bound assumes strong controls, limited exposure, or a user base with relatively low risk.
The upper bound assumes weaker controls, higher exposure, or greater variability in user behavior.

The range is intentionally broad because loss behavior is sensitive to product design details, enforcement mechanisms, and external conditions.

### Notes

This parameter represents expected losses at an aggregate level.
It does not model tail risk or rare extreme events separately.

---

## Delivery regression risk

### Definition

Risk of introducing regressions or incidents as a result of delivery work.

This reflects the chance that changes intended to improve the product create new issues, along with the potential impact when that happens.

### Baseline value

Low probability, high impact

### Plausible range

Modeled as a heavy-tailed distribution

### Rationale

Most delivery changes are safe and have limited side effects.
However, a small number of changes can trigger serious regressions with outsized impact.

This assumption reflects that asymmetry: frequent low-impact outcomes combined with rare but costly failures.
The exact probability is uncertain, but the risk cannot be ignored when evaluating irreversible or tightly coupled changes.

### Notes

This captures delivery-related risk only.
It does not include ongoing operational issues unrelated to recent changes.

---

## Important note

All values are intentionally approximate.
The model is designed to test decision robustness across uncertainty,
not to rely on precise estimates.
