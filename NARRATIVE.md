# The Decision: A Walk-Through

This file tells the story of the decision this model frames. If you want the
methodology, read [METHODOLOGY.md](METHODOLOGY.md). If you want the code,
start with [simulator/simulation.py](simulator/simulation.py). This document
is neither — it is the *decision* itself: why it was hard, what the model
said, and what you would actually do with the output in a real review.

---

## The situation

A checkout platform is under reliability pressure. The failure rate on the
legacy flow is somewhere between 4% and 15% of volume — the range exists
because the team does not have clean telemetry on it yet. The platform still
needs forward motion: stopping investment to stabilise is not obviously the
right call, because the business also needs new capability and adoption growth.

Four options are on the table. Only one primary path can be funded for the
next 24 months:

1. **Do Nothing** — absorb the drift, keep the current path
2. **Stabilize Core** — refactor the legacy checkout flow and remove the
   failure exposure
3. **Feature Extension** — add an optional express path that helps adopters
   but leaves the legacy drift in place
4. **New Capability** — build a fraud-prevention layer that adds upside on
   successful volume while legacy drift continues

In a standard business case, you would estimate the expected payoff for each,
pick the highest, and present three scenarios (base / upside / downside). This
is not that kind of decision.

---

## Why expected value alone is the wrong objective here

Here is the trap: Feature Extension has the highest expected value in some
plausible worlds. In a growth-friendly environment with high extension uptake,
it leads on EV by a meaningful margin. A naive "pick the highest EV option"
rule would often select it.

But expected value treats a -€420k tail and a +€100k boost symmetrically, as
long as the mean comes out right. That is fine if the decision maker is
genuinely risk-neutral — willing to absorb a bad result in exchange for a
good expected outcome. Most platform investment decisions are not made that
way. A P05 of -€420k on a 24-month investment is a real outcome that a
leadership team has to explain, not a statistical artefact to average away.

The model encodes this directly with two guardrail conditions:

- **Downside floor**: the P05 outcome (the worst result in the bottom 5% of
  sampled worlds) must stay above -€300k. Options that fail this are too
  likely to destroy value.
- **Regret cap**: the mean regret (how much worse than the best available
  option the decision looks in expectation, over all sampled worlds) must stay
  below €450k. Options that fail this expose the team to large opportunity
  cost in a wide range of scenarios.

An option must clear both guardrails to stay eligible. Within the eligible
set, the highest EV wins. If no option clears both, the policy falls back to
EV — not because EV is now right, but because the guardrails have told you
something important: none of the options is safe, so you pick the least bad.

This is not the model making a decision. It is the model mapping the team's
risk preferences — encoded as thresholds — onto the option set.

---

## What the published run told us

In the default mid-range pressure scenario, no option clears both guardrails.

- **Stabilize Core**: fails the downside floor (P05 = -€361k, floor is -€300k)
- **Feature Extension**: fails the downside floor worse (P05 = -€421k)
- **Do Nothing**: passes the downside floor, but its mean regret is €609k
  against a cap of €450k — too much opportunity cost
- **New Capability**: fails both, severely

The policy falls back to EV. Stabilize Core leads on expected value at
€373k, ahead of Feature Extension at €197k. The recommendation is
**Stabilize Core**, but the honest label on that recommendation is:
*the best option given that none of them is safe*.

That is a different conversation to have with a leadership team than "Stabilize
Core wins." The model's job here is to surface that honesty, not to obscure it
behind a clean recommendation table.

---

## The frontier — what would change the call?

This is the most useful output for a real stakeholder review.

The policy frontier re-runs the full model across a grid of threshold values
and records the first change that flips the recommendation. In the published
run:

- **Downside floor**: could move anywhere from -€350k to -€250k without
  changing the recommendation. Stabilize Core still wins on EV even if you
  tighten the floor.
- **Regret cap**: must be loosened to approximately **€609k** before the
  recommendation flips — at that point, Do Nothing becomes eligible (it passes
  the downside floor), and under this more permissive cap it beats Stabilize
  Core on EV. If your real regret tolerance is closer to €600k than €450k,
  the call is different.
- **EV tolerance**: not binding in this run. The EV gap between options is
  large enough that the tolerance band does not change selection.

In a real review, you bring this table to the stakeholders, not the
recommendation. You ask: "Our current regret cap is €450k. If that threshold
moved to €600k — if we were willing to accept more opportunity cost in
exchange for keeping the legacy drift option open — would you want to revisit
the recommendation?" That question is productive. "The model says Stabilize
Core" is not.

Separately: Feature Extension becomes eligible on the downside guardrail once
the floor relaxes to -€421k. If the business decides the actual downside
tolerance is €420k rather than €300k (perhaps because the investment can be
staged), Feature Extension re-enters the eligible set. This is a scope
discussion, not a modelling one. The frontier makes it legible.

---

## What the model cannot tell you

**The correlations are elicited, not fit.** The Gaussian copula in the joint
sampler uses rank correlations: 0.55 between baseline failure rate and failure
cost, 0.60 between failure rate and churn, 0.40 between extension uptake and
extension value, 0.45 between regression event probability and severity. These
numbers were set by judgment, not calibrated from data. The
dependency-value frontier in the robustness section shows how sensitive the
recommendation is to each of these — but you should know going in that they
are elicited assumptions.

**The options are cleaner than real options usually are.** Stabilize Core
and Feature Extension are modelled as mutually exclusive and fully separable.
In practice, a real platform decision might allow partial investment in both,
staged gates, or options that unlock further options. The four-option frame is
a deliberate simplification to make the decision legible.

**The drift cost is a single uncertain number.** In reality, do-nothing drift
is not a smooth €50k/year cost — it is a collection of discrete pain points,
incident risks, and opportunity costs that compound differently across teams.
The model collapses this into `do_nothing_drift_cost_eur`. That is a
meaningful simplification and its sensitivity drives the Do Nothing option
almost entirely, which is why it shows up in the driver analysis.

**There is no real company behind this.** The parameters are calibrated
against public benchmarks and domain judgment, not against any organisation's
telemetry. The point of the case is the decision pattern, not the numbers.

---

## How to use this pattern in a real review

Three practical suggestions:

**Bring the frontier, not the recommendation.** The single most useful thing
this approach produces is the threshold sensitivity analysis. A leadership team
that cannot agree on whether to approve Stabilize Core can often agree on
whether their actual downside tolerance is -€300k or -€420k. Once that
conversation happens, the model selection follows automatically. The
recommendation is an output of their preferences, not an input to the
discussion.

**Name the fallback clearly.** When no option clears the guardrails, the policy
falls back to EV. That fallback should be stated explicitly in any review —
it means the team is operating outside safe bounds and the decision carries
more downside exposure than the policy was designed for. Burying it in a clean
recommendation table loses information the stakeholders need.

**Use the stress tests to decide what evidence to gather next.** The
robustness section runs directional stress tests on the top material drivers.
In this case, `do_nothing_drift_cost_eur` drives the Do Nothing option almost
entirely (rank correlation ≈ 1.0), which means if there is any way to get
better telemetry on internal drift costs before making the call, it is worth
getting. The model can tell you which uncertain inputs it is most sensitive to;
those are the inputs worth measuring first.
