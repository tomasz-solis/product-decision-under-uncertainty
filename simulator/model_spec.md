# Model Specification

## Purpose

The simulator compares three structurally different product decisions:

1. Stabilize the core system
2. Ship a feature extension
3. Introduce a new capability

The comparison focuses on robustness under uncertainty rather than point
estimates of success.

The model is designed to answer:
- Which decisions perform well across many plausible worlds?
- Where does downside risk concentrate?
- Which assumptions most strongly affect the outcome?

---

## Modeling approach

- All key inputs are modeled as probability distributions
- Synthetic data is generated via Monte Carlo simulation
- Results are interpreted comparatively, not absolutely

Each simulation run represents one plausible state of the world given the
assumptions and their uncertainty.

---

## What the model does

- Estimates expected value distributions for each option
- Estimates downside outcomes (e.g. worst credible cases)
- Compares regret across decisions
- Highlights which assumptions drive results

---

## What the model does NOT do

- Predict real-world outcomes
- Estimate company-specific metrics
- Recommend a universally optimal strategy
- Replace judgment or product strategy

This model is a structured way to reason about trade-offs, not a source of truth.

---

## Decision options

### Option 1 — Stabilize the Core

Focus capacity on reducing known failure modes in the existing system.
Improve reliability, predictability, and maintainability.
New feature development is intentionally slowed during this period.

### Option 2 — Feature Extension

Ship an extension to the current product that augments the existing flow.
This may change user behavior or reduce exposure to failures without fully
resolving underlying system issues.

### Option 3 — New Capability

Introduce a new capability that expands functional or market scope.
This creates new value streams while increasing complexity and long-term
ownership costs.

---

## Core parameters

The model relies on the following high-level parameters:

- Baseline failure rate
- Failure-to-churn impact
- Cost per failure
- Feature uptake rate
- Credit or extension loss rate
- Delivery regression risk

Each parameter is explicitly documented in `assumptions.md`.
