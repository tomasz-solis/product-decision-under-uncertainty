# Decision Quality Under Uncertainty

## What this is

This tool exists to make decision quality visible under uncertainty.

It is designed to help reason about difficult product and business decisions where:
- outcomes are uncertain,
- experiments are not possible,
- and the cost of being wrong is asymmetric.

The framework compares alternative decisions side by side, including the option to do nothing, using explicit and traceable assumptions.

The goal is not to find the “right” answer, but to understand the trade-offs involved in choosing one option over another.

---

## What this is not

This project is **not** a forecasting tool.

It does not try to predict outcomes or assess accuracy.

It is also **not** an optimization or recommendation engine.

The framework does not choose for you, rank decisions automatically, or replace judgment. It exists to support clearer thinking, not to automate decisions.

---

## How it works (high level)

The framework is intentionally small and opinionated.
It follows a simple, consistent flow.

### Decision framing

Each decision is defined explicitly:
- the decision question,
- the decision owner,
- the available options (including “do nothing”),
- constraints,
- and reversibility.

If these are not clear, the framework is not applied.

### Uncertainty modeling

Key assumptions are expressed as ranges and distributions rather than single numbers.

All assumptions are:
- explicit,
- documented in plain language,
- and traceable.

Synthetic data is used deliberately to explore uncertainty, not to approximate reality.

### Simulation

The same uncertainty draws are used across all options.

This allows outcomes to be compared under identical beliefs, rather than evaluated in isolation.

The simulation exists to support comparison, not to impress or optimize.

### Comparative evaluation

Each option is evaluated using decision-quality metrics rather than point estimates alone.

---

## How to read the outputs

The framework produces several complementary views.

### Expected value

The average outcome across many plausible futures.

This shows what tends to work well, but should never be read alone.

### Downside risk

A view of how bad things can get under unfavorable conditions, when recovery is slow or impossible.

This matters when failures are costly or irreversible.

### Regret

How much worse an option performs compared to the best alternative in the same situation.

Regret highlights decisions that are painful when wrong, even if they look reasonable on average.

### Win rates

How often an option performs best across plausible futures.

This helps distinguish reliable choices from those that rely on optimistic assumptions.

No single metric is sufficient on its own. Look at all of them together.

---

## Example decision

V1 of this project includes a single, fully worked decision case.

The case describes a familiar situation in mature products:
- ongoing issues in part of a product,
- pressure to keep moving forward,
- limited capacity,
- and no safe way to experiment before committing.

The decision involves multiple realistic options, including doing nothing, each with different risks and degrees of irreversibility.

The case is written to be understandable without reading the code.

---

## Intended audience

This project is written for:
- senior product managers,
- senior data practitioners,
- and decision-makers responsible for irreversible or high-risk choices.

It assumes comfort with uncertainty and trade-offs, and does not attempt to simplify decisions beyond what clarity allows.

---

## Design philosophy

Several principles guide this project.

- **Clarity over scale**  
  This is a thinking tool, not a general-purpose library.

- **Judgment over automation**  
  The framework supports thinking; it does not replace it.

- **Transparency over performance**  
  Clear assumptions matter more than clever techniques.

- **Comparison over conclusions**  
  Decisions are evaluated relative to alternatives, not in isolation.

---

## Final note

If this project starts to feel like a model, a product, or a demo, it has gone too far.

The purpose is to reason about decision quality under uncertainty, not to predict outcomes or recommend actions.

---

## How to run

This project currently runs as a local, exploratory tool.

The interface is intentionally minimal. The goal is to make assumptions and trade-offs visible, not to provide a polished application.

### Setup
```bash
pip install -r requirements.txt
```

### Run
```bash
streamlit run app.py
```

The app will open locally in your browser.

From there, you can:

- review the example decision included in the project
- adjust assumptions and ranges
- compare decision options under the same uncertainty draws

There is no persistence layer, user management, or production configuration.
This is deliberate.

### Using historical data to inform ranges

If you have historical data (CSV format), you can upload it directly in the app:

1. Click "Upload data to suggest ranges" in the sidebar
2. Upload your CSV file (must have numeric columns)
3. Map CSV columns to parameters (app will auto-match when possible)
4. Review suggested ranges (5th/50th/95th percentiles)
5. Click "Apply suggested ranges"

The app calculates ranges from your data but you can still adjust them manually afterward.

**Important**: The app suggests ranges based on historical data. You review, adjust, and own the final ranges. Past patterns may not reflect future uncertainty.

For batch processing, use the CLI tool:
```bash
python3 simulator/analyze_data.py your_data.csv --summary --output ranges.yaml
```

### Running tests

To run the test suite:

```bash
pip install -r requirements.txt
pytest test_simulator.py -v
```

The tests cover core simulation logic, parameter sampling, and edge cases.

### Why this is not deployed

The framework is meant to be used close to the decision being discussed.

Running it locally keeps:
- assumptions inspectable,
- iteration cheap,
- ownership clear.

If the tool ever needs to be deployed, that will be a separate decision—made with its own trade-offs.

---

## Documentation

- [METHODOLOGY.md](METHODOLOGY.md) - Step-by-step process for applying the framework
- [CASE_STUDY.md](CASE_STUDY.md) - Illustrative example (€2M platform decision)
- [EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md) - One-page overview for non-technical stakeholders
- [STATUS.md](STATUS.md) - Current state and next steps
- [simulator/assumptions.md](simulator/assumptions.md) - Parameter definitions
- [simulator/model_spec.md](simulator/model_spec.md) - Technical specification

---

## Contact

Tomasz Solis
- [LinkedIn](https://linkedin.com/in/tomaszsolis)
- [GitHub](https://github.com/tomasz-solis)