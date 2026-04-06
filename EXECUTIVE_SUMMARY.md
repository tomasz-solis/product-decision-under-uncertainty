# Executive Summary

This is the shortest path to the current published result. For assumptions, formulas, and the evidence workflow, start from [README.md](README.md).

## Recommendation

<!-- GENERATED:EXEC_SUMMARY_RECOMMENDATION:START -->
- Recommendation: **Stabilize Core**.
- Policy: `guardrailed_expected_value`.
- Why it wins: No option clears both guardrails, so the policy falls back to expected value.
- Guardrail reality: no option passes both guardrails, so **Stabilize Core** wins on expected value.
- Best remaining excluded alternative: **Feature Extension**.
- Expected-value comparison: the selected option leads **Feature Extension** by €176,174.
- Published run: `20,000` worlds, seed `42`, annual volume `250,000`, horizon `2` years, discount rate `8%`, declared model version `5.0.0`.
<!-- GENERATED:EXEC_SUMMARY_RECOMMENDATION:END -->

## Default run table

<!-- GENERATED:EXEC_SUMMARY_RESULTS:START -->
| Option | Expected Value | P05 | Median | P95 |
| --- | --- | --- | --- | --- |
| Stabilize Core | €373,233 | €-360,761 | €326,965 | €1,268,354 |
| Feature Extension | €197,060 | €-420,804 | €156,314 | €944,158 |
| Do Nothing | €-104,671 | €-145,374 | €-102,378 | €-69,716 |
| New Capability | €-991,538 | €-1,408,501 | €-1,001,211 | €-539,891 |
<!-- GENERATED:EXEC_SUMMARY_RESULTS:END -->

## What this summary is for

- It tells you which option the current policy selects.
- It shows the default-run value distribution for each option.
- It leaves the assumption chain and modeling detail to the longer docs.
