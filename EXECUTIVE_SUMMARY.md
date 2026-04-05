# Executive Summary

This is the shortest path to the current published result. For assumptions, formulas, and the evidence workflow, start from [README.md](README.md).

## Recommendation

<!-- GENERATED:EXEC_SUMMARY_RECOMMENDATION:START -->
- Recommendation: **Do Nothing**.
- Policy: `guardrailed_expected_value`.
- Why it wins: Do Nothing is the only option that passes both guardrails.
- Best excluded alternative: **Feature Extension** has the strongest excluded EV case, but it misses the downside floor by about €247,304.
- Expected-value comparison: the selected option trails **Feature Extension** by €44,651.
- Published run: `20,000` worlds, seed `42`, annual volume `250,000`, horizon `2` years, discount rate `8%`, declared model version `4.0.0`.
<!-- GENERATED:EXEC_SUMMARY_RECOMMENDATION:END -->

## Default run table

<!-- GENERATED:EXEC_SUMMARY_RESULTS:START -->
| Option | Expected Value | P05 | Median | P95 |
| --- | --- | --- | --- | --- |
| Feature Extension | €-60,019 | €-547,304 | €-91,349 | €530,971 |
| Do Nothing | €-104,671 | €-145,374 | €-102,378 | €-69,716 |
| Stabilize Core | €-207,577 | €-693,811 | €-249,949 | €431,055 |
| New Capability | €-956,398 | €-1,382,691 | €-967,359 | €-492,711 |
<!-- GENERATED:EXEC_SUMMARY_RESULTS:END -->

## What this summary is for

- It tells you which option the current policy selects.
- It shows the default-run value distribution for each option.
- It leaves the assumption chain and modeling detail to the longer docs.
