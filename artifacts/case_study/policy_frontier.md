- The first table is the full-option frontier. It re-runs the whole policy and records the first threshold change that flips the recommendation. That is the actual decision question.
- The second table is secondary. It shows when the current runner-up clears its own blocking threshold, which is useful context but not the same as the first switch.

| Threshold | Current value | Raw switching value | Display switching value | First switching option | Switch type | Direction | Interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Downside floor | €-300,000 | -145,373.6 | €-145,373.6 | Feature Extension | grid bracket | more restrictive | Full-option sweep: Do Nothing switches to Feature Extension once the downside floor moves up past -145,373.57 EUR. |
| Regret cap | €450,000 | 197,615.8 | €197,615.8 | Feature Extension | grid bracket | more restrictive | Full-option sweep: Do Nothing switches to Feature Extension once the regret cap moves down past 197,615.84 EUR. |
| EV tolerance | €100,000 | not observed | not observed | not observed | no switch observed | not observed | Re-evaluating the full option set across the tested threshold domain did not change the selected option. |

| Threshold | Current value | Runner-up threshold | Status | Interpretation |
| --- | --- | --- | --- | --- |
| Downside floor | €-300,000 | €-544,674.9 | runner-up threshold reached | Feature Extension becomes eligible on downside once the floor is relaxed to -544,674.89 EUR. |
| Regret cap | €450,000 | not needed | already non-binding | Feature Extension already passes the regret cap. |
| EV tolerance | €100,000 | not needed | not binding | EV tolerance is not the binding threshold because both options are not yet eligible. |

| Threshold | Tested range | Selection switched? | Switching option(s) |
| --- | --- | --- | --- |
| EV tolerance | €75,000 to €125,000 | no | none |
| Regret cap | €147,615.8 to €497,615.8 | yes | Feature Extension |
| Downside floor | €-350,000 to €-100,000 | yes | Feature Extension |