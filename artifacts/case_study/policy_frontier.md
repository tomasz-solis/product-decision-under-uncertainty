- The first table is the full-option frontier. It re-runs the whole policy and records the first threshold change that flips the recommendation.

| Threshold | Current value | Raw switching value | Display switching value | First switching option | Switch type | Direction | Interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Downside floor | €-300,000 | not observed | not observed | not observed | no switch observed | not observed | Re-evaluating the full option set across the tested threshold domain did not change the selected option. |
| Regret cap | €450,000 | 609,360.6 | €609,360.6 | Do Nothing | exact match | more permissive | Full-option sweep: Stabilize Core switches to Do Nothing when the regret cap moves up to 609,360.57 EUR. |
| EV tolerance | €100,000 | not observed | not observed | not observed | no switch observed | not observed | Re-evaluating the full option set across the tested threshold domain did not change the selected option. |
- The second table is secondary context. It follows the main comparison option, which can be the policy runner-up or the best excluded alternative depending on the branch.

| Threshold | Current value | Comparison threshold | Status | Interpretation |
| --- | --- | --- | --- | --- |
| Downside floor | €-300,000 | €-420,803.7 | runner-up threshold reached | Feature Extension becomes eligible on downside once the floor is relaxed to -420,803.72 EUR. |
| Regret cap | €450,000 | not needed | already non-binding | Feature Extension already passes the regret cap. |
| EV tolerance | €100,000 | not needed | not binding | EV tolerance is not the binding threshold because both options are not yet eligible. |

| Threshold | Tested range | Selection switched? | Switching option(s) |
| --- | --- | --- | --- |
| EV tolerance | €75,000 to €125,000 | no | none |
| Regret cap | €400,000 to €650,000 | yes | Do Nothing |
| Downside floor | €-350,000 to €-250,000 | no | none |