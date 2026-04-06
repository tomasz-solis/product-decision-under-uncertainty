- This is the policy-defining table for the current run.
- An option must pass both the downside floor and the regret cap to stay eligible.

| Option | Expected Value | P05 | Downside Slack | Mean Regret | Regret Slack | Eligible | Failure Reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Stabilize Core | €373,233 | €-360,761 | €-60,761 | €131,457 | €318,543 | no | fails the downside floor |
| Feature Extension | €197,060 | €-420,804 | €-120,804 | €307,630 | €142,370 | no | fails the downside floor |
| Do Nothing | €-104,671 | €-145,374 | €154,626 | €609,361 | €-159,361 | no | fails the regret cap |
| New Capability | €-991,538 | €-1,408,501 | €-1,108,501 | €1,496,228 | €-1,046,228 | no | fails both guardrails |