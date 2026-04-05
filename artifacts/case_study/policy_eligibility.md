- This is the policy-defining table for the current run.
- An option must pass both the downside floor and the regret cap to stay eligible.

| Option | Expected Value | P05 | Downside Slack | Mean Regret | Regret Slack | Eligible | Failure Reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Feature Extension | €-64,899 | €-544,675 | €-244,675 | €157,845 | €292,155 | no | fails the downside floor |
| Do Nothing | €-104,671 | €-145,374 | €154,626 | €197,616 | €252,384 | yes | passes both guardrails |
| Stabilize Core | €-218,673 | €-691,700 | €-391,700 | €311,619 | €138,381 | no | fails the downside floor |
| New Capability | €-954,763 | €-1,376,447 | €-1,076,447 | €1,047,708 | €-597,708 | no | fails both guardrails |