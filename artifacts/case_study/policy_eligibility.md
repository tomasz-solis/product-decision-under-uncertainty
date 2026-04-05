- This is the policy-defining table for the current run.
- An option must pass both the downside floor and the regret cap to stay eligible.

| Option | Expected Value | P05 | Downside Slack | Mean Regret | Regret Slack | Eligible | Failure Reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Feature Extension | €-60,019 | €-547,304 | €-247,304 | €159,584 | €290,416 | no | fails the downside floor |
| Do Nothing | €-104,671 | €-145,374 | €154,626 | €204,236 | €245,764 | yes | passes both guardrails |
| Stabilize Core | €-207,577 | €-693,811 | €-393,811 | €307,142 | €142,858 | no | fails the downside floor |
| New Capability | €-956,398 | €-1,382,691 | €-1,082,691 | €1,055,963 | €-605,963 | no | fails both guardrails |