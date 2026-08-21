# Rank-23 complexity follow-up

Low-support campaign winners are exactified independently and their linear-form
addition counts are compared before and after a deterministic greedy exact CSE pass.
CSE counts are heuristic straight-line-program upper bounds, not global minima; scalar
constant multiplications are not charged in the addition count.

| seed | winner exact | snapped zeros exact | naive adds baseline -> numeric -> exact | greedy-CSE adds old -> exact | field |
|---:|:---:|:---:|---:|---:|:---|
| 211 | yes | yes | 143 -> 127 -> 124 | 94 -> 82 | Q(alpha), alpha root of 17694947777*t + 23082777 |
| 401 | yes | yes | 137 -> 128 -> 109 | 87 -> 71 | Q(alpha), alpha root of 23*t**2 - 4 |

The exact-zero check is the key guardrail: every coefficient deliberately snapped by
the complexity search must remain zero in the symbolic certificate.

