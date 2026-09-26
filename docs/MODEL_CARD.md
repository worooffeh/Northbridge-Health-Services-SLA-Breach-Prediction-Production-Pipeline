# Model card — Northbridge SLA breach champion

The file `models/champion/metadata.json` is the machine-readable source of truth after training.

## Intended use
Prioritise service tickets for proactive review after assignment (T1).

## Not intended for
- automated denial of service;
- employee performance ranking without further causal/HR review;
- scoring before agent assignment if agent-specific fields are present;
- use after major process changes without drift/performance checks.

## Evaluation
Champion selection uses validation ROC-AUC, PR-AUC and Lift@10%. The final future test period is unlocked only after the champion is chosen.

## Known limitation
Low discrimination can reflect weak pre-event signal. The architecture deliberately prefers an honest near-random score over inflated leakage-driven performance.
