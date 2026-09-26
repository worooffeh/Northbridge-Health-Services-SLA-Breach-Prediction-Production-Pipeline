# Current smoke-test result

This repository was executed against the supplied Northbridge dataset before packaging.

- Champion selected on validation: **E1_baseline / Logistic Regression**
- Validation ROC-AUC: **0.5596**
- Validation PR-AUC: **0.2551**
- Test ROC-AUC: **0.5080**
- Test PR-AUC: **0.2602**
- Test Lift@10%: **1.1792**

The weak held-out test discrimination is intentionally retained. It demonstrates the core project lesson: honest leakage-aware validation can reveal that the available T1 signal is not yet strong enough for confident production use. The repository is therefore designed to support better signal discovery, dynamic T2/T3 models, monitoring and challenger retraining rather than hiding the limitation.
