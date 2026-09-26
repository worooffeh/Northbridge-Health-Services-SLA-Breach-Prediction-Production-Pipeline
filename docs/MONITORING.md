# Monitoring and retraining

Monitor five layers:

1. **Data quality** — schema, missingness, invalid category values.
2. **Input drift** — PSI by numeric/categorical feature.
3. **Prediction drift** — score distribution and alert volume.
4. **Performance drift** — ROC-AUC, PR-AUC, Lift@K, Brier once labels mature.
5. **Explanation drift** — top-driver distribution changes.

Retraining is gated, not automatic. A drift alert starts diagnosis. Retrain only when a fresh chronological evaluation beats the current champion and passes leakage verification.
