# Feature contract

| Feature family | T1 allowed? | Source | Rule |
|---|---:|---|---|
| Priority / Category / Channel | Yes | Ticket intake | Known at creation |
| Contract tier / credit clause | Yes | Client contract | Must be active/known at creation |
| Agent capacity | Yes | Agent assignment | Only because T1 is after assignment |
| SLA window | Conditional | SLA scheduler | Only if SLADueAt is assigned by T1 |
| Queue pressure | Yes | Event history | Reconstruct strictly before prediction timestamp |
| Historical burden | Yes | Older completed tickets | Completion must pre-date the current prediction |
| Historical target encoding | Yes | Older outcomes | Ordered / causal only; never full-data means |
| FirstResponseAt | No at T1 | Future lifecycle | Legal from T2 onward |
| ResolutionNotes / ResolvedAt | No at T1 | Future lifecycle | Historical teacher only |
| Final Status | No at T1 | Future lifecycle | Requires timestamped status history for dynamic models |
