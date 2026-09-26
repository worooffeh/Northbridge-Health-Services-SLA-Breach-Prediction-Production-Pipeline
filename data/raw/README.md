# Raw data location

Place the three source files here before running the pipeline:

- `Tickets.csv`
- `Agents.csv`
- `Clients.csv`

They are intentionally ignored by Git to reduce the risk of publishing operational or personal data.

For this project workspace, run:

```bash
python scripts/copy_local_data.py --source /mnt/data
```

For a real production system, replace file copying with a governed database/API ingestion connector.
