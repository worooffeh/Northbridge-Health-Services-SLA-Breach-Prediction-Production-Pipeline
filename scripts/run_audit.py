from src.config import data_dir
from src.data.load import load_source_tables, build_master_table
from src.data.audit import audit_frame, write_audit


def main():
    tickets, agents, clients = load_source_tables(data_dir())
    master = build_master_table(tickets, agents, clients)
    audit = audit_frame(master)
    write_audit(audit, "reports/audit/data_audit.json")
    print(audit)


if __name__ == "__main__":
    main()
