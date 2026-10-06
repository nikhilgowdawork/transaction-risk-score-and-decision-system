import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional


class DatabaseManager:
    """
    Handles persistent storage for transaction history and simulation state.

    SQLite is used because this application currently runs as a single backend
    service and does not need a separate database server.
    """

    def __init__(self, database_path: Path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize_database()

    def _connect(self):
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize_database(self):
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    transaction_id TEXT UNIQUE NOT NULL,
                    source TEXT NOT NULL,
                    step INTEGER,
                    amount REAL,
                    risk_score REAL NOT NULL,
                    decision TEXT NOT NULL,
                    actual_label INTEGER,
                    predicted_label INTEGER,
                    transaction_data TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS simulation_state (
                    state_key TEXT PRIMARY KEY,
                    state_value TEXT NOT NULL,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            connection.commit()

    def save_transaction(self, record: Dict[str, Any], source: str = "static"):
        input_raw = record.get("input_raw", {})

        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO transactions (
                    transaction_id,
                    source,
                    step,
                    amount,
                    risk_score,
                    decision,
                    actual_label,
                    predicted_label,
                    transaction_data
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["transaction_id"],
                    source,
                    int(record.get("step", input_raw.get("step", 1))),
                    float(record.get("amount", input_raw.get("amount", 0.0))),
                    float(record.get("risk_score", 0.0)),
                    record.get("decision", "UNKNOWN"),
                    record.get("ground_truth_label", record.get("actual_label")),
                    record.get("predicted_label"),
                    json.dumps(record, default=str),
                ),
            )
            connection.commit()

    def get_transaction_count(self) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM transactions"
            ).fetchone()
            return int(row["count"])

    def get_average_risk_score(self) -> float:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT AVG(risk_score) AS average FROM transactions"
            ).fetchone()
            return round(float(row["average"] or 0.0), 4)

    def get_decision_counts(self) -> Dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT decision, COUNT(*) AS count
                FROM transactions
                GROUP BY decision
                """
            ).fetchall()

        counts = {
            "approved": 0,
            "flagged_for_review": 0,
            "blocked": 0,
        }

        for row in rows:
            if row["decision"] == "APPROVE":
                counts["approved"] = int(row["count"])
            elif row["decision"] == "FLAG FOR REVIEW":
                counts["flagged_for_review"] = int(row["count"])
            elif row["decision"] == "BLOCK TRANSACTION":
                counts["blocked"] = int(row["count"])

        return counts

    def get_recent_transactions(self, limit: int = 10) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, transaction_data
                FROM transactions
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        records = []
        for row in reversed(rows):
            try:
                record = json.loads(row["transaction_data"])
                record["transaction_index"] = int(row["id"])
                records.append(record)
            except json.JSONDecodeError:
                pass

        return records

    def get_last_stream_index(self) -> int:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT state_value
                FROM simulation_state
                WHERE state_key = 'last_stream_index'
                """
            ).fetchone()
            return int(row["state_value"]) if row else 0

    def save_last_stream_index(self, index: int) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO simulation_state (state_key, state_value, updated_at)
                VALUES ('last_stream_index', ?, CURRENT_TIMESTAMP)
                ON CONFLICT(state_key) DO UPDATE SET
                    state_value = excluded.state_value,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (str(index),),
            )

    def clear_transactions(self) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM transactions")
            connection.execute("DELETE FROM simulation_state")
            connection.execute(
                "DELETE FROM sqlite_sequence WHERE name = 'transactions'"
            )

    def get_risk_score_graph(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        query = """
            SELECT
                id,
                transaction_id,
                risk_score,
                decision
            FROM transactions
            ORDER BY id ASC
        """

        params = ()

        if limit is not None:
            query = """
                SELECT
                    id,
                    transaction_id,
                    risk_score,
                    decision
                FROM transactions
                ORDER BY id DESC
                LIMIT ?
            """
            params = (limit,)

        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()

        if limit is not None:
            rows = reversed(rows)

        return [
            {
                "index": int(row["id"]),
                "transaction_id": row["transaction_id"],
                "risk_score": round(float(row["risk_score"]), 4),
                "decision": row["decision"],
            }
            for row in rows
        ]

    
    
