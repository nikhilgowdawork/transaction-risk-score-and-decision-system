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

    STATIC_SOURCES = ("static", "analyze", "simulate-single")

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

    def get_transaction_count(self, source: Optional[str] = None) -> int:
        with self._connect() as connection:
            if source is None:
                row = connection.execute(
                    """
                    SELECT COUNT(*) AS count
                    FROM transactions
                    WHERE source IN ('static', 'analyze', 'simulate-single')
                    """
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT COUNT(*) AS count FROM transactions WHERE source = ?",
                    (source,),
                ).fetchone()
            return int(row["count"])

    def get_average_risk_score(self) -> float:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT AVG(risk_score) AS average
                FROM transactions
                WHERE source IN ('static', 'analyze', 'simulate-single')
                """
            ).fetchone()
            return round(float(row["average"] or 0.0), 4)

    def get_decision_counts(self) -> Dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT decision, COUNT(*) AS count
                FROM transactions
                WHERE source IN ('static', 'analyze', 'simulate-single')
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
                SELECT transaction_index, transaction_data
                FROM (
                    SELECT
                        ROW_NUMBER() OVER (ORDER BY id) AS transaction_index,
                        transaction_data,
                        id
                    FROM transactions
                    WHERE source IN ('static', 'analyze', 'simulate-single')
                )
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        records = []
        for row in reversed(rows):
            try:
                record = json.loads(row["transaction_data"])
                record["transaction_index"] = int(row["transaction_index"])
                records.append(record)
            except json.JSONDecodeError:
                pass

        return records

    def get_transaction_by_id(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT transaction_data
                FROM transactions
                WHERE transaction_id = ?
                  AND source IN ('static', 'analyze', 'simulate-single')
                """,
                (transaction_id,),
            ).fetchone()

        if row is None:
            return None
        return json.loads(row["transaction_data"])

    def update_transaction_analysis(
        self,
        transaction_id: str,
        shap_contributions: List[Dict[str, Any]],
        llm_audit: str,
    ) -> None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT transaction_data
                FROM transactions
                WHERE transaction_id = ?
                  AND source IN ('static', 'analyze', 'simulate-single')
                """,
                (transaction_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"Static transaction not found: {transaction_id}")

            record = json.loads(row["transaction_data"])
            record["shap_contributions"] = shap_contributions
            record["llm_audit"] = llm_audit
            connection.execute(
                """
                UPDATE transactions
                SET transaction_data = ?
                WHERE transaction_id = ?
                  AND source IN ('static', 'analyze', 'simulate-single')
                """,
                (json.dumps(record, default=str), transaction_id),
            )

    def clear_transactions(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                DELETE FROM transactions
                WHERE source IN ('static', 'analyze', 'simulate-single')
                """
            )

    def get_risk_score_graph(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        query = """
            SELECT
            ROW_NUMBER() OVER (ORDER BY id) AS transaction_index,
                transaction_id,
                risk_score,
                decision
            FROM transactions
            WHERE source IN ('static', 'analyze', 'simulate-single')
            ORDER BY id ASC
        """

        params = ()

        if limit is not None:
            query = """
                SELECT
                    ROW_NUMBER() OVER (ORDER BY id) AS transaction_index,
                    transaction_id,
                    risk_score,
                    decision
                FROM transactions
                WHERE source IN ('static', 'analyze', 'simulate-single')
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
                "index": int(row["transaction_index"]),
                "transaction_id": row["transaction_id"],
                "risk_score": round(float(row["risk_score"]), 4),
                "decision": row["decision"],
            }
            for row in rows
        ]

    
    
