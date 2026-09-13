import sqlite3
import threading
import time
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "healthcare_chain.db")
_lock = threading.Lock()


def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _lock:
        conn = get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tx_hash TEXT NOT NULL,
                block_number INTEGER NOT NULL,
                entity TEXT NOT NULL,
                role TEXT NOT NULL,
                action TEXT NOT NULL,
                sensitive_record_accessed INTEGER NOT NULL,
                role_mismatch INTEGER NOT NULL,
                credential_revocation INTEGER NOT NULL,
                geolocation_mismatch INTEGER NOT NULL,
                issuer_verifier_collusion INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                score REAL NOT NULL,
                is_anomaly INTEGER NOT NULL,
                attention_weights TEXT,
                feature_gate TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id INTEGER NOT NULL,
                tx_hash TEXT NOT NULL,
                timestamp REAL NOT NULL,
                entity TEXT NOT NULL,
                score REAL NOT NULL,
                severity TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'open',
                top_features TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_id INTEGER NOT NULL,
                verdict TEXT NOT NULL,
                note TEXT,
                created_at REAL NOT NULL
            )
        """)
        conn.commit()
        conn.close()


def insert_event(ev, attention_weights, feature_gate):
    with _lock:
        conn = get_conn()
        cur = conn.execute(
            """INSERT INTO events
               (tx_hash, block_number, entity, role, action, sensitive_record_accessed,
                role_mismatch, credential_revocation, geolocation_mismatch,
                issuer_verifier_collusion, timestamp, score, is_anomaly,
                attention_weights, feature_gate)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                ev["tx_hash"], ev["block_number"], ev["entity"], ev["role"], ev["action"],
                int(ev["sensitive_record_accessed"]), int(ev["role_mismatch"]),
                int(ev["credential_revocation"]), int(ev["geolocation_mismatch"]),
                int(ev["issuer_verifier_collusion"]), ev["timestamp"], ev["score"],
                int(ev["is_anomaly"]),
                ",".join(f"{w:.4f}" for w in attention_weights),
                ",".join(f"{g:.4f}" for g in feature_gate),
            ),
        )
        conn.commit()
        event_id = cur.lastrowid
        conn.close()
        return event_id


def insert_alert(event_id, tx_hash, timestamp, entity, score, severity, top_features):
    with _lock:
        conn = get_conn()
        cur = conn.execute(
            """INSERT INTO alerts (event_id, tx_hash, timestamp, entity, score, severity, status, top_features)
               VALUES (?,?,?,?,?,?, 'open', ?)""",
            (event_id, tx_hash, timestamp, entity, score, severity, ",".join(top_features)),
        )
        conn.commit()
        alert_id = cur.lastrowid
        conn.close()
        return alert_id


def get_alert_by_event(event_id):
    with _lock:
        conn = get_conn()
        row = conn.execute("SELECT * FROM alerts WHERE event_id = ? ORDER BY id DESC LIMIT 1", (event_id,)).fetchone()
        conn.close()
        return dict(row) if row else None


def list_events(limit=150, offset=0, only_anomalies=False):
    with _lock:
        conn = get_conn()
        q = "SELECT * FROM events"
        if only_anomalies:
            q += " WHERE is_anomaly = 1"
        q += " ORDER BY id DESC LIMIT ? OFFSET ?"
        rows = conn.execute(q, (limit, offset)).fetchall()
        conn.close()
        return [dict(r) for r in rows]


def get_event(event_id):
    with _lock:
        conn = get_conn()
        row = conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
        conn.close()
        return dict(row) if row else None


def list_alerts(limit=100, status=None):
    with _lock:
        conn = get_conn()
        q = "SELECT * FROM alerts"
        params = []
        if status:
            q += " WHERE status = ?"
            params.append(status)
        q += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(q, params).fetchall()
        conn.close()
        return [dict(r) for r in rows]


def insert_feedback(alert_id, verdict, note):
    with _lock:
        conn = get_conn()
        conn.execute(
            "INSERT INTO feedback (alert_id, verdict, note, created_at) VALUES (?,?,?,?)",
            (alert_id, verdict, note, time.time()),
        )
        new_status = "confirmed" if verdict == "true_positive" else "dismissed"
        conn.execute("UPDATE alerts SET status = ? WHERE id = ?", (new_status, alert_id))
        conn.commit()
        conn.close()


def counts():
    with _lock:
        conn = get_conn()
        total = conn.execute("SELECT COUNT(*) c FROM events").fetchone()["c"]
        total_alerts = conn.execute("SELECT COUNT(*) c FROM alerts").fetchone()["c"]
        open_alerts = conn.execute("SELECT COUNT(*) c FROM alerts WHERE status = 'open'").fetchone()["c"]
        anomalies = conn.execute("SELECT COUNT(*) c FROM events WHERE is_anomaly = 1").fetchone()["c"]
        conn.close()
        return total, total_alerts, open_alerts, anomalies


def all_events_for_export():
    with _lock:
        conn = get_conn()
        rows = conn.execute("SELECT * FROM events ORDER BY id ASC").fetchall()
        conn.close()
        return [dict(r) for r in rows]
