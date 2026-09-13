import asyncio
import csv
import io
import json
import os
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import database as db
from .chain_client import ChainClient, event_to_feature_vector
from .model_service import model_service, ANOMALY_THRESHOLD
from .onchain_model import FEATURE_ORDER

app = FastAPI(title="CNN-BiLSTM+Attention for Abnormal Behaviors Detection")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

chain_client = None
manager_connections = []
recent_latencies = []


class FeedbackIn(BaseModel):
    alert_id: int
    verdict: str
    note: str | None = None


async def broadcast(message: dict):
    dead = []
    for ws in manager_connections:
        try:
            await ws.send_text(json.dumps(message))
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in manager_connections:
            manager_connections.remove(ws)


def severity_for_score(score):
    if score >= 0.85:
        return "critical"
    if score >= ANOMALY_THRESHOLD:
        return "high"
    if score >= 0.4:
        return "watch"
    return "normal"


def top_feature_names(gate_values, feat_vector, k=3):
    scored = sorted(zip(FEATURE_ORDER, gate_values, feat_vector), key=lambda t: t[1] * (1 + t[2]), reverse=True)
    return [name for name, _, _ in scored[:k]]


async def chain_actor_loop():
    while True:
        anomaly_rate = 0.1
        await asyncio.to_thread(chain_client.submit_random_event, anomaly_rate)
        await asyncio.sleep(1.2)


async def chain_listener_loop():
    while True:
        events = await asyncio.to_thread(chain_client.poll_new_events)
        for ev in events:
            vec = event_to_feature_vector(ev)
            t0 = time.perf_counter()
            score, attention_weights, gate_values = model_service.score_event(ev["entity"], vec)
            latency_ms = (time.perf_counter() - t0) * 1000
            recent_latencies.append(latency_ms)
            if len(recent_latencies) > 500:
                recent_latencies.pop(0)

            is_anomaly = score >= ANOMALY_THRESHOLD
            record = {
                "tx_hash": ev["tx_hash"],
                "block_number": ev["block_number"],
                "entity": ev["entity"],
                "role": ev["role"],
                "action": ev["action"],
                "sensitive_record_accessed": ev["sensitive_record_accessed"],
                "role_mismatch": ev["role_mismatch"],
                "credential_revocation": ev["credential_revocation"],
                "geolocation_mismatch": ev["geolocation_mismatch"],
                "issuer_verifier_collusion": ev["issuer_verifier_collusion"],
                "timestamp": ev["timestamp"],
                "score": score,
                "is_anomaly": is_anomaly,
            }
            event_id = db.insert_event(record, attention_weights, gate_values)
            record["id"] = event_id

            alert = None
            if is_anomaly:
                severity = severity_for_score(score)
                top_feats = top_feature_names(gate_values, vec)
                alert_id = db.insert_alert(event_id, ev["tx_hash"], ev["timestamp"],
                                            ev["entity"], score, severity, top_feats)
                alert = {
                    "id": alert_id, "event_id": event_id, "tx_hash": ev["tx_hash"],
                    "timestamp": ev["timestamp"], "entity": ev["entity"],
                    "score": score, "severity": severity, "status": "open",
                    "top_features": top_feats,
                }

            await broadcast({"type": "event", "data": record, "alert": alert})
        await asyncio.sleep(1.0)


@app.on_event("startup")
async def startup():
    global chain_client
    db.init_db()
    print("Training on-chain detector on synthetic historical schema...")
    await asyncio.to_thread(model_service.train_on_synthetic_history)
    print("Connecting to local chain...")
    chain_client = ChainClient()
    if not chain_client.is_connected():
        raise RuntimeError("Could not connect to local Ethereum node (ganache). Is it running on :8545?")
    asyncio.create_task(chain_actor_loop())
    asyncio.create_task(chain_listener_loop())


@app.websocket("/ws/feed")
async def ws_feed(websocket: WebSocket):
    await websocket.accept()
    manager_connections.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in manager_connections:
            manager_connections.remove(websocket)


@app.get("/api/events")
def get_events(limit: int = 150, offset: int = 0, only_anomalies: bool = False):
    return db.list_events(limit=limit, offset=offset, only_anomalies=only_anomalies)


@app.get("/api/events/{event_id}")
def get_event_detail(event_id: int):
    row = db.get_event(event_id)
    if not row:
        raise HTTPException(status_code=404, detail="Event not found")
    row["attention_weights"] = [float(x) for x in row["attention_weights"].split(",") if x]
    gate_vals = [float(x) for x in row["feature_gate"].split(",") if x]
    row["feature_gate"] = [{"feature": name, "value": val} for name, val in zip(FEATURE_ORDER, gate_vals)]
    alert = db.get_alert_by_event(event_id)
    row["alert_id"] = alert["id"] if alert else None
    row["alert_status"] = alert["status"] if alert else None
    return row


@app.get("/api/alerts")
def get_alerts(limit: int = 100, status: str = None):
    rows = db.list_alerts(limit=limit, status=status)
    for r in rows:
        r["top_features"] = r["top_features"].split(",") if r["top_features"] else []
    return rows


@app.post("/api/feedback")
def post_feedback(feedback: FeedbackIn):
    db.insert_feedback(feedback.alert_id, feedback.verdict, feedback.note)
    return {"ok": True}


@app.get("/api/chain_status")
def get_chain_status():
    if chain_client is None:
        return {"connected": False}
    return {
        "connected": chain_client.is_connected(),
        "chain_id": chain_client.chain_id,
        "contract_address": chain_client.address,
        "latest_block": chain_client.latest_block_number(),
        "accounts": len(chain_client.accounts),
    }


@app.get("/api/stats")
def get_stats():
    total, total_alerts, open_alerts, anomalies = db.counts()
    anomaly_rate = (anomalies / total * 100.0) if total else 0.0
    avg_latency = sum(recent_latencies) / len(recent_latencies) if recent_latencies else 0.0
    chain_status = get_chain_status()
    return {
        "total_events": total,
        "total_alerts": total_alerts,
        "open_alerts": open_alerts,
        "anomaly_rate": round(anomaly_rate, 2),
        "avg_latency_ms": round(avg_latency, 2),
        "chain": chain_status,
    }


@app.get("/api/audit/export")
def export_audit():
    rows = db.all_events_for_export()
    buffer = io.StringIO()
    if rows:
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    buffer.seek(0)
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=onchain_audit_trail.csv"},
    )


FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
