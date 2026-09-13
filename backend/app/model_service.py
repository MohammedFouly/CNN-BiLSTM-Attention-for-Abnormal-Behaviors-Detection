import random
import numpy as np
from collections import defaultdict, deque

from .onchain_model import build_model, FEATURE_ORDER, NUM_FEATURES, L_MAX

ANOMALY_THRESHOLD = 0.6

ROLES = ["physician", "nurse", "billing_clerk", "lab_tech", "insurance_auditor", "admin"]
ACTIONS = ["record_access", "prescription_write", "insurance_claim", "consent_update", "credential_check", "lab_result_upload"]
HIGH_RISK_ROLES = {"billing_clerk", "admin"}


def synthetic_event(rng, anomaly_rate):
    role = rng.choice(ROLES)
    is_anom = rng.random() < anomaly_rate

    gas_dev = abs(rng.gauss(0, 1))
    sensitive = rng.random() < 0.25
    role_mismatch = False
    cred_revocation = False
    geoloc = False
    collusion = False
    access_freq = rng.uniform(0, 5)
    session_dur = rng.uniform(30, 900)
    time_since_last = rng.uniform(0, 7200)
    input_size = rng.uniform(200, 4000)

    if is_anom:
        gas_dev *= rng.uniform(4, 9)
        sensitive = True
        role_mismatch = role not in HIGH_RISK_ROLES and rng.random() < 0.7
        cred_revocation = rng.random() < 0.5
        access_freq *= rng.uniform(4, 10)
        input_size *= rng.uniform(3, 6)
        time_since_last = rng.uniform(0, 5)
        geoloc = rng.random() < 0.3
        collusion = rng.random() < 0.15

    hour = rng.randint(0, 23)
    features = {
        "hour_of_day": hour / 23.0,
        "time_since_last_access": min(time_since_last / 7200.0, 1.0),
        "access_frequency": min(access_freq / 20.0, 1.0),
        "avg_session_duration": min(session_dur / 900.0, 1.0),
        "gas_price_deviation": min(gas_dev / 10.0, 1.0),
        "input_data_size": min(input_size / 24000.0, 1.0),
        "sensitive_record_accessed": 1.0 if sensitive else 0.0,
        "role_mismatch": 1.0 if role_mismatch else 0.0,
        "credential_revocation": 1.0 if cred_revocation else 0.0,
        "geolocation_mismatch": 1.0 if geoloc else 0.0,
        "issuer_verifier_collusion": 1.0 if collusion else 0.0,
    }
    vector = [features[f] for f in FEATURE_ORDER]
    return vector, 1.0 if is_anom else 0.0


def generate_training_sequences(n_records=9000, seed=42, anomaly_rate=0.12):
    rng = random.Random(seed)
    entities = [f"entity_{i}" for i in range(12)]
    per_entity = defaultdict(list)
    for _ in range(n_records):
        entity = rng.choice(entities)
        vec, label = synthetic_event(rng, anomaly_rate)
        per_entity[entity].append((vec, label))

    X, y = [], []
    for entity, seq in per_entity.items():
        n = len(seq)
        for start in range(0, n, L_MAX):
            chunk = seq[start:start + L_MAX]
            feats = np.array([c[0] for c in chunk], dtype=np.float32)
            labels = np.array([c[1] for c in chunk], dtype=np.float32)
            pad = L_MAX - feats.shape[0]
            if pad > 0:
                feats = np.vstack([np.zeros((pad, NUM_FEATURES), dtype=np.float32), feats])
            X.append(feats)
            y.append(float(labels[-1]))

    X = np.stack(X)
    y = np.array(y, dtype=np.float32)
    idx = np.random.RandomState(seed).permutation(len(X))
    return X[idx], y[idx]


class EntityWindowBuffer:
    def __init__(self, maxlen=L_MAX):
        self.maxlen = maxlen
        self.buffers = defaultdict(lambda: deque(maxlen=maxlen))

    def push(self, entity, vector):
        self.buffers[entity].append(vector)

    def window(self, entity):
        buf = list(self.buffers[entity])
        pad = self.maxlen - len(buf)
        if pad > 0:
            buf = [[0.0] * NUM_FEATURES] * pad + buf
        return np.array(buf, dtype=np.float32)


class ModelService:
    def __init__(self):
        self.inference_model, self.train_model, self.gate_layer = build_model()
        self.window_buffer = EntityWindowBuffer(L_MAX)
        self.trained = False
        self.metrics = {}

    def train_on_synthetic_history(self, n_records=9000, epochs=15, batch_size=64):
        X, y = generate_training_sequences(n_records=n_records)
        split = int(len(X) * 0.85)
        X_train, y_train = X[:split], y[:split]
        X_val, y_val = X[split:], y[split:]

        n_pos = float(y_train.sum())
        n_neg = float(len(y_train) - n_pos)
        total = n_pos + n_neg
        class_weight = {
            0: total / (2.0 * n_neg) if n_neg > 0 else 1.0,
            1: total / (2.0 * n_pos) if n_pos > 0 else 1.0,
        }

        self.train_model.fit(
            X_train, y_train, validation_data=(X_val, y_val),
            epochs=epochs, batch_size=batch_size, verbose=0,
            class_weight=class_weight,
        )
        val_metrics = self.train_model.evaluate(X_val, y_val, verbose=0, return_dict=True)
        self.metrics = val_metrics
        self.trained = True
        return val_metrics

    def score_event(self, entity, feat_vector):
        self.window_buffer.push(entity, feat_vector)
        window = self.window_buffer.window(entity)
        batch = np.expand_dims(window, axis=0)
        score, attention = self.inference_model.predict(batch, verbose=0)
        score = float(score[0][0])
        attention_weights = attention[0].tolist()
        gate_values = self.gate_layer.gate_values(
            np.expand_dims(window[-1], axis=0)
        ).numpy()[0].tolist()
        return score, attention_weights, gate_values


model_service = ModelService()
