import json
import time
import numpy as np
import pandas as pd

from common import (
    build_model, encode_features, clean_dataframe, make_sequences,
    class_weights_for, L_MAX,
)
from nslkdd_columns import NSLKDD_COLUMNS

RESULTS = {}


def run_dataset(name, df, label_col, normal_values, drop_cols, categorical_cols,
                 test_df=None, epochs=6, batch_size=256, onehot_max_cardinality=12):
    print(f"\n=== {name} ===")
    t0 = time.time()

    has_predefined_test = test_df is not None
    if has_predefined_test:
        n_train_raw = len(df)
        combined = pd.concat([df, test_df], axis=0, ignore_index=True)
    else:
        combined = df

    y_raw = combined[label_col]
    if not pd.api.types.is_numeric_dtype(y_raw):
        y_bin = (~y_raw.astype(str).str.strip().str.lower().isin(
            [str(v).lower() for v in normal_values])).astype(np.float32)
    else:
        y_bin = (~y_raw.isin(normal_values)).astype(np.float32)

    exclude = set(drop_cols) | {label_col}
    exclude = {c for c in exclude if c in combined.columns}
    df_clean = clean_dataframe(combined, categorical_cols, exclude)
    enc_df, feature_cols, scaler = encode_features(df_clean, categorical_cols, exclude,
                                                     onehot_max_cardinality=onehot_max_cardinality)

    if has_predefined_test:
        enc_train = enc_df.iloc[:n_train_raw].reset_index(drop=True)
        enc_test = enc_df.iloc[n_train_raw:].reset_index(drop=True)
        y_train_full = y_bin.values[:n_train_raw]
        y_test_full = y_bin.values[n_train_raw:]

        X_all, y_all = make_sequences(enc_train, feature_cols, y_train_full, L_MAX)
        idx = np.random.RandomState(42).permutation(len(X_all))
        X_all, y_all = X_all[idx], y_all[idx]
        split = int(len(X_all) * 0.85)
        X_train, y_train = X_all[:split], y_all[:split]
        X_val, y_val = X_all[split:], y_all[split:]

        X_test, y_test = make_sequences(enc_test, feature_cols, y_test_full, L_MAX)
    else:
        X, y = make_sequences(enc_df, feature_cols, y_bin.values, L_MAX)
        idx = np.random.RandomState(42).permutation(len(X))
        X, y = X[idx], y[idx]
        n = len(X)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]

    print(f"F={len(feature_cols)}  train={len(X_train)}  val={len(X_val)}  test={len(X_test)}  "
          f"pos_rate_train={y_train.mean():.3f}  pos_rate_test={y_test.mean():.3f}")

    model = build_model(L_max=L_MAX, F=len(feature_cols))
    cw = class_weights_for(y_train)
    model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=epochs, batch_size=batch_size, class_weight=cw, verbose=2,
    )

    test_metrics = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0, return_dict=True)
    elapsed = time.time() - t0
    print(f"{name} test metrics:", test_metrics, f"({elapsed:.1f}s)")

    model.save(f"../models/{name}.keras")

    RESULTS[name] = {
        "num_features": len(feature_cols),
        "train_windows": int(len(X_train)),
        "val_windows": int(len(X_val)),
        "test_windows": int(len(X_test)),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_precision": float(test_metrics["precision"]),
        "test_recall": float(test_metrics["recall"]),
        "test_f1": float(test_metrics["f1_score"]),
        "train_seconds": round(elapsed, 1),
    }


import os
os.makedirs("../models", exist_ok=True)

print("Loading NSL-KDD...")
nsl_train = pd.read_csv("../data/nslkdd_train.csv", names=NSLKDD_COLUMNS, header=None)
nsl_test = pd.read_csv("../data/nslkdd_test.csv", names=NSLKDD_COLUMNS, header=None)
run_dataset(
    "NSL-KDD", nsl_train, label_col="label", normal_values=["normal"],
    drop_cols=["difficulty"], categorical_cols=["protocol_type", "service", "flag"],
    test_df=nsl_test,
)

print("Loading UNSW-NB15...")
unsw = pd.read_csv("../data/unsw.csv")
unsw.columns = [c.strip().lstrip("\ufeff") for c in unsw.columns]
run_dataset(
    "UNSW-NB15", unsw, label_col="label", normal_values=[0],
    drop_cols=["id", "attack_cat"], categorical_cols=["proto", "service", "state"],
)

print("Loading Edge-IIoTset...")
edge = pd.read_csv("../data/edge_iiotset.csv", low_memory=False)
edge_drop = [
    "Attack_type", "frame.time", "ip.src_host", "ip.dst_host",
    "arp.dst.proto_ipv4", "arp.src.proto_ipv4", "http.file_data",
    "http.request.full_uri", "icmp.transmit_timestamp",
    "http.request.uri.query", "tcp.options", "tcp.payload",
    "tcp.srcport", "tcp.dstport", "udp.port", "mqtt.msg",
    "dns.qry.name", "mqtt.topic",
]
edge_cat = ["http.request.method", "http.referer", "http.request.version",
            "mqtt.conack.flags", "mqtt.protoname"]
run_dataset(
    "Edge-IIoTset", edge, label_col="Attack_label", normal_values=[0],
    drop_cols=edge_drop, categorical_cols=edge_cat,
)

with open("../models/benchmark_results.json", "w") as f:
    json.dump(RESULTS, f, indent=2)

print("\n\nFINAL RESULTS")
print(json.dumps(RESULTS, indent=2))
