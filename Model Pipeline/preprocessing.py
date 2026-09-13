import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, LabelEncoder

DATASET_PRESETS = {
    "UNSW-NB15": {
        "label_column": "label",
        "normal_values": [0],
        "drop_columns": ["id", "attack_cat"],
        "categorical_columns": ["proto", "service", "state"],
        "entity_column": "srcip",
        "timestamp_column": "Stime",
    },
    "Edge-IIoTset": {
        "label_column": "Attack_label",
        "normal_values": [0],
        "drop_columns": [
            "Attack_type", "frame.time", "ip.src_host", "ip.dst_host",
            "arp.dst.proto_ipv4", "arp.src.proto_ipv4", "http.file_data",
            "http.request.full_uri", "icmp.transmit_timestamp",
            "http.request.uri.query", "tcp.options", "tcp.payload",
            "tcp.srcport", "tcp.dstport", "udp.port", "mqtt.msg",
        ],
        "categorical_columns": [
            "http.request.method", "http.referer", "http.request.version",
            "dns.qry.name.len", "mqtt.conack.flags", "mqtt.protoname", "mqtt.topic",
        ],
        "entity_column": None,
        "timestamp_column": None,
    },
    "NSL-KDD": {
        "label_column": "label",
        "normal_values": ["normal"],
        "drop_columns": ["difficulty"],
        "categorical_columns": ["protocol_type", "service", "flag"],
        "entity_column": None,
        "timestamp_column": None,
    },
    "TON_IoT": {
        "label_column": "label",
        "normal_values": [0],
        "drop_columns": ["type"],
        "categorical_columns": ["proto", "service", "conn_state"],
        "entity_column": "src_ip",
        "timestamp_column": "ts",
    },
}


def resolve_config(dataset_name=None, label_column=None, normal_values=None,
                    drop_columns=None, categorical_columns=None,
                    entity_column=None, timestamp_column=None):
    base = {
        "label_column": "label",
        "normal_values": [0],
        "drop_columns": [],
        "categorical_columns": [],
        "entity_column": None,
        "timestamp_column": None,
    }
    if dataset_name is not None and dataset_name in DATASET_PRESETS:
        base.update(DATASET_PRESETS[dataset_name])
    if label_column is not None:
        base["label_column"] = label_column
    if normal_values is not None:
        base["normal_values"] = normal_values
    if drop_columns is not None:
        base["drop_columns"] = drop_columns
    if categorical_columns is not None:
        base["categorical_columns"] = categorical_columns
    if entity_column is not None:
        base["entity_column"] = entity_column
    if timestamp_column is not None:
        base["timestamp_column"] = timestamp_column
    return base


def load_raw_dataset(path, column_names=None, **read_csv_kwargs):
    if column_names is not None:
        df = pd.read_csv(path, names=column_names, header=None, **read_csv_kwargs)
    else:
        df = pd.read_csv(path, **read_csv_kwargs)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def binarize_label(df, label_column, normal_values):
    y = df[label_column]
    if not pd.api.types.is_numeric_dtype(y):
        y = y.astype(str).str.strip().str.lower()
        normal_values = [str(v).strip().lower() for v in normal_values]
        binary = (~y.isin(normal_values)).astype(np.float32)
    else:
        binary = (~y.isin(normal_values)).astype(np.float32)
    return binary


def clean_dataframe(df, categorical_columns, exclude_columns):
    df = df.drop_duplicates()
    for col in df.columns:
        if col in exclude_columns:
            continue
        if col in categorical_columns or not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna("Unknown")
            mode_vals = df[col].mode()
            if len(mode_vals) > 0:
                df[col] = df[col].replace("Unknown", mode_vals.iloc[0])
        else:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    return df


def encode_features(df, categorical_columns, exclude_columns, onehot_max_cardinality=10):
    df = df.copy()
    feature_columns = [c for c in df.columns if c not in exclude_columns]
    numeric_columns = [c for c in feature_columns if c not in categorical_columns and pd.api.types.is_numeric_dtype(df[c])]
    cat_columns = [c for c in feature_columns if c in categorical_columns or not pd.api.types.is_numeric_dtype(df[c])]

    low_card = [c for c in cat_columns if df[c].nunique() <= onehot_max_cardinality]
    high_card = [c for c in cat_columns if c not in low_card]

    for c in high_card:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))
        numeric_columns.append(c)

    if low_card:
        df = pd.get_dummies(df, columns=low_card, dtype=np.float32)
        onehot_columns = [c for c in df.columns if any(c.startswith(f"{orig}_") for orig in low_card)]
    else:
        onehot_columns = []

    scaler = RobustScaler()
    if numeric_columns:
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns].astype(np.float64))

    final_feature_columns = numeric_columns + onehot_columns
    df[final_feature_columns] = df[final_feature_columns].astype(np.float32)
    return df, final_feature_columns, scaler


def create_sequences(df, feature_columns, label_series, L_max=20, entity_column=None, timestamp_column=None):
    work = df.copy()
    work["_label_"] = label_series.values

    if timestamp_column is not None and timestamp_column in work.columns:
        work = work.sort_values(timestamp_column)

    if entity_column is not None and entity_column in work.columns:
        groups = [g for _, g in work.groupby(entity_column)]
    else:
        groups = [work]

    F = len(feature_columns)
    X_list = []
    y_list = []

    for g in groups:
        values = g[feature_columns].values.astype(np.float32)
        labels = g["_label_"].values.astype(np.float32)
        n = len(values)
        if n == 0:
            continue
        for start in range(0, n, L_max):
            end = min(start + L_max, n)
            window = values[start:end]
            window_labels = labels[start:end]
            pad_len = L_max - window.shape[0]
            if pad_len > 0:
                pad = np.zeros((pad_len, F), dtype=np.float32)
                window = np.vstack([pad, window])
            X_list.append(window)
            y_list.append(1.0 if window_labels.max() > 0 else 0.0)

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y


def split_sequences(X, y, ratios=(0.7, 0.15, 0.15)):
    n = len(X)
    train_end = int(n * ratios[0])
    val_end = train_end + int(n * ratios[1])
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    return X_train, y_train, X_val, y_val, X_test, y_test


def preprocess_dataset(path, dataset_name=None, column_names=None, label_column=None,
                        normal_values=None, drop_columns=None, categorical_columns=None,
                        entity_column=None, timestamp_column=None, L_max=20,
                        onehot_max_cardinality=10, split_ratios=(0.7, 0.15, 0.15),
                        read_csv_kwargs=None):
    read_csv_kwargs = read_csv_kwargs or {}
    config = resolve_config(dataset_name, label_column, normal_values, drop_columns,
                             categorical_columns, entity_column, timestamp_column)

    df = load_raw_dataset(path, column_names=column_names, **read_csv_kwargs)

    label_col = config["label_column"]
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataset columns.")

    exclude_columns = set(config["drop_columns"]) | {label_col}
    for col in list(exclude_columns):
        if col not in df.columns:
            exclude_columns.discard(col)

    entity_col = config["entity_column"] if config["entity_column"] in df.columns else None
    timestamp_col = config["timestamp_column"] if config["timestamp_column"] in df.columns else None

    cat_cols = [c for c in config["categorical_columns"] if c in df.columns]

    y_full = binarize_label(df, label_col, config["normal_values"])

    df = clean_dataframe(df, cat_cols, exclude_columns)

    encode_exclude = set(exclude_columns)
    if entity_col:
        encode_exclude.add(entity_col)
    if timestamp_col:
        encode_exclude.add(timestamp_col)

    encoded_df, feature_columns, scaler = encode_features(df, cat_cols, encode_exclude, onehot_max_cardinality)

    if entity_col:
        encoded_df[entity_col] = df[entity_col].values
    if timestamp_col:
        encoded_df[timestamp_col] = df[timestamp_col].values

    X, y = create_sequences(encoded_df, feature_columns, y_full, L_max=L_max,
                             entity_column=entity_col, timestamp_column=timestamp_col)

    X_train, y_train, X_val, y_val, X_test, y_test = split_sequences(X, y, split_ratios)

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "feature_columns": feature_columns,
        "scaler": scaler,
        "L_max": L_max,
        "F": len(feature_columns),
    }
