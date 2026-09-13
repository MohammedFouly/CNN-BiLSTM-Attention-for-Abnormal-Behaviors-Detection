import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, Model, Input

L_MAX = 20


class FeatureGate(layers.Layer):
    def __init__(self, num_features, prior=None, gate_lambda=0.01, **kwargs):
        super().__init__(**kwargs)
        self.num_features = num_features
        self.gate_lambda = gate_lambda
        if prior is None:
            prior = np.full((num_features,), 0.5, dtype=np.float32)
        self.prior = np.asarray(prior, dtype=np.float32)

    def build(self, input_shape):
        self.w = self.add_weight(name="gate_w", shape=(self.num_features,),
                                  initializer="glorot_uniform", trainable=True)
        prior_clipped = np.clip(self.prior, 1e-6, 1 - 1e-6)
        b_init = np.log(prior_clipped / (1 - prior_clipped)).astype(np.float32)
        self.b = self.add_weight(name="gate_b", shape=(self.num_features,),
                                  initializer=tf.keras.initializers.Constant(b_init), trainable=True)
        self.prior_const = tf.constant(self.prior, dtype=tf.float32)
        super().build(input_shape)

    def call(self, x):
        z = x * self.w + self.b
        g = tf.sigmoid(z)
        if self.gate_lambda > 0:
            reg = self.gate_lambda * tf.reduce_mean(tf.square(g - self.prior_const))
            self.add_loss(reg)
        return x * g


class AdditiveAttention(layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.W = self.add_weight(name="att_W", shape=(input_shape[-1], self.units),
                                  initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="att_b", shape=(self.units,),
                                  initializer="zeros", trainable=True)
        self.v = self.add_weight(name="att_v", shape=(self.units, 1),
                                  initializer="glorot_uniform", trainable=True)
        super().build(input_shape)

    def call(self, H):
        u = tf.tanh(tf.tensordot(H, self.W, axes=1) + self.b)
        e = tf.tensordot(u, self.v, axes=1)
        e = tf.squeeze(e, axis=-1)
        alpha = tf.nn.softmax(e, axis=1)
        alpha_expanded = tf.expand_dims(alpha, axis=-1)
        context = tf.reduce_sum(H * alpha_expanded, axis=1)
        return context, alpha


class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name="f1_score", **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * p * r / (p + r + tf.keras.backend.epsilon())

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()


def build_model(L_max, F, gate_lambda=0.01, conv_filters=(64, 128), kernel_size=3,
                 lstm_units=64, attention_dim=64, dense_units=64, dropout_rate=0.3,
                 learning_rate=0.001):
    inputs = Input(shape=(L_max, F))
    gated = FeatureGate(F, gate_lambda=gate_lambda)(inputs)

    x = layers.Conv1D(conv_filters[0], kernel_size, activation="relu", padding="same")(gated)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Conv1D(conv_filters[1], kernel_size, activation="relu", padding="same")(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))(x)
    context, attention_weights = AdditiveAttention(attention_dim)(x)

    d = layers.Dense(dense_units, activation="relu")(context)
    d = layers.Dropout(dropout_rate)(d)
    output = layers.Dense(1, activation="sigmoid")(d)

    train_model = Model(inputs=inputs, outputs=output)
    train_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall"), F1Score(name="f1_score")],
    )
    return train_model


def encode_features(df, categorical_columns, exclude_columns, onehot_max_cardinality=12):
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
    final_feature_columns = numeric_columns + onehot_columns
    if numeric_columns:
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns].astype(np.float64))
    for c in onehot_columns:
        df[c] = df[c].astype(np.float32)

    return df, final_feature_columns, scaler


def clean_dataframe(df, categorical_columns, exclude_columns):
    df = df.drop_duplicates()
    for col in df.columns:
        if col in exclude_columns:
            continue
        if col in categorical_columns or not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna("Unknown")
        else:
            df[col] = df[col].fillna(df[col].median())
    return df


def make_sequences(df, feature_columns, label_array, L_max=L_MAX):
    values = df[feature_columns].values.astype(np.float32)
    labels = np.asarray(label_array, dtype=np.float32)
    n = len(values)
    F = len(feature_columns)
    X, y = [], []
    for start in range(0, n, L_max):
        end = min(start + L_max, n)
        window = values[start:end]
        window_labels = labels[start:end]
        pad = L_max - window.shape[0]
        if pad > 0:
            window = np.vstack([np.zeros((pad, F), dtype=np.float32), window])
        X.append(window)
        y.append(float(window_labels[-1]))
    return np.stack(X), np.array(y, dtype=np.float32)


def class_weights_for(y):
    n_pos = float(y.sum())
    n_neg = float(len(y) - n_pos)
    total = n_pos + n_neg
    return {
        0: total / (2.0 * n_neg) if n_neg > 0 else 1.0,
        1: total / (2.0 * n_pos) if n_pos > 0 else 1.0,
    }
