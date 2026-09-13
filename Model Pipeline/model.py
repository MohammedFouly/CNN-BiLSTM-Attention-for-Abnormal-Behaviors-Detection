import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input


class FeatureGate(layers.Layer):
    def __init__(self, num_features, prior=None, gate_lambda=0.0, **kwargs):
        super().__init__(**kwargs)
        self.num_features = num_features
        self.gate_lambda = gate_lambda
        if prior is None:
            prior = np.full((num_features,), 0.5, dtype=np.float32)
        self.prior = np.asarray(prior, dtype=np.float32)

    def build(self, input_shape):
        self.w = self.add_weight(
            name="gate_w", shape=(self.num_features,),
            initializer="glorot_uniform", trainable=True,
        )
        prior_clipped = np.clip(self.prior, 1e-6, 1 - 1e-6)
        b_init = np.log(prior_clipped / (1 - prior_clipped)).astype(np.float32)
        self.b = self.add_weight(
            name="gate_b", shape=(self.num_features,),
            initializer=tf.keras.initializers.Constant(b_init), trainable=True,
        )
        self.prior_const = tf.constant(self.prior, dtype=tf.float32)
        super().build(input_shape)

    def call(self, x):
        z = x * self.w + self.b
        g = tf.sigmoid(z)
        if self.gate_lambda > 0:
            reg = self.gate_lambda * tf.reduce_mean(tf.square(g - self.prior_const))
            self.add_loss(reg)
        return x * g

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_features": self.num_features,
            "prior": self.prior.tolist(),
            "gate_lambda": self.gate_lambda,
        })
        return config


class AdditiveAttention(layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_W", shape=(input_shape[-1], self.units),
            initializer="glorot_uniform", trainable=True,
        )
        self.b = self.add_weight(
            name="att_b", shape=(self.units,),
            initializer="zeros", trainable=True,
        )
        self.v = self.add_weight(
            name="att_v", shape=(self.units, 1),
            initializer="glorot_uniform", trainable=True,
        )
        super().build(input_shape)

    def call(self, H):
        u = tf.tanh(tf.tensordot(H, self.W, axes=1) + self.b)
        e = tf.tensordot(u, self.v, axes=1)
        e = tf.squeeze(e, axis=-1)
        alpha = tf.nn.softmax(e, axis=1)
        alpha_expanded = tf.expand_dims(alpha, axis=-1)
        context = tf.reduce_sum(H * alpha_expanded, axis=1)
        return context, alpha

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


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


def build_domain_prior_vector(F, high_risk_indices=None, high_value=0.8, neutral_value=0.5):
    prior = np.full((F,), neutral_value, dtype=np.float32)
    if high_risk_indices:
        for idx in high_risk_indices:
            prior[idx] = high_value
    return prior


def build_model(L_max, F, domain_prior_indices=None, domain_prior_high=0.8,
                 domain_prior_neutral=0.5, gate_lambda=0.0, conv_filters=(64, 128),
                 kernel_size=3, lstm_units=128, attention_dim=128, dense_units=128,
                 dropout_rate=0.3, learning_rate=0.001):
    prior = build_domain_prior_vector(F, domain_prior_indices, domain_prior_high, domain_prior_neutral)

    inputs = Input(shape=(L_max, F))
    gated = FeatureGate(F, prior=prior, gate_lambda=gate_lambda)(inputs)

    x = layers.Conv1D(filters=conv_filters[0], kernel_size=kernel_size,
                       activation="relu", padding="same")(gated)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Conv1D(filters=conv_filters[1], kernel_size=kernel_size,
                       activation="relu", padding="same")(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))(x)

    context, attention_weights = AdditiveAttention(attention_dim)(x)

    d = layers.Dense(dense_units, activation="relu")(context)
    d = layers.Dropout(dropout_rate)(d)
    output = layers.Dense(1, activation="sigmoid")(d)

    model = Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            F1Score(name="f1_score"),
        ],
    )
    return model


def build_attention_extractor(model):
    attention_layer = None
    for layer in model.layers:
        if isinstance(layer, AdditiveAttention):
            attention_layer = layer
            break
    return Model(inputs=model.input, outputs=attention_layer.output)


def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=64,
                 class_weight=None, callbacks=None):
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )
    return history


def evaluate_model(model, X_test, y_test, batch_size=64):
    results = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0, return_dict=True)
    return results
