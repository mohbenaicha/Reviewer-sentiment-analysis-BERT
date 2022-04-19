import json
import os
import warnings

import numpy

from sentiment_model import __version__ as _version
from sentiment_model.config.base import OPTIMIZER_DIR, TRAINED_MODEL_DIR, config
from sentiment_model.utilities.bert_mapping import (
    map_model_to_preprocess,
    map_name_to_handle,
)


def func():
    warnings.warn("deprecated", DeprecationWarning)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    func()
    import tensorflow as tf
    import tensorflow_addons as tfa
    import tensorflow_hub as hub
    from official.nlp import optimization
    from tensorflow.python.keras.engine.functional import Functional


def build_model(
    train_dataset,
    bert_model_name: str = config.model_config.bert_model_name,
    epochs: int = 5,
    lr: float = 3e-5,
):
    """
    This function builds and compiles a BERT and DNN classifier model for
    text sentiment analysis.
    """

    tfhub_handle_encoder = map_name_to_handle[bert_model_name]
    tfhub_handle_preprocess = map_model_to_preprocess[bert_model_name]

    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="Reviews")
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name="Preporcessing")

    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name="Encoding")
    outputs = encoder(encoder_inputs)
    net = outputs["pooled_output"]
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(3, activation="softmax", name="Classifier")(net)
    model = tf.keras.Model(text_input, net)

    loss = tf.keras.losses.CategoricalCrossentropy()
    metrics = [
        tfa.metrics.F1Score(average="weighted", num_classes=3, name="Weighted-F1"),
        tf.keras.metrics.AUC(num_thresholds=5, name="Area_under_the_curve"),
        tf.keras.metrics.CategoricalAccuracy(),
    ]

    epochs = epochs
    steps_per_epoch = tf.data.experimental.cardinality(train_dataset).numpy()
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1 * num_train_steps)

    init_lr = lr
    optimizer = optimization.create_optimizer(
        init_lr=init_lr,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        optimizer_type="adamw",
    )

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model, optimizer


def load_model(
    *, save_model_name: str = config.app_config.model_name, test: bool = False
) -> Functional:

    save_file_name = f"{save_model_name}{_version}"
    if test:
        save_path = os.path.join(TRAINED_MODEL_DIR, save_file_name, save_file_name)
    else:
        save_path = os.path.join(TRAINED_MODEL_DIR, save_file_name)

    with open(
        os.path.join(
            OPTIMIZER_DIR, f"{config.app_config.optimizer_name}{_version}.json"
        ),
        "r",
    ) as f:
        optimizer = optimization.create_optimizer(
            init_lr=None, num_train_steps=None, num_warmup_steps=None
        ).from_config(json.load(f))
        f.close()

    loaded_model = tf.keras.models.load_model(
        save_path,
        custom_objects={"KerasLayer": hub.KerasLayer, "AdamWeightDecay": optimizer},
    )

    return loaded_model


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        if isinstance(obj, numpy.floating):
            return float(obj)
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)
