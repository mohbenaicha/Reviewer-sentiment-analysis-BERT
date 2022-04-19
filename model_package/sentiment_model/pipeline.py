import json
import os
import warnings
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as P
from tensorflow.python.keras.engine.functional import Functional

import sentiment_model.utilities.model_tools as mt
from sentiment_model import __version__ as _version
from sentiment_model.config.base import OPTIMIZER_DIR, config
from sentiment_model.utilities import preprocessing_tools as pp


def func():
    warnings.warn("deprecated", DeprecationWarning)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    func()
    import tensorflow as tf


class Pipeline:
    def __init__(
        self,
        data: pd.DataFrame,
        trained_model: Functional = None,
    ):
        self.data = data
        self.trained_model = trained_model

    def preprocess(
        self,
        data: pd.DataFrame,
        feature_col: str = config.model_config.feature_col,
        target_col: str = config.model_config.target_col,
    ) -> None:

        self.feature_col = feature_col
        self.target_col = target_col

        if not data and self.data.empty:
            print("Error: no data provided.")
        elif data is None:
            assert not self.data.empty
            print("self.data.empty found")
        else:
            assert isinstance(data, pd.DataFrame)
            self.data = data

        preprocess_pipeline = P(
            [
                ("Normalize", pp.Normalizer(review_col=self.feature_col)),
                ("LabelTransform", pp.RatingTransformer(label_col=self.target_col)),
            ]
        )

        print("Processing data...")
        self.data = preprocess_pipeline.fit_transform(self.data)

    def train_classifier(
        self,
        lr: float = config.model_config.starting_learning_rate,
        n_epochs: int = config.model_config.epochs,
        batch_size: int = config.model_config.batch_size,
        shuffle_buffer: int = config.model_config.shuffle_buffer_size,
    ) -> None:

        X = self.data[self.feature_col]
        y = self.data["Label"]
        print(X.shape, y.shape)

        x_train, x_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=0.2
        )
        y_train, y_test = tf.one_hot(y_train, 3), tf.one_hot(y_test, 3)

        # set up data in tensor format and batch data
        train_batches = (
            tf.data.Dataset.from_tensor_slices((x_train, y_train))
            .shuffle(shuffle_buffer)
            .batch(batch_size)
        )
        test_batches = (
            tf.data.Dataset.from_tensor_slices((x_test, y_test))
            .shuffle(shuffle_buffer // 2)
            .batch(batch_size)
        )

        # build preprocessor, encoder and classifier layer and compile

        self.classifier_model, self.optimizer = mt.build_model(
            lr=lr, train_dataset=train_batches, epochs=n_epochs
        )

        json.dump(
            self.optimizer.get_config(),
            open(
                os.path.join(
                    OPTIMIZER_DIR, f"{config.app_config.optimizer_name}{_version}.json"
                ),
                "w",
            ),
            cls=mt.NumpyEncoder,
        )

        print("Fitting BERT classifier. This may take a while...")
        self.classifier_model.fit(
            train_batches,
            validation_data=test_batches,
            epochs=n_epochs,
            batch_size=batch_size,
        )

    def predict(self, query: pd.DataFrame, test: bool = False) -> List[np.ndarray]:
        feat_col = config.model_config.feature_col
        data = pp.Normalizer(review_col=feat_col).fit_transform(query)

        if self.trained_model:
            loaded_model = self.trained_model
        else:
            loaded_model = mt.load_model(test=test)

        results = loaded_model.predict(data[feat_col])

        return results
