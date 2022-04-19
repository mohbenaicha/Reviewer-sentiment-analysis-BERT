import pandas as pd
import tensorflow as tf
import tensorflow_text as tf_text
from sklearn.base import BaseEstimator, TransformerMixin


class RatingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, label_col: str):

        self.label_col = label_col

    def fit(
        self,
        y: pd.DataFrame,
        X: pd.DataFrame = None,
    ):
        return self

    def transform(self, data):
        new_data = data.copy()
        new_data.loc[new_data[self.label_col] >= 4, "Label"] = 2
        new_data.loc[new_data[self.label_col] == 3, "Label"] = 1
        new_data.loc[new_data[self.label_col] <= 2, "Label"] = 0
        return new_data


class Normalizer(BaseEstimator, TransformerMixin):
    def __init__(self, review_col):
        self.review_col = review_col

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame = None,
    ):
        return self

    def normalize(self, text: str) -> str:
        """
        Normalizes a single string; no tokenization applied!!
        This function will 'detensorify' strings to be 'retensored'
        later when created tensorflow from_tensor_slices.
        """
        text = tf_text.normalize_utf8(text, "NFKD")
        text = tf.strings.lower(text)
        text = tf.strings.regex_replace(text, "[^ a-z]", "")
        text = tf.strings.strip(text).numpy().decode()
        return text

    def transform(self, data):
        new_data = data.copy()
        new_data[self.review_col] = new_data[self.review_col].apply(self.normalize)

        new_data["temp_col"] = new_data[self.review_col].apply(lambda x: len(x))
        new_data.drop(index=new_data[new_data["temp_col"] == 0].index, inplace=True)

        return new_data
