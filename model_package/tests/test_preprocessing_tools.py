from sklearn.pipeline import Pipeline

from sentiment_model.config.base import config
from sentiment_model.utilities.preprocessing_tools import Normalizer, RatingTransformer


def test_preprocessing_pipeline(test_input_data):

    # Test
    preprocess_pipeline = Pipeline(
        [
            ("Normalize", Normalizer(review_col=config.model_config.feature_col)),
            (
                "LabelTransform",
                RatingTransformer(label_col=config.model_config.target_col),
            ),
        ]
    )

    test_object_1 = preprocess_pipeline.fit_transform(test_input_data)

    # Assert
    assert (
        test_object_1["Review"][0]
        == "fascinating  funny the essence of life captured d"
    )
    assert test_object_1["Label"][0] == 2


def test_saved_model_packaging(test_zip_unzip):
    # Test: test_zip_unzip

    # Assert
    assert test_zip_unzip == (True, True)
