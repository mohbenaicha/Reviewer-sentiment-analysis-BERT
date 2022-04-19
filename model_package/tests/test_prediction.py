import numpy as np

from sentiment_model.predict import make_prediction


def test_make_prediction(test_input_data):
    # Given
    expected_no_predictions = len(test_input_data)
    expected_first_prediction_value = np.array([0, 0, 1]).argmax()
    print("expected_first_prediction_value", expected_first_prediction_value)
    # Test
    results = make_prediction(input_data=test_input_data, test=True)
    predictions = results.get("predictions")

    # Assert
    assert results.get("errors") is None
    assert len(predictions) == expected_no_predictions
    assert isinstance(predictions, list)
    assert isinstance(predictions[-1], list)
    assert np.array(predictions[0]).argmax() == expected_first_prediction_value
