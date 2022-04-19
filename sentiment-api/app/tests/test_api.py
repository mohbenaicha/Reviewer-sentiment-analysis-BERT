import math
from typing import List

from fastapi.testclient import TestClient


def test_make_prediction(client: TestClient, test_data: pd.DataFrame) -> None:
    # Test
    payload = {
        "inputs": test_data.to_dict(orient="records")
    }


    response = client.post(
        "http://localhost:8001/api/v1/predict",
        json=payload,
    )
    
    # Assert
    assert response.status_code == 200
    

    # Test
    result = response.json()
    expected_no_predictions = len(test_data)
    expected_first_prediction_value = np.array([0, 0, 1]).argmax()
    print("expected_first_prediction_value", expected_first_prediction_value)
    


    # Assert
    assert results.get("errors") is None
    assert len(predictions) == expected_no_predictions
    assert isinstance(predictions, list)
    assert isinstance(predictions[-1], list)
    assert np.array(predictions[0]).argmax() == expected_first_prediction_value
