from typing import Generator, List

import pytest
from fastapi.testclient import TestClient
from sentiment_model.config.base import config
from sentiment_model.utilities.data_manager import load_dataset

from app.main import app


@pytest.fixture(scope="module")
def test_data() -> List[str]:
    return load_dataset(
        file_name=config.app_config.test_data_file,
        features_to_drop=None
    )


@pytest.fixture()
def client() -> Generator:
    with TestClient(app) as _client:
        yield _client
        app.dependency_overrides = {}
