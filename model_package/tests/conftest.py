import pytest

from sentiment_model.config.base import config
from sentiment_model.utilities.data_manager import load_dataset, zip_unzip_model


@pytest.fixture()
def test_input_data():
    return load_dataset(
        file_name=config.app_config.test_data_file,
        features_to_drop=None,
    )


@pytest.fixture()
def test_zip_unzip():
    return (zip_unzip_model(zip=True, test=True), zip_unzip_model(zip=False, test=True))
