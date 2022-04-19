import os
import shutil
import warnings
import zipfile
from pathlib import Path
from typing import List

import pandas as pd

from sentiment_model import __version__ as _version
from sentiment_model.config.base import DATASET_DIR, TRAINED_MODEL_DIR, config


def func():
    warnings.warn("deprecated", DeprecationWarning)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    func()
    import tensorflow as tf
    from tensorflow.python.keras.engine.functional import Functional


def load_dataset(
    *, file_name: str = None, features_to_drop: List[str] = None
) -> pd.DataFrame:

    print("Attempting to load data...")
    df = pd.read_csv(
        (f"{DATASET_DIR}/{file_name}" + f"{config.app_config.data_version}.csv")
    )
    if features_to_drop:
        df.drop(columns=features_to_drop, inplace=True)

    # The decision to drop nan is not arbitrary since nan values are almost
    # useless  when it comes to sentiment analysis.
    df.dropna(axis=0, inplace=True)
    return df


def save_model(*, model_to_persist: Functional) -> None:
    # define name pipeline of newely trained model
    save_folder_name = f"{config.app_config.model_name}{_version}"
    save_path = TRAINED_MODEL_DIR / save_folder_name

    # a mix of Path objects and string formatting is used here since
    # the Keras save method doesn't work well with Path objects

    if Path(save_path).exists():
        shutil.rmtree(save_path)
    tf.keras.models.save_model(
        model_to_persist, os.path.join(save_path, save_folder_name)
    )


def zip_unzip_model(
    folder_path: str = os.path.join(
        TRAINED_MODEL_DIR, f"{config.app_config.model_name}{_version}"
    ),
    zip_path: str = os.path.join(
        TRAINED_MODEL_DIR, f"{config.app_config.zipped_model_name}{_version}.zip"
    ),
    zip: bool = True,
    test: bool = False,
):
    if zip:
        with zipfile.ZipFile(
            zip_path, mode="w", compression=zipfile.ZIP_DEFLATED
        ) as zipf:
            len_dir_path = len(folder_path)
            for root, _, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, file_path[len_dir_path:])
            zipf.close()
            shutil.rmtree(folder_path)
            if test:
                result = os.path.isdir(folder_path) and os.path.isfile(
                    zip_path
                )  # false and true
                return result is False and True

    else:
        if test:
            os.mkdir(folder_path)
        else:
            folder_path = TRAINED_MODEL_DIR
        with zipfile.ZipFile(file=zip_path, mode="r") as f:
            f.extractall(folder_path)
            f.close()
            os.remove(zip_path)
            if test:
                result = os.path.isfile(zip_path) and os.path.isdir(folder_path)
                return result is False and True
