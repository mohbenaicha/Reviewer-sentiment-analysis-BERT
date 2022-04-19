from sentiment_model.config.base import config
from sentiment_model.pipeline import Pipeline
from sentiment_model.utilities.data_manager import load_dataset, save_model


def train() -> None:
    loaded_data = load_dataset(
        file_name=(f"{config.app_config.train_data_file}"), features_to_drop=None
    )

    # instantiate pipeline
    pipeline = Pipeline(data=loaded_data)

    # Normalize and transform labels
    pipeline.preprocess(data=None)

    # train BERT and classifier dnn and persist
    pipeline.train_classifier()
    save_model(model_to_persist=(pipeline.classifier_model))


if __name__ == "__main__":
    train()
