import json

from sentiment_model.config.base import BERT_MAPPERS_DIR

with open(BERT_MAPPERS_DIR / "map_model_to_preprocess.json", "r") as f:
    map_model_to_preprocess = json.load(f)
    f.close()

with open(BERT_MAPPERS_DIR / "map_name_to_handle.json", "r") as f:
    map_name_to_handle = json.load(f)
    f.close()


# if __name__ == '__main__':
# import json
# json.dump(map_model_to_preprocess, open(
# "map_model_to_preprocess.json", "w"))
# json.dump(map_name_to_handle, open(
# "map_name_to_handle.json", "w"))
