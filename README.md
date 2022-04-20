This is an NLP sentiment analysis model built using TF Hub's pre-constructed BERT preprocessors and encoders along with a classifier.  The model can be tested through it's api or directly installing the package.

### Access through the API:

1. In a bash shell, enter command `pip install tox` (preferrably in a new environment) cd into an empty directory
2. enter command, `git init` then, `git clone https://github.com/mohbenaicha/Reviewer-sentiment-analysis-BERT`
3. cd into /sentiment-api
4. a/ enter command `tox -e run`, then use the url provided by uvicorn and follow the instructions on usin the API's GUI; alternatively, send a cURL request to the API:

```
curl -X 'POST' \
  'http://localhost:8001/api/v1/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "inputs": [
    {
      "Review": "An example of a positive review."
    },
    {
      "Review": "An example of a negative comment."
    }
  ]
}'
```

### Direct install
1. In a new environment, `pip install --extra-index-url https://pypi.fury.io/mohbenaicha/ review-sentiment==0.0.1`
2. Lauch Python
```
# making a prediction requires unzipping the model and that the data be in a Pandas DataFrame since it 
# was primarily built to be used with API, hence direct usage of the model entails a number of steps

import pandas as pd
from sentiment_model.predict import make_prediction
from sentiment_model.utilities.data_manager import zip_unzip_model

zip_unzip_model(test=False, zip=False) # unzip
make_prediction(input_data=pd.DataFrame(data=[{'Review': 'This is a positive review'}, 
                                              {'Review': 'This is a negative review'}]))
```

