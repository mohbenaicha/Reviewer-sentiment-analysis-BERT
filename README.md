# Review Sentiment Analyzer

This is a Python package and API I built for analyzing reviewer sentment based on millions of user reviews that have I have collected. 
The package and API are kept in this single repo to facilitate access to them; otherwise, the package is downloadable through `pip install --extra-index-url https://pypi.fury.io/mohbenaicha/ review-sentiment==0.0.1` for the latest version and the API can be launched locally as follows:

1. clone this repo, enter shell command `git clone https://github.com/mohbenaicha/Reviewer-sentiment-analysis-BERT`
2. create new Python environment such as: `conda create --name newenv`
3. setup tox `pip install tox`
4. cd into the appropriate directory then cd into sentiment-api/: `cd Reviewer-sentiment-analysis-BERT/recommender-api/`
5.  `tox -e run` (give it some time to setup dependecies and including review-sentiment0.0.1, `tox -e run -vv` for verbosity)
6. Use the link provided by uvicorn to access the API's GUI (assumes port 8001 is free on localhost)
7. Follow the steps to using hte GUI, or send a cURL (see following)

#### Using the API through cURL:
1. While the app is hosted, run the following command from your Bash shell that supports cURL:

```
curl -X 'POST' \
  'http://localhost:8001/api/v1/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '  
  {
  "inputs": [
    {
      "Review": "This book had descent character development."
    },
    {
      "Review": "I thought the plot was shallow and underdeveloped."
    }
  ]
}
```


## Alternatively, get sentiment predictions manually:
1. create new environment such as: `conda create --name newenv`
2. `pip install --extra-index-url https://pypi.fury.io/mohbenaicha/ review-sentiment==0.0.1`
- launch Python cli in the environment used above (`newenv`)
- import the package, unzip the model and make sentiment predictions:

```
from sentiment_model.utilities.data_manager import zip_unzip_model
from sentiment_model.predict import make_prediction

zip_unzip_model(zip=False, test=False)

sent = ['This book did not have a good plot.'] # edit input data per your liking as type: list(str)
print(make_prediction(input_data=pd.DataFrame(data=['I thought the plot was shallow and underdeveloped.', # edit data parameter 
                                                    'This book had descent character development.'],
                                              columns=['Review']
                                              ), 
                     test=False
                     )
      )
```

## Appendix: Notes and Disclaimers:

- Note: If you're using the API, status code 200 means the post request was valid and should yield recommendations within the response body for the user ids you've provided.
