import nltk
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer

# initialize NLTK sentiment analyzer
analyzer = SentimentIntensityAnalyzer()


def get_sentiment(text):
    ''' Enhance the dataset with its sentiment (-1.0,1.0) using nltk SentimentIntensityAnalyzer (https://www.nltk.org/howto/sentiment.html)

    Parameters
    ---------
    text: string, each row of news article text in dataframe

    Returns
    -------
    sentiment: float, polarity compound score

    '''

    scores = analyzer.polarity_scores(text)
    sentiment = scores['compound']

    return sentiment


