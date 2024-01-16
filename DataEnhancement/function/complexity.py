import nltk
nltk.download('punkt')
from readability import Readability


def get_complexity(text):
  ''' Enhance the dataset with its complexity using python Readability library (https://pypi.org/project/readability/)

  Parameters
  ----------
  text: string, each row of news article text in dataframe

  Returns
  -------
  score: int, Flesch-Kincaid Grade Level (https://pypi.org/project/py-readability-metrics/)

  '''

  r = Readability(text)
  try: 
    fk = r.flesch_kincaid()
    score = fk.score
  except: 
    score = None
      
  return score 


