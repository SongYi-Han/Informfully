import numpy as np
from transformers import pipeline


def get_category(row, **kwargs):
	''' Enhance the dataset with its category (e.g. news, sports, life)
  	This function requires category list defined by user

  	Parameters
  	----------
  	row: string, news id or article text in input file dataframe
  	kwargs : list or dataframe, candidate labels or meta data dataframe


  	Returns
  	-------
  	cat: string, corresponding category name for each news id row

  	'''

	if 'candidate_labels' in kwargs.keys() :
		candidate_labels = kwargs['candidate_labels']

		# initialize huggingface zero shot classifier
		classifier = pipeline("zero-shot-classification")

		# run classifier
		res = classifier(row, candidate_labels, multi_label=True)
	
		categories = res['labels']
		scores = res['scores']

		# get the highest score with a threshold 
		if max(scores) > 0.5:
			i = np.argmax(scores)
			cat = categories[i]
		else:
			cat = -1
		return cat

	else:
		meta_data = kwargs['meta_data']

		cat = meta_data[meta_data['id']==row]['category']
		if not cat.empty : 
			cat = cat.values[0]
		else:
			cat = -1

		return cat

