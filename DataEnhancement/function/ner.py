import numpy as np
import pandas as pd
import requests

import spacy
from spacy import displacy

from collections import Counter
import itertools
from difflib import SequenceMatcher
from collections import defaultdict

import networkx as nx
import community.community_louvain as community_louvain


# Initialize spacy 
NER = spacy.load("en_core_web_sm")

def get_ner(text, **kwargs):
  ''' Enhance the dataset with its named entity tags using Spacy and clustering using community louvain algorithm

    Parameters
    ----------
    text: string, each row of news article text in dataframe

    Returns
    -------
    ner: dictionary, key is named entity and value is a list of tuples with frequency of its named entities

  '''
  ner = []
  ne_list = []
  text= NER(text)
  entities = kwargs['entities']

  # start named entity recognition 
  for word in text.ents:

    if word.label_ in entities:
      ner.append({'text': word.text, 'alternative':[],  'label': word.label_, 'start_char': word.start_char, 'end_char': word.end_char})
    else:
      continue

  # calculate the distances between all the entity names which has same label
  types = list(set([x['label'] for x in ner]))
  for entity_type in types:
    of_type = [entity for entity in ner if entity['label'] == entity_type]
    names = [entity['text'] for entity in of_type]
    distances = []
    for a, b in itertools.combinations(names, 2):
      if len(a) < 3 or len(b) < 3: #  If the label has less than three characters, skip
        similarity = 0
      elif a in b or b in a: # If a is contained in b or the other way around (as in "Barack Obama", "Obama"), full match
        similarity = 1
      else :
        similarity = SequenceMatcher(None, a, b).ratio() # Otherwise calculate the SequenceMatcher ratio to account for slight spelling errors
      distances.append({'a': a, 'b': b, 'metric': similarity})

    # Cluster the names based on the distances found
    if distances:
      df = pd.DataFrame(distances)
      thr = df[df.metric > 0.9]
      G = nx.from_pandas_edgelist(thr, 'a', 'b', edge_attr='metric')
      clusters = community_louvain.best_partition(G)
      if clusters: 
        v, k = max((v, k) for k, v in clusters.items())
        length = v + 1
      else:
        clusters = {}
        length = 0
    else: 
      clusters = {}
      length = 0

    temp = {}
    for name in names:
      if name in clusters:
        temp[name] = clusters[name]
      else:
        temp[name] = length
        length += 1

    d = defaultdict(list)
    for k, v in temp.items():
      d[v].append(k)
    processed = dict(d)
    
    for _, v in processed.items():
        # find all entries of this cluster
        with_name = [entity for entity in of_type if entity['text'] in v]
        # find all unique occurrences of the cluster
        all_names = [entity['text'] for entity in with_name]
        label = with_name[0]['label']
        # find the name that was most often used to refer to this cluster
        most_frequent_name = max(set(all_names), key=all_names.count)
        # if the cluster is about people names, favor names that contain a space
        # eg: Barack Obama over just Obama
        if label == 'PERSON':
          with_space = [name for name in all_names if len(name.split(" ")) > 1]
          if len(with_space) > 0:
            most_frequent_name = max(set(with_space), key=with_space.count)
        alternative_names = v
        spans = [(entity['start_char'], entity['end_char']) for entity in with_name]
        ne_list.append(dict({'text': most_frequent_name, 
                             'alternative': alternative_names, 
                             'spans': spans,
                             'frequency': len(with_name),
                             'label': label}))
  
  return ne_list
