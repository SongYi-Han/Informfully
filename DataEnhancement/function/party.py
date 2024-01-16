import pandas as pd
import numpy as np


def get_party(ne_list):
  '''Enhance the dataset with political parties  using named entity tags and external lookup table

    Parameters
    ----------
    ne_list: list, list of dictionaries with named entities extracted from original text 
    
    Returns
    -------
    parties: dictionary, key is a political parties appeared in the text and values is frequency
  '''
  parties = {}

  for entity in ne_list: 
    for entity_name, entity_dict in entity.items(): 

      # if there is a party attribute in extended named entity
      if 'party' in entity_dict.keys():
        if entity_dict['party'] : 
          for party in entity_dict['party']: 
            if not parties.get(party):
              parties[party] = 0
            parties[party] += entity_dict['frequency']

      # if there is a ideology attribute in extended named entity
      if 'ideology' in entity_dict.keys():
        if entity_dict['ideology'] : 
          for party in entity_dict['ideology']: 
            if not parties.get(party):
              parties[party] = 0
            parties[party] += entity_dict['frequency']

  return parties

