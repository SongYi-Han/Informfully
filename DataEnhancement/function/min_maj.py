import numpy as np
import pandas as pd


def get_min_maj_ratio(ne_list, **kwargs): 
  '''Enhance the dataset with minority-majority score using named entity tags extended by wikidata

    Parameters
    ----------
    ner_list: list, list of dictionaries, key is the name of person appeared in the text, and values are another dictionary which is extended by wikidata.

    Returns
    -------
    rato: dictionary, key is the score types and value is a list of minority ratio and majority ratio
  '''
  major_genders = kwargs['major_gender']
  major_citizens = kwargs['major_citizen']
  major_ethinicities = kwargs['major_ethinicity']
  major_place_of_births = kwargs['major_place_of_birth']

  count = {'gender' : [0,0], 'ethnicity': [0,0], 'mainstream': [0,0]}
  ratio = {}

  for entity in ne_list:
    for entity_name, entity_dict in entity.items():

      # calculate gender score (male as majority, others are minority)
      if 'gender' in entity_dict.keys() and len(entity_dict['gender']) == 1 :
        if entity_dict['gender'][0] in major_genders:
          count['gender'][1] += entity_dict['frequency']
        else: 
          count['gender'][0] += entity_dict['frequency']

      # calculate Ethnicity score (people with a 'United States' ethnicity or place of birth are majority)
      if 'citizen' in entity_dict:
        loop_break = False
        etinity_match = False
        place_of_birth_match = False

        for major_citizen in major_citizens:
            loop_break = False
            if major_citizen in entity_dict['citizen']:
                loop_break = True

                for major_ethnicity in major_ethinicities:
                    if (major_ethnicity in entity_dict['ethnicity']) or (entity_dict['ethnicity'] == []):
                        etinity_match = True

                for major_place_of_birth in major_place_of_births:
                    if (major_place_of_birth in entity_dict['place_of_birth']) or (entity_dict['place_of_birth'] == []):
                        place_of_birth_match = True

                if (etinity_match and place_of_birth_match):
                    count['ethnicity'][1] += entity_dict['frequency']
                    break

                count['ethnicity'][0] += entity_dict['frequency']
                break

        if not loop_break:
          count['ethnicity'][0] += entity_dict['frequency']

      # calculate mainstream score by checking givenname 
      if 'givenname' in entity_dict.keys() :
        count['mainstream'][1] += entity_dict['frequency']
      else:
        count['mainstream'][0] += entity_dict['frequency']

  for k, v in count.items():
    try: 
      ratio[k] = [round(v[0] /(v[1]+v[0]),4), round(v[1] /(v[1]+v[0]),4)]
    except: 
      continue

  return ratio
