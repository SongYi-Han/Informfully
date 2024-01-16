import requests
import json
import pandas as pd

def get_region(ne_list):
    ''' Enhance the dataset with its region (e.g. city, country and so on)
    This function requires named entity list 

    Parameters
    ----------
    ne_list: list, list of dictionaries with named entities extracted from original text 
    
    
    Returns
    -------
    regions: set, all geographical name such as city and contury

    '''

    regions = []

    for ner in ne_list:
        if ner['label'] == 'GPE':
            region = get_region_data(ner['text'])

            if region =='true' : 
                regions.append(ner['text'].lower())

    return set(regions)


def get_region_data(region):
    """
    Check if the region name is existing in wikidata (if it is the valid name)
    """
    
    r = region_data_query(region)
    try:
        data = r.json()
        print('found')
        return data['results']['bindings'][0]['boolean']['value'] 
    except: 
        return []
    

def execute_query(query):
    """
    Sends a SPARQL query to Wikidata.
    """
    url = 'https://query.wikidata.org/sparql'

    try:
        r = requests.get(url, params={'format': 'json', 'query': query})
        return r

    except (ConnectionAbortedError, ConnectionError):
        return ConnectionError


def region_data_query(label):
    """
    Create SPARQL query 
    """
    language_tag = 'en'
    try:
        query = """
                SELECT ?boolean WHERE { 
                BIND(EXISTS{?item ?label '""" + label + """'@"""+language_tag+""" .} AS ?boolean)
                }"""
        r = execute_query(query)
        return r
            

    except (ConnectionAbortedError, requests.exceptions.ChunkedEncodingError):  # in case the connection fails
        return []

        