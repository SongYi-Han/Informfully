import requests
import json
import pandas as pd

def get_enriched_ne(ne_list):
    ''' Enhance the dnamed entity tags sending sparql to wikidata 

    Parameters
    ----------
    ne_list: list, list of dictionaries with named entities extracted from original text 

    Returns
    -------
    enriched_ner: list, list of dictionaries with person and organization as key and its extended information as value

    '''

    enriched_ne = []
    for entity in ne_list:

        if entity['label'] == 'PERSON':
            #print("found person")
            name = entity['text']
            full_name, info = get_person_data(entity['alternative'])
            info['key'] = name
            info['frequency'] = entity['frequency']
            info['alternative'] = entity['alternative']
            enriched_ne.append(dict({full_name:info}))

        if entity['label'] == 'ORG':
            #print("found organization")
            name = entity['text']
            name, info = get_org_data(name)
            info['frequency'] = entity['frequency']
            info['alternative'] = entity['alternative']
            enriched_ne.append(dict({name:info}))

    return enriched_ne


def get_person_data(alternatives):
    """
    get the respond from Wikidata.
    """    
    for alternative in alternatives:
        if " " in alternative:
            #send a query with full name 
            response = person_data_query(alternative)
            if response and (response['givenname'] or response['gender'] or response['citizen']):
                return alternative, response
        return alternative, {}


def get_org_data(name):
    """
    get the respond from Wikidata.
    """    
    #send a query with org name 
    response = org_data_query(name)
    if response and (response['ideology']):
        return name, response
    return name, {}


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

def read_response(data, label):
    """
    Returns all unique values for a particular label.
    Lowercases occupation data, which is relevant when returning results in English
    """
    output = []
    for x in data['results']['bindings']:
        if label in x:
            if label == 'occupations':
                output.append(x[label]['value'].lower())
            else:
                output.append(x[label]['value'])
    return list(set(output))


def read_person_response_list(response):
    """
    Attempt to retrieve values for each of the value types relevant for people data.
    Leaves value empty in case Wikidata has no information about it.
    """
    try:
        data = response.json()

        givenname = read_response(data, 'givenname')
        familyname = read_response(data, 'familyname')
        occupations = read_response(data, 'occupations')
        party = read_response(data, 'party')
        gender = read_response(data, 'gender')
        citizen = read_response(data, 'citizen')
        ethnicity = read_response(data, 'ethnicity')
        place_of_birth = read_response(data, 'place_of_birth')

        return {
            'givenname': givenname,
            'familyname': familyname,
            'gender': gender,
            'occupations': occupations,
            'party': party,
            'citizen': citizen,
            'ethnicity': ethnicity,
            'place_of_birth': place_of_birth
        }

    except json.decoder.JSONDecodeError:
        return []
    except IndexError:
        return []


def read_org_response_list(response):
    """
    Attempt to retrieve values for each of the value types relevant for organization data.
    Leaves value empty in case Wikidata has no information about it.
    """
    try:
        data = response.json()

        ideology = read_response(data, 'ideology')

        return {'ideology': ideology}

    except json.decoder.JSONDecodeError:
        return []


def org_data_query(label):
    """
    Create SPARQL query 
    """
    language_tag = 'en'
    try:
        query = """
        SELECT DISTINCT ?ideology WHERE { 
            ?s wdt:P31 wd:Q7210356. # instance of political organization
            ?s rdfs:label '""" + label + """'@""" + language_tag + """ .

            OPTIONAL {
              ?s wdt:P1142 ?a .
              ?a rdfs:label ?ideology .
              FILTER(LANG(?ideology) = "" || LANGMATCHES(LANG(?ideology), \"""" + language_tag + """\")) }                
        }            
        """

        r = execute_query(query)
        return read_org_response_list(r)

    except ConnectionAbortedError:
        return []


def person_data_query(label):
    """
    Create SPARQL query 
    """
    language_tag = 'en'
    try:
        query = """
            SELECT DISTINCT ?s ?givenname ?familyname ?occupations ?party ?position ?gender ?citizen ?ethnicity ?place_of_birth ?sexuality WHERE { 
            ?s ?label '""" + label + """'@""" + language_tag + """ .
          OPTIONAL {
            ?s wdt:P735 ?a . 
            ?a rdfs:label ?givenname .
            FILTER(LANG(?givenname) = "" || LANGMATCHES(LANG(?givenname), \"""" + language_tag + """\"))
          }
          OPTIONAL {
            ?s wdt:P734 ?b . 
            ?b rdfs:label ?familyname .
            FILTER(LANG(?familyname) = "" || LANGMATCHES(LANG(?familyname), \"""" + language_tag + """\"))
          }
          OPTIONAL {
            ?s wdt:P106 ?c .
            ?c rdfs:label ?occupations .
            FILTER(LANG(?occupations) = "" || LANGMATCHES(LANG(?occupations), \"""" + language_tag + """\"))
          }
          OPTIONAL {
            ?s wdt:P102 ?d .
            ?d rdfs:label ?party .
            FILTER(LANG(?party) = "" || LANGMATCHES(LANG(?party), \"""" + language_tag + """\"))
          }
          OPTIONAL {
            ?s wdt:P21 ?f .
            ?f rdfs:label ?gender .
            FILTER(LANG(?gender) = "" || LANGMATCHES(LANG(?gender), \"""" + language_tag + """\"))
          }
          OPTIONAL {
               ?s wdt:P172 ?g . 
               ?g rdfs:label ?ethnicity .
               FILTER(LANG(?ethnicity) = "" || LANGMATCHES(LANG(?ethnicity), \"""" + language_tag + """\"))
            }
           OPTIONAL {
               ?s wdt:P19 ?pb . 
               ?pb wdt:P17 ?country .
               ?country rdfs:label ?place_of_birth .
               FILTER(LANG(?place_of_birth) = "" || LANGMATCHES(LANG(?place_of_birth), \"""" + language_tag + """\"))
            }
          OPTIONAL {
            ?s wdt:P27 ?h .
            ?h rdfs:label ?citizen
            FILTER(LANG(?citizen) = "" || LANGMATCHES(LANG(?citizen), \"""" + language_tag + """\"))
            }
          
        }"""
        
        r = execute_query(query)
        return read_person_response_list(r)

    except (ConnectionAbortedError, requests.exceptions.ChunkedEncodingError):  # in case the connection fails
        return []


