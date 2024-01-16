import sys
import requests
import pandas as pd
import config
from function import category, sentiment, complexity, ner, region, party, enrich_ne, min_maj, story


def enhance_data(input_file_path, output_file_path):
    # Load original dataset
    df = pd.read_csv(input_file_path)
    df = df[3:20]

    # Add Category
    if config.attributes['category']['enrich'] :
        print("Categories are extracting ... ")

        # if user want to extend category with zero-shot classifier, method need to be 'zero-shot' and 'candidate_labels' need to be given in config.py
        if (config.attributes['category']['method'] == 'zero-shot') and config.attributes['category']['candidate_labels'] : 
            df['category'] = df['text'].apply(category.get_category, candidate_labels = config.attributes['category']['candidate_labels'])
        # if user want to extend category with metadata, method need to be 'metadata' and 'meta_file_path' need to be given in config.py
        elif (config.attributes['category']['method'] == 'metadata') and config.attributes['category']['meta_file_path'] : 
            meta_data = pd.read_csv(config.attributes['category']['meta_file_path'])
            df['category'] = df['id'].apply(category.get_category, meta_data = meta_data)
        else:
            raise Exception("If you want to enrich the category feature, you need to provide either candidate labels (e.g. ['news','sports', ...]) or metadata file which includes item id and category data.")
        

    # Add Sentiment
    if config.attributes['sentiment']['enrich'] :
        print("Sentiments are extracting ... ") 
        df['sentiment'] = df['text'].apply(sentiment.get_sentiment)
    
    # Add Complexity
    if config.attributes['complexity']['enrich'] :
        print("Complexities are extracting ... ") 
        df['complexity'] = df['text'].apply(complexity.get_complexity)
    
    # Add Named Entity
    if config.attributes['named_entities']['enrich'] : 
        print("Named entities are extracting ... ")
        df['named_entities'] = df['text'].apply(ner.get_ner, entities=config.attributes['named_entities']['entities'] )

        if config.attributes['named_entities']['extend_ne_with_wiki']:
            print("Enriching the people named entities with wikidata ...") 
            df['enriched_ne'] = df['named_entities'].apply(enrich_ne.get_enriched_ne)

    # Add region
    if config.attributes['region']['enrich'] :
        if 'named_entities' not in df.columns or ('GPE' not in config.attributes['named_entities']['entities']):
            raise Exception("If you want to enrich the region feature, you need a named entity information. Please mark as True for named entities and add 'GPE' in entities attribute in config file.")
        
        print("Regions are extracting ... ") 
        df['region'] = df['named_entities'].apply(region.get_region)
           
    # Add political party
    if config.attributes['political_party']['enrich'] :
        if 'enriched_ne' not in df.columns:
            raise Exception("If you want to enrich the political party feature, you need a extended named entity information. Please mark as True for named entities  and enrich with wiki attribute in config file.")
        
        print("Political parties are extracting ... ")
        df['party'] = df['enriched_ne'].apply(party.get_party)

    # Add alternative    
    if config.attributes['min_maj_ratio']['enrich'] : 

        if 'enriched_ne' not in df.columns:
            raise Exception("If you want to enrich the miniority-majority ratio feature, you need a named entity information. Please mark as True for named entities and enrich with wiki attribute in config file.")

        major_gender = config.attributes['min_maj_ratio']['major_gender']
        major_citizen = config.attributes['min_maj_ratio']['major_citizen']
        major_ethinicity = config.attributes['min_maj_ratio']['major_ethinicity']
        major_place_of_birth = config.attributes['min_maj_ratio']['major_place_of_birth']

        if not major_gender or not major_citizen or not major_ethinicity or not major_place_of_birth:
            raise ValueError("There must be at least one value in major_gender (e.g. male), major_citizen (e.g. United States), major_ethinicity (e.g. White), and major_place_of_birth (e.g. France)")

        print("Minority-Majority ratios are extracting ... ")
        df['min_maj_ratio'] = df['enriched_ne'].apply(min_maj.get_min_maj_ratio, major_gender=major_gender, major_citizen=major_citizen, major_ethinicity=major_ethinicity, major_place_of_birth=major_place_of_birth)

    # Add story
    if config.attributes['story']['enrich'] : 
        print("Stories are extracting ... ")
        df_story = story.get_story(df)
        pd.concat([df, df_story], axis=1)

    # save enriched data to json file
    df = df.set_index('id')
    df.to_json(config.output_file_path, orient='index')


if __name__ == '__main__':
    
    enhance_data(config.input_file_path, config.output_file_path) 
    sys.exit(f"Dataset is succsssfully enhanced! Check out {config.output_file_path}")
