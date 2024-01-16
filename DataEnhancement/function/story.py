from datetime import datetime, timedelta
import pandas as pd
import networkx as nx
import community.community_louvain as community_louvain
from collections import defaultdict
from statistics import mode, StatisticsError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import math

import itertools
from difflib import SequenceMatcher



def get_story(df):
    '''
    this code is mainly from https://github.com/svrijenhoek/RADio/blob/f1ce0d8bb0d7235f0c48b1745a8a81060683846a/dart/preprocess/identify_stories.py
    
    Parameters
    ----------
    df: dataframe with id,text,date columns

    Returns
    -------
    df : extended dataframe with new 'story' column
    '''

    categories = list(df['category'].unique())
    v = TfidfVectorizer(stop_words='english')
    threshold = 0.5
    cosines = []

    df['date'] = pd.to_datetime(df['date']) # datetime.strptime("2019-10-01", '%Y-%m-%d')
    first_date = df.date.min() 
    last_date = df.date.max() 
    delta = last_date - first_date

    for i in range(delta.days+1):
        today = first_date + timedelta(days=i)
        yesterday = today - timedelta(days=1)
        past_three_days = today - timedelta(days=3)
            
        documents_3_days = df.loc[(df['date'] >= past_three_days) & (df['date'] <= today)] # self.handlers.articles.get_all_in_timerange(past_three_days, today)
        documents_1_day = df.loc[(df['date'] >= yesterday) & (df['date'] <= today)] # self.handlers.articles.get_all_in_timerange(yesterday, today)
            
        for category in categories:
            subset_3 = documents_3_days.loc[documents_3_days['category'] == category]
            subset_1 = documents_1_day.loc[documents_1_day['category'] == category]

            if not subset_1.empty and not subset_3.empty:

                subset_1_matrix = v.fit_transform(subset_1['text'].tolist())
                subset_3_matrix = v.fit_transform(subset_1['text'].tolist())

                cosine_similarities = cosine_similarity(subset_1_matrix, subset_3_matrix)

                for x, row in enumerate(cosine_similarities):
                    for y, cosine in enumerate(row):
                        if threshold <= cosine < 1:
                            x_id = list(subset_1.index.values)[x]
                            y_id = list(subset_3.index.values)[y]
                            cosines.append({'x': x_id, 'y': y_id, 'cosine': cosine})   
   

    cosine_df = pd.DataFrame(cosines)
    cosine_df = cosine_df.drop_duplicates()

    G = nx.from_pandas_edgelist(cosine_df, 'x', 'y', edge_attr=True)
    # create partitions, or stories
    partition = community_louvain.best_partition(G)
    stories = partition

    ## rewrite the code to return as series
    df['story'] = 0
    for k, v in stories.items():
        df.at[k, 'story'] = v
        
    return df
