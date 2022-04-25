import os, glob
from pathlib import Path
from tqdm import tqdm
from nltk import word_tokenize

import requests

from collections import Counter

import numpy as np
import pandas as pd

def filter_ConceptNet(word, language_key='en', same_language=True):

    # call the ConceptNet API
    json_object = requests.get(f'http://api.conceptnet.io/c/{language_key}/{word}').json()
    
    # check if the concept exists in the graph
    try:
        query_error = json_object['error']['status'] == 404
    except KeyError:
        query_error = False        
    
    # if queried concept exists in conceptnet
    if not query_error:
        # create a dataframe from the nested json-object
        df = pd.json_normalize(json_object, record_path='edges', meta=['@id'], record_prefix='_')        
        try:
            df = df[df['_start.language'] == language_key]

            # dataframe might be empty due to language filtering
            if not df.empty:

                # assert that all rows are of form (Node, Edge, Node)
                try:
                    type_assertion = ['_start.@type', '_@type', '_end.@type']
                    assert all(df[type_assertion] == ('Node', 'Edge', 'Node'))

                except AssertionError:
                    df = df[np.all(df[['_start.@type', '_@type', '_end.@type']] == ('Node', 'Edge', 'Node'), axis=1)]

                # attributes of interest
                aoi = ['start_label', 'relation', 'end_label', 'surfaceText', 'weight', 'dataset']

                # only find relations with identical language
                if same_language:
                    df['language'] = [language_key] * df.__len__()
                    language = [concept['_start.language'] == concept['_end.language'] for (_, concept) in df.iterrows()]
                    df = df[language]
                    aoi.append('language')
                else:
                    df = df.rename(columns={'_start.language':'start_language',
                                            '_end.language': 'end_language'})
                    aoi += ['start_language', 'end_language']

                # rename columns
                df.columns = df.columns.str.lstrip("_")
                df = df.rename(columns={'start.label':'start_label', 
                                        'rel.label':'relation', 
                                        'end.label':'end_label',
                                       })

                df = df[aoi].reset_index(drop=True)
                df['query_word'] = [word] * df.__len__()
                return df
        except KeyError:
            pass
    
    # if concept doesn't exist in conceptnet
    else:
        return pd.DataFrame()
        #raise KeyError("Concept does not exist in ConceptNet...")


# path to okvqa dataset (from text-investigation)
filename = 'OKVQA_object.json'
data_path = Path(os.getcwd()) / 'data_investigation/data'
print(data_path)

okvqa = {}
okvqa['full'] = pd.read_json(data_path/filename)


all_answers = okvqa['full'].answers.apply(pd.Series).stack().reset_index(drop=True)
unique_answers = all_answers.unique()



all_answer_tokens = []
for answer in tqdm(unique_answers):
    all_answer_tokens += word_tokenize(answer)

all_answer_tokens = set(all_answer_tokens)


temp_conceptnet = pd.DataFrame()

for answer_token in tqdm(all_answer_tokens):
    temp = filter_ConceptNet(answer_token, language_key='en', same_language=True)
    temp_conceptnet = temp_conceptnet.append(temp)

temp_conceptnet.reset_index(drop=True, inplace=True)

# save data
temp_conceptnet.to_json('data_investigation/data/okvqa_answers_conceptnet.json')


