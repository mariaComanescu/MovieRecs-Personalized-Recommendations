from ast import literal_eval

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

metadata = pd.read_csv(r"path_to_movies_metadata.csv", low_memory=False)
credits = pd.read_csv(r"path_to_credits.csv", low_memory=False)
keywords = pd.read_csv(r"path_to_keywords.csv", low_memory=False)

metadata = metadata.iloc[0:10000, :]

keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
metadata['id'] = metadata['id'].astype('int')

metadata = metadata.merge(credits, on='id')
metadata = metadata.merge(keywords, on='id')

properties = ['cast', 'crew', 'keywords', 'genres']
for key in properties:
    metadata[key] = metadata[key].apply(literal_eval)


def get_director(x):
    for i in x:
        if i['job'] != 'Director':
            return np.nan
        return i["name"]


def get_elements(arg1):
    if isinstance(arg1, list) is True:
        elements = []
        for i in arg1:
            elements.append(i['name'])
        return elements if (len(elements) < 4) else elements[:4]


metadata['director'] = metadata['crew'].apply(get_director)

properties = ['cast', 'keywords', 'genres']
for key in properties:
    metadata[key] = metadata[key].apply(get_elements)


def clean_dataset(arg1):
    if isinstance(arg1, list) is True:
        return [str.lower(j.replace(" ", "")) for j in arg1]
    return str.lower(arg1.replace(" ", "")) if (isinstance(arg1, str)) else ''


properties = ['cast', 'keywords', 'director', 'genres']
for key in properties:
    metadata[key] = metadata[key].apply(clean_dataset)


def create_context(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])


metadata['soup'] = metadata.apply(create_context, axis=1)


def make_recommendation(searchTerms, metadata=metadata):
    new_row = metadata.iloc[-1, :].copy()
    new_row.iloc[-1] = " ".join(searchTerms)
    metadata = metadata.append(new_row)

    count_vectorizer = CountVectorizer(stop_words='english')
    matrix = count_vectorizer.fit_transform(metadata['soup'])

    cosine_matrix = cosine_similarity(matrix, matrix)
    sim_val = list(enumerate(cosine_matrix[-1, :]))
    sim_val = sorted(sim_val, key=lambda i: i[1], reverse=True)
    print(sim_val[:3])

    add_recommendation = []
    for j in range(1, 4):
        indx = sim_val[j][0]
        add_recommendation.append(
            [metadata['title'].iloc[indx], metadata['keywords'].iloc[indx], metadata['cast'].iloc[indx],
             metadata['genres'].iloc[indx]])

    return add_recommendation
