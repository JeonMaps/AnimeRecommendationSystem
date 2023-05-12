import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import streamlit as st

anime = pd.read_csv('anime.csv')

tf_idf = TfidfVectorizer(lowercase=True, stop_words='english')
anime['genre'] = anime['genre'].fillna('')
tf_idf_matrix = tf_idf.fit_transform(anime['genre'])

cosine_sim = linear_kernel(tf_idf_matrix, tf_idf_matrix)
indices = pd.Series(anime.index, index=anime['name'])
indices = indices.drop_duplicates()


def recommendations(name, cosine_sim=cosine_sim):
    title_index = indices[name]
    similarity_scores = list(enumerate(cosine_sim[title_index]))
    #similarity_scores = list(enumerate(cosine_sim[indices[name]]))
    similarity_scores = sorted(
        similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:21]
    anime_indices = [i[0] for i in similarity_scores]
    return anime.loc[anime_indices, 'name':'rating']


count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(anime['genre'])
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
cosine_sim3 = linear_kernel(count_matrix, count_matrix)

st.title('Anime Recommendation System')

all_titles = anime['name'].unique()
title_input = st.selectbox('Select an anime title', all_titles)

if st.button('Recommend'):
    st.write(recommendations(title_input,cosine_sim2))
    


