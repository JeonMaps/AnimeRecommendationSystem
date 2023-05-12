import streamlit as st
import pandas as pd
import joblib
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split

anime = pd.read_csv('anime.csv')

model = joblib.load('anime_model.pkl')

def recommend_animes(model, liked_anime_ids, n_recommendations):
    pseudo_user_id = 'pseudo_user'
    all_anime_ids = anime['anime_id'].unique()

    predictions = [(anime_id, model.predict(pseudo_user_id, anime_id).est) for anime_id in all_anime_ids if anime_id not in liked_anime_ids]

    predictions.sort(key=lambda x: x[1], reverse=True)

    top_n = [anime_id for anime_id, _ in predictions[:n_recommendations]]

    return top_n

def filter_by_genre(recommended_anime_ids, liked_anime_ids):
    liked_genres = set()
    for genres in anime.loc[anime['anime_id'].isin(liked_anime_ids), 'genre']:
        liked_genres.update(genres.split(', '))
    print(f"Liked genres: {liked_genres}")

    filtered_anime_ids = []
    for id in recommended_anime_ids:
        anime_genres = anime.loc[anime['anime_id'] == id, 'genre'].item().split(', ')
        print(f"Anime ID {id} genres: {anime_genres}")
        if any(genre in liked_genres for genre in anime_genres):
            filtered_anime_ids.append(id)
    
    return filtered_anime_ids


anime_dict = pd.Series(anime.anime_id.values,index=anime.name).to_dict()

anime_list = list(anime.name.values)

selected_anime_1 = st.selectbox('Select first anime', anime_list)
selected_anime_2 = st.selectbox('Select second anime', anime_list)
selected_anime_3 = st.selectbox('Select third anime', anime_list)

if st.button('Predict'):
    anime_ids = [anime_dict[selected_anime_1], anime_dict[selected_anime_2], anime_dict[selected_anime_3]]
    
    recommended_anime_ids = recommend_animes(model, anime_ids, n_recommendations=10)

    recommended_anime_ids = filter_by_genre(recommended_anime_ids, anime_ids)
    
    recommended_anime_names = anime[anime['anime_id'].isin(recommended_anime_ids)]['name'].tolist()
    
    st.table(recommended_anime_names)
