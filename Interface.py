import pandas as pd
import streamlit as st
from RecommendationAlgorithm import make_recommendation
import base64


def set_bg_hack(main_bg):
    main_bg_ext = "image.png"

    st.markdown(
        f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover
         }}
         </style>
         """,
        unsafe_allow_html=True
    )


set_bg_hack('image.png')

st.title('Movie Recommender System')
addFeature = []

with st.form("Choose features"):
    genres_val = st.text_input(
        "Type the genre for movie!(type 'skip' to skip the option)")
    genres = " ".join(["".join(i.split()) for i in genres_val.lower().split(',')])
    if genres != 'skip':
        addFeature.append(genres)

    directors_val = st.text_input(
        "Type some directors!(type 'skip' to skip the option)")
    directors = " ".join(["".join(n.split()) for n in directors_val.lower().split(',')])
    if directors != 'skip':
        addFeature.append(directors)

    actors_val = st.text_input(
        "Type some actors!(type 'skip' to skip the option)")
    actors = " ".join(["".join(n.split()) for n in actors_val.lower().split(',')])
    if actors != 'skip':
        addFeature.append(actors)

    keywords_val = st.text_input(
        "Type some keywords!(type 'skip' to skip the option)")
    keywords = " ".join(["".join(n.split()) for n in keywords_val.lower().split(',')])
    if keywords != 'skip':
        addFeature.append(keywords)
        
    submitted = st.form_submit_button("Find similar movies")
    if submitted:
        st.write(pd.DataFrame(make_recommendation(addFeature), columns=["Title", "Keywords","Cast","Directors"]))
