# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd 
import seaborn as sns 
import numpy as np 
import matplotlib.pyplot as plt
import requests
import pandas as pd
import io, os
import nltk
from io import BytesIO
import pickle
import requests
from scipy import spatial
import re    
import sys

from preprocess_functions import *

nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
from sentence_transformers import SentenceTransformer
    

st.title('Summarizing smartphone reviews')


@st.cache()
def load_data():
    link1 = 'https://github.com/taaresh7/Summarizing-phone-reviews/blob/main/aspects_.pickle?raw=true'
    file1 = BytesIO(requests.get(link1).content)
    aspects = pickle.load(file1)

    link2 = 'https://github.com/taaresh7/Summarizing-phone-reviews/blob/main/aux_sentences_.pickle?raw=true'
    file2 = BytesIO(requests.get(link2).content)
    aux_sentences = pickle.load(file2)
    
    link3 = 'https://raw.githubusercontent.com/taaresh7/Summarizing-phone-reviews/main/sentiment_df2.csv'
    df = pd.read_csv(link3)
    return aspects, aux_sentences, df
    
data_load_state = st.text('Loading data...')
aspects, aux_sentences, df = load_data()    
data_load_state.text('Loading data...done!')

@st.cache(allow_output_mutation=True)
def load_model():
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return model

data_model_state = st.text('Loading model...')
model = load_model()    
data_model_state.text('Loading model...done!')


text = st.text_input('Enter review')

encoded_aspects = model.encode(aspects)


text_list = preprocess(text)
aspect_sent = aspect_sentiment(text_list , model ,aspects, 
                                    encoded_aspects , aux_sentences )

st.write('Aspects' , aspect_sent)


df2 = df.copy()
import ast
df2['sentiment'] = df['sentiment'].apply(eval)
option = st.selectbox(
      'Choose a phone to analyse?',
      tuple(df2.Title.values))

st.write('You selected:', option)

def num_aspects(option):
    option2 = st.selectbox('number of top aspects to show', (5,10, 15))
    st.write('You selected:', option2)

    dict_ = df2[df2['Title'] == option]['sentiment'].values[0]

    for idx , (k, v) in enumerate(dict_.items()):
        if idx <= option2:
            st.write(k, v)
        

num_aspects(option)