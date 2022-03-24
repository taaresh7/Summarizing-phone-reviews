# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 14:26:24 2022

@author: MT
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 15:09:56 2021

@author: MT
"""

import streamlit as st
import pandas as pd 
import seaborn as sns 
import numpy as np 
import matplotlib.pyplot as plt
import requests
import pandas as pd
import io, os
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
from sentence_transformers import SentenceTransformer
    

st.title('Summarizing smartphone reviews')


from io import BytesIO
import pickle
import requests
    
@st.cache
def load_data():
    link1 = 'https://github.com/taaresh7/Summarizing-phone-reviews/blob/main/aspects.pickle?raw=true'
    file1 = BytesIO(requests.get(link1).content)
    aspects = pickle.load(file1)

    link2 = 'https://github.com/taaresh7/Summarizing-phone-reviews/blob/main/aux_sentences.pickle?raw=true'
    file2 = BytesIO(requests.get(link2).content)
    aux_sentences = pickle.load(file2)
    
    return aspects, aux_sentences
    
data_load_state = st.text('Loading data...')
aspects, aux_sentences = load_data()    
data_load_state.text('Loading data...done!')

@st.cache
def load_model():
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return model

data_model_state = st.text('Loading model...')
model = load_model()    
data_model_state.text('Loading model...done!')


# details_df = pd.read_csv('C:/Users/MT/Desktop/phone_details_df.csv')
#st.subheader('Raw data')
#st.write(df)

# df2= df.loc[df.cluster != -1 , :]
# all_aspects = ','.join(list(df2['aspects'].values)).split(',')
# all_aspects = list(set(map(str.strip , all_aspects)))
# #print(all_aspects)

# from wordcloud import WordCloud
# import matplotlib.pyplot as plt

# # # Create some sample text
# # text = ' '.join(all_aspects)
# # # Create and generate a word cloud image:
# # wordcloud = WordCloud(background_color="white", width=800, height=400).generate(text)

# # fig, ax = plt.subplots(figsize = (20,10))
# # # Display the generated image:
# # ax.imshow(wordcloud, interpolation='bilinear')
# # ax.axis("off")
# # plt.show()
# # st.pyplot(fig)


# option = st.selectbox(
#       'Choose a phone to analyse?',
#       tuple(details_df.Title.values))

# st.write('You selected:', option)

text = st.text_input('Enter review')
encoded_aspects = model.encode(aspects)

from scipy import spatial

#print(p)
p = 'PHONE GETS HEATED WHILE CHARGING BUT PERFORMACE IS GOOD'
lis = list(filter(lambda x: x != ' ' and x != '', text.lower().split('.')))

final_to_encode = []
for l in lis:
    final_to_encode.extend(list(filter(lambda x: x != ' ' or x != '', l.split(','))))
final_to_encode2 = []    
for l in final_to_encode:
    final_to_encode2.extend(list(filter(lambda x: x != ' ' or x != '', l.split(' and '))))
final_to_encode3 = []
for l in final_to_encode2:
    final_to_encode3.extend(list(filter(lambda x: x != ' ' or x != '', l.split(' but '))))

    
encoded_list = model.encode(final_to_encode3)
    
print(final_to_encode3)
print('*'*80)

results_tup = []
for i in range(len(encoded_list)):
    results = []
    for idx, aspect in enumerate(aspects):

        result = 1 - spatial.distance.cosine(encoded_aspects[idx], encoded_list[i])

        results.append(result)
    key = np.array(aspects)[np.argsort(results)[::-1]][0] 
    value = np.sort(results)[::-1][0]
    results_tup.append((key, value, final_to_encode3[i]))
    

#results_tup
final_dict = {}
for idx , r in enumerate(results_tup):
    aspect, cos_score ,sentence = r
    if len(sentence.strip().split()) > 1:
        if cos_score >= 0.35:
            
            compound_score = sid.polarity_scores(sentence)['compound']
            #print(sentence,'*',aspect,'*',cos_score,'*' , compound_score)
            #print('\n')
            #
            if aspect not in aux_sentences: 
                rat = 1 - spatial.distance.cosine(model.encode(f'{aspect} is good',  show_progress_bar  = False), 
                                                  model.encode(sentence,  show_progress_bar  = False) )
                rat2 = 1 - spatial.distance.cosine(model.encode(f'{aspect} is bad',  show_progress_bar  = False),
                                                   model.encode(sentence,  show_progress_bar  = False) )
                #print(rat , rat2)
                if rat>rat2:
                    sentiment = 'pos'
                    
                else:
                    sentiment = 'neg'
                #print('*'*80)
            else:
                #print('using sentences')
                rat = 1 - spatial.distance.cosine(model.encode(aux_sentences.get(aspect)[0] , show_progress_bar = False),
                                                  model.encode(sentence, show_progress_bar  = False) )
                rat2 = 1 - spatial.distance.cosine(model.encode(aux_sentences.get(aspect)[1] , show_progress_bar  = False), 
                                                   model.encode(sentence, show_progress_bar = False) )
                #print(rat , rat2)
                if rat>rat2:
                    sentiment = 'pos'
                else:
                    sentiment = 'neg'
                #print('*'*80)            
            if aspect in final_dict:
                final_dict[aspect + "_" + str(idx)] = sentiment
            else:
                final_dict[aspect] = sentiment
                
st.write('Aspects' , final_dict)



#pip install pipreqs
