import re 
from scipy import spatial
import numpy as np 
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
#from sentence_transformers import SentenceTransformer
    
def preprocess(phrase):
    
    """ Given a phrase function 
    returns the processed phrase """
    
    phrase = re.sub(r'\n' , '.' , phrase)
    phrase = re.sub(r'\.\.+', ' .', phrase) # substitutes multiple fullstop to a single fullstop 
    phrase = phrase.lower()     # lower
    phrase = re.sub(r"the media could not be loaded", " ", phrase) # some reviews have this extra lines
    
    
    #decontractions 
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    phrase = re.sub(r" n ", " and ", phrase)
    phrase = re.sub(r" ?phn ?", " ", phrase)
    phrase = re.sub(r" ?phone ?", " ", phrase)
    phrase = re.sub(r" ?mobile ?", " ", phrase)
    phrase = re.sub(r" ?smartphone ?", " ", phrase)
    
    lis = list(filter(lambda x: x != ' ' or x != '', phrase.split('.')))

    final_to_encode = []
    for l in lis:
        final_to_encode.extend(list(filter(lambda x: x != ' ' or x != '', l.split(','))))
    
    final_to_encode2 = []    
    for l in final_to_encode:
        final_to_encode2.extend(list(filter(lambda x: x != ' ' or x != '', l.split(' and '))))
    
    final_to_encode3 = []
    for l in final_to_encode2:
        final_to_encode3.extend(list(filter(lambda x: x != ' ' or x != '', l.split(' but '))))

    return final_to_encode3
    

def aspect_sentiment(sentence_list , model ,aspects,  encoded_aspects , aux_sentences ):    
    results_tup = []
    
    encoded_list = model.encode(sentence_list, show_progress_bar  = False)
    for i in range(len(encoded_list)):
        results = []
        for idx, aspect in enumerate(aspects):

            result = 1 - spatial.distance.cosine(encoded_aspects[idx], encoded_list[i])

            results.append(result)
        key = np.array(aspects)[np.argsort(results)[::-1]][0] 
        value = np.sort(results)[::-1][0]
        results_tup.append((key, value, sentence_list[i]))

    final_dict = {}
    for idx , r in enumerate(results_tup):
        aspect, cos_score ,sentence = r
        if len(sentence.strip().split()) > 1 and cos_score >= 0.32:
            compound_score = sid.polarity_scores(sentence)['compound']
            
            if aspect not in aux_sentences: 
                rat = 1 - spatial.distance.cosine(model.encode(f'{aspect} is good',  show_progress_bar  = False), 
                                                  model.encode(sentence,  show_progress_bar  = False) )
                rat2 = 1 - spatial.distance.cosine(model.encode(f'{aspect} is bad',  show_progress_bar  = False),
                                                   model.encode(sentence,  show_progress_bar  = False) )
                if rat>rat2:
                    sentiment = 'positive'

                else:
                    sentiment = 'negative'

            else:
                rat = 1 - spatial.distance.cosine(model.encode(aux_sentences.get(aspect)[0] , show_progress_bar = False),
                                                  model.encode(sentence, show_progress_bar  = False) )
                rat2 = 1 - spatial.distance.cosine(model.encode(aux_sentences.get(aspect)[1] , show_progress_bar  = False), 
                                                   model.encode(sentence, show_progress_bar = False) )


                if rat>rat2:
                    sentiment = 'positive'
                else:
                    sentiment = 'negative'

            if aspect in final_dict:
                final_dict[aspect + "_" + str(idx)] = sentiment
            else:
                final_dict[aspect] = sentiment
    
    no_aspect = 'No aspects were found'
    return final_dict if final_dict != {} else no_aspect  