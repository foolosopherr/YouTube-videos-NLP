# File containing all the analysis functions for the streamlit app

# Standard Libraries
import re 
import string 
import numpy as np
from collections import Counter

# Text Processing Library 
import nltk 
# from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from wordcloud import WordCloud
import streamlit as st
import warnings
warnings.filterwarnings(action='ignore')


# Data Visualisation 
import matplotlib.pyplot as plt 
import plotly.express as px


# Constants 
# STOPWORDS = stopwords.words('english')
# STOPWORDS += ['said']


# Text cleaning function 
def clean_text(text, STOPWORDS):
    '''
        Function which returns a clean text 
    '''    
    # Lower case 
    text = text.lower()
    
    # Remove numbers
    text = re.sub(r'\d', '', text)
    
    # Replace \n and \t functions 
    text = re.sub(r'\n', '', text)
    text = text.strip()
    
    # Remove punctuations
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove Stopwords and Lemmatise the data
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text.split() if word not in STOPWORDS]
    text = ' '.join(text)
    
    return text

# Create a word cloud function 
def create_wordcloud(text, STOPWORDS, image = None):
    '''
    Pass a string to the function and output a word cloud
    
    ARGS 
    text: The text for wordcloud
    image_path (optional): The image mask with a white background (default None)
    
    '''
    
    # st.write('Creating Word Cloud..')
    
    text = clean_text(text, STOPWORDS)
    
    if not image:
        
        # Generate the word cloud
        wordcloud = WordCloud(width = 1200, height = 1200, 
                    background_color ='white', 
                    stopwords = STOPWORDS, 
                    min_font_size = 8).generate(text) 
    
    else:
        mask = np.array(image)
        wordcloud = WordCloud(width = 1200, height = 1200, 
                    background_color ='white', 
                    stopwords = STOPWORDS,
                    mask=mask,
                    min_font_size = 8).generate(text) 
    
    # plot the WordCloud image 
    fig, ax = plt.subplots(figsize=(20,20), facecolor=None)    
    ax.imshow(wordcloud, interpolation = 'nearest')
    plt.axis("off") 
    plt.tight_layout(pad = 0) 

    # plt.show()
    st.pyplot(fig)
    


# Function to plot the ngrams based on n and top k value
def plot_ngrams(text_list, STOPWORDS, n=2, topk=15, height=500, log_x=False):
    '''
    Function to plot the most commonly occuring n-grams in bar plots 
    
    ARGS
        text: data to be enterred
        n: n-gram parameters
        topk: the top k phrases to be displayed

    '''

    all_ngram_phrases = []
    for text in text_list:
        _text = clean_text(text, STOPWORDS)
        tokens = _text.split()

        # get the ngrams 
        ngram_phrases = ngrams(tokens, n)
        all_ngram_phrases += ngram_phrases


    # Get the most common ones 
    most_common = Counter(all_ngram_phrases).most_common(topk)

    # Make word and count lists 
    words, counts = [], []
    for phrase, count in most_common:
        word = ' '.join(phrase)
        words.append(word)
        counts.append(count)
    
    fig = px.histogram(x=counts, y=words, labels={'x':'N-gram frequences', 'y':'N-grams in the text'}, 
                       title=f"Most Common {n}-grams in the text", log_x=log_x, height=height)
    
    fig.update_layout(yaxis={'categoryorder':'total ascending'})

    st.plotly_chart(fig)


# Function to return POS tags of a sentence 
def pos_tagger(s):
    
    # Define the tag dictionary 
    output = ''
    
    # Remove punctuations
    s = s.translate(str.maketrans('', '', string.punctuation))
    
    tagged_sentence = nltk.pos_tag(nltk.word_tokenize(s))
    for tag in tagged_sentence:
        out = tag[0] + ' ---> ' + tag[1] + '<br>'
        output += out

    return output


