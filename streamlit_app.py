import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import text_analysis as ta
import en_core_web_sm
import spacy_streamlit



st.title('Youtube Video Analysis\n', )
st.subheader("by Aleksander Petrov")

display = Image.open('images/youtube logo.jpeg')
display = np.array(display)
st.image(display)

# Reading .csv file for all titles and comments
titles_and_comments = pd.read_csv('data/titles_and_comments.csv', lineterminator='\n', index_col=0)
titles = titles_and_comments['Video title'].dropna().unique()

STOPWORDS = stopwords.words('english')
additional_stopwords = ['feat', 'ft', 'ep']

# Sidebar
option = st.sidebar.selectbox('Navigation',
    ["Home",
     "Word Cloud", 
     "N-Gram Analysis", 
     "Parts of Speech Analysis", 
     "Named Entity Recognition"
    ])

# st.set_option('deprecation.showfileUploaderEncoding', False)

if option == 'Home':
	st.write(
			"""
				## Project Description
				### This is a complete text analysis tool developed by Aleksandr Petrov. It's built in with multiple features which can be accessed from the left side bar.
			"""
		)

elif option == 'Word Cloud':
    st.header('Word Cloud')
    wordcloud_option = st.sidebar.selectbox('Data', ['Titles', 'Comments'])
    wordcloud_t = st.sidebar.multiselect('Titles', titles, default=titles)
    if wordcloud_option == 'Titles':
        _text = '; '.join(titles)
    
    if wordcloud_option == 'Comments':
        st.write(""" # Comments are taken from YouTube videos with selected title on the left""")
        comments = titles_and_comments[titles_and_comments['Video title'].isin(wordcloud_t)]['Comments'].dropna().unique()
        _text = '; '.join(comments)

    st.write("""
    ## The main stop words are taken from nltk.corpus.stopwords
    ## But you can write down your own
    """)
    users_stopword = st.text_input("Add your own stop word separated by commas")
    _users_stopwords = users_stopword.split(',')
    _users_stopwords = list(map(lambda x: x.strip(), _users_stopwords))
    _users_stopwords = [i for i in _users_stopwords if i]
    if 'add_sw' not in st.session_state:
        st.session_state['add_sw'] = additional_stopwords
    else:
        st.session_state['add_sw'] += _users_stopwords

    _stopwords = st.multiselect('Stop words', st.session_state['add_sw'], default=st.session_state['add_sw'])
    
    st.write('You can upload your own mask for Word Cloud image or choose one of the default ones.')
    mask1 = Image.open('images/youtube mask.png')
    mask2 = Image.open('images/cloud mask.png')
    uploaded_mask = st.file_uploader('Upload a mask', 'png')
    mask = st.selectbox('Mask', ['None', 'YouTube mask', 'Cloud mask', 'Uploaded mask'])
    if mask == 'Uploaded mask' and uploaded_mask:
        _mask = uploaded_mask
    elif mask == 'YouTube mask':
        _mask = mask1
    elif mask == 'Cloud mask':
        _mask = mask2
    else:
        _mask = None
    ta.create_wordcloud(_text, STOPWORDS + _stopwords, _mask)


elif option == 'N-Gram Analysis':
    st.header('N-Gram Analysis')
    ngram_option = st.sidebar.selectbox('Data', ['Titles', 'Comments'])
    ngram_t = st.sidebar.multiselect('Titles', titles, default=titles)
    if ngram_option == 'Titles':
        text_list = titles
    
    if ngram_option == 'Comments':
        st.write(""" # Comments are taken from YouTube videos with selected title on the left""")
        comments = titles_and_comments[titles_and_comments['Video title'].isin(ngram_t)]['Comments'].dropna().unique()
        text_list = comments

    st.write("""
    ## The main stop words are taken from nltk.corpus.stopwords
    ## But you can write down your own
    """)
    users_stopword = st.text_input("Add your own stop word separated by commas")
    _users_stopwords = users_stopword.split(',')
    _users_stopwords = list(map(lambda x: x.strip(), _users_stopwords))
    _users_stopwords = [i for i in _users_stopwords if i]
    if 'add_sw_ng' not in st.session_state:
        st.session_state['add_sw_ng'] = additional_stopwords
    else:
        st.session_state['add_sw_ng'] += _users_stopwords

    _stopwords = st.multiselect('Stop words', st.session_state['add_sw_ng'], default=st.session_state['add_sw_ng'])
    col1, col2 = st.columns(2) 
    col3, col4 = st.columns([3, 1])
    with col1:
        n_gram = st.slider('N-gram', 1, 5, 2)
    with col2:
        top_k = st.slider('Top K', 1, 30, 15)
    with col3:
        height = st.slider('Plot height', 300, 800, 500)
    with col4:
        scale = st.selectbox('X axis scale', ['Normal', 'Logarithmic'])
        _scale = scale == 'Logarithmic'
    
    ta.plot_ngrams(text_list, STOPWORDS + _stopwords, n_gram, top_k, height, _scale)

elif option == 'Parts of Speech Analysis': 
    st.header('Parts of Speech Analysis')
    ngram_option = st.sidebar.selectbox('Data', ['Titles', 'Comments'])
    ngram_t = st.sidebar.multiselect('Titles', titles, default=titles)
    col1, col2 = st.columns([2, 1])
    if ngram_option == 'Titles':
        text_list = titles
        with col1:
            st.write(""" ## Choose number of title""")
    
    if ngram_option == 'Comments':
        st.write(""" # Comments are taken from YouTube videos with selected title on the left""")
        comments = titles_and_comments[titles_and_comments['Video title'].isin(ngram_t)]['Comments'].dropna().unique()
        text_list = comments
        with col1:
            st.write(""" ## Choose number of comment""")

    # format %d %e %f %g %i %u
    max_val = len(text_list)
    with col2:
        number = st.number_input('Number', 1, max_val, max_val//2, step=1, format='%d')
    
    if st.button("Show POS Tags"):
        output = ta.pos_tagger(text_list[number-1])
        st.markdown("The POS Tags for this sentence are: ")
        st.markdown(output, unsafe_allow_html=True)

        st.markdown("### Penn-Treebank Tagset")
        st.markdown("The tags can be referenced from here:")
		
        # Show image
        display_pos = Image.open('images/Penn.png')
        # display_pos = np.array(display_pos)
        st.image(display_pos)

elif option == 'Named Entity Recognition':
    st.header('Named Entity Recognition')
    st.subheader("Enter the statement that you want to analyze")

    # st.markdown("**Random Sentence:** A Few Good Men is a 1992 American legal drama film set in Boston directed by Rob Reiner and starring Tom Cruise, Jack Nicholson, and Demi Moore. The film revolves around the court-martial of two U.S. Marines charged with the murder of a fellow Marine and the tribulations of their lawyers as they prepare a case to defend their clients.")
	# text_input = st.text_area("Enter sentence")

    ner_option = st.sidebar.selectbox('Data', ['Titles', 'Comments'])
    ner_t = st.sidebar.multiselect('Titles', titles, default=titles)
    col1, col2 = st.columns([2, 1])
    if ner_option == 'Titles':
        text_list = titles
        with col1:
            st.write(""" ## Choose number of title""")
    
    if ner_option == 'Comments':
        st.write(""" # Comments are taken from YouTube videos with selected title on the left""")
        comments = titles_and_comments[titles_and_comments['Video title'].isin(ner_t)]['Comments'].dropna().unique()
        text_list = comments
        with col1:
            st.write(""" ## Choose number of comment""")

    max_val = len(text_list)
    with col2:
        number = st.number_input('Number', 1, max_val, 153, step=1, format='%d')
    
    ner = en_core_web_sm.load()
    doc = ner(text_list[number-1])

	# Display 
    spacy_streamlit.visualize_ner(doc, labels=ner.get_pipe('ner').labels)
    spacy_streamlit.visualize_tokens(doc)
    spacy_streamlit.visualize_parser(doc)