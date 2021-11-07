import streamlit as st

import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize
from summarization import ExtractiveSummarySimilarity, ExtractiveSummaryClusturing, ExtractiveSummaryDouble
from sentence_transformers import SentenceTransformer

from polarization import load_sklearn_polarity_classifier, getPolarity
from topic_extraction import *
from vizualisation import *
from search_engine import search
from web_scrapping import *

from scipy.spatial import distance

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots





def get_significance_score(df_topic, mode):
    df = df_topic.copy()
    try:
        df['len'] = df[mode].apply(lambda x:len(x))
        df['max_len'] = max(df['len'])
        df['len_score'] = df['len']/df['max_len']
        df['Polarity_score'] = abs(df['Polarity'])
        df['significance_score'] = df['len_score']*df['Topic_score']*df['Polarity_score']
        return df['significance_score']
    except:
        return 0





# Use the full page instead of a narrow central column
st.set_page_config(layout="wide")

st.title('Reviews analyzer')
st.subheader("Analyze tripAdvisor reviews of your hotel")


st.sidebar.title('Parameters')
n = st.sidebar.number_input('Number of comments to scrape :',min_value=10, value=200)
MODE = st.sidebar.radio('Work on :', ['Sentences','Comments'])


url = ''#'https://www.tripadvisor.in/Hotel_Review-g187147-d239642-Reviews-Hotel_des_Batignolles-Paris_Ile_de_France.html'
url = st.text_input('Your hotel url : ','https://www.tripadvisor.in/Hotel_Review-g187147-d239642-Reviews-Hotel_des_Batignolles-Paris_Ile_de_France.html')



if url != '':
    # Load models
    transformer = SentenceTransformer("models/bert-base-nli-mean-tokens")
    polarity_classifier = load_sklearn_polarity_classifier(file='models/polarity_classifier_sklearn_imdb.sav')


    ### GET DATA
    st.write('Load data from '+url)
    progress_text = st.empty()

    ## Load comments
    progress_bar_zone = st.empty()
    progress_bar = progress_bar_zone.progress(0)
    progress_text.write('Scrapping reviews')
    comments = scrape_tripadvisor_st(url, progress_bar, n=n)
    progress_bar_zone.write("")
    text = '[SEP]'.join(comments)


    ### ENCODDING ###
    # Encode text
    progress_text.write('encode text')
    text = text.lower()
    text_embedding = transformer.encode(text)

    # Encode sentences
    progress_text.write('encode sentences')
    sentences = sent_tokenize(text.replace('[sep]','.'))
    sentences_embeddings = transformer.encode(sentences)

    # Encode comments
    progress_text.write('encode comments')
    comments = text.split('[sep]')
    comments_embedding = transformer.encode(comments)


    ### BUILD DATAFRAMES ###
    data_embedding = [{'Embedding':comments_embedding[i], 'Comments':comments[i]} for i in range(len(comments_embedding))]
    df_comments = pd.DataFrame(data_embedding)

    data_embedding = [{'Embedding':sentences_embeddings[i], 'Sentences':sentences[i]} for i in range(len(sentences_embeddings))]
    df_sentences = pd.DataFrame(data_embedding)


    
    ### TOPICS EXTRACTION ###
    progress_text.write('Get comments topics')
    lda_topics, lda_model, corpus = topic_extraction(df_comments['Comments'], num_topics=5, tags=['NOUN'])#, 'ADJ'])
    topics_comments = filter_topics_keywords_naive(lda_topics)

    df_topic = classify_sentences(ldamodel=lda_model, corpus=corpus, texts=df_comments['Comments'])
    df_topic = df_topic.reset_index()
    df_topic.columns = ['Document_No', 'Topic', 'Topic_score', 'Keywords', 'Comments']
    df_topic = df_topic[['Comments', 'Topic', 'Topic_score']]
    df_comments = pd.concat([df_comments, df_topic[['Topic', 'Topic_score']]], axis=1)


    progress_text.write('Get sentences topics')
    lda_topics, lda_model, corpus = topic_extraction(df_sentences['Sentences'], num_topics=5, tags=['NOUN'])#, 'ADJ'])
    topics_sentences = filter_topics_keywords_naive(lda_topics)

    df_topic = classify_sentences(ldamodel=lda_model, corpus=corpus, texts=df_sentences['Sentences'])
    df_topic = df_topic.reset_index()
    df_topic.columns = ['Document_No', 'Topic', 'Topic_score', 'Keywords', 'Comments']
    df_topic = df_topic[['Comments', 'Topic', 'Topic_score']]

    df_sentences = pd.concat([df_sentences, df_topic[['Topic', 'Topic_score']]], axis=1)


    ### POLARITY ###
    progress_text.write('Get comments polarity')
    df_comments['Polarity'] = getPolarity([[float(j) for j in i] for i in list(df_comments['Embedding'])], polarity_classifier, mode='proba')

    progress_text.write('Get sentences polarity')
    df_sentences['Polarity'] = getPolarity([[float(j) for j in i] for i in list(df_sentences['Embedding'])], polarity_classifier, mode='proba')


    progress_text.write('Download complete')

##    df_comments.to_pickle('df_comments.p')
##    df_sentences.to_pickle('df_sentences.p')
##    pickle.dump(topics_sentences, open('topics_sentences.p','wb'))
##    pickle.dump(topics_comments, open('topics_comments.p','wb'))
##    progress_text.write('stored')  
##
##
##
##
##    df_comments = pd.read_pickle('df_comments.p')
##    df_sentences = pd.read_pickle('df_sentences.p')
##    topics_sentences = pickle.load(open('topics_sentences.p','rb'))
##    topics_comments = pickle.load(open('topics_comments.p','rb'))
    


    ### SEARCH ENGINE ###
    st.header('Search engine')
    exp = st.beta_expander('Search in the comments')

    col1, col2 = exp.beta_columns(2)
    n_search = col1.number_input('Number of results', min_value=1, max_value=len(df_comments['Comments']), value=5)
    col_search = col2.selectbox('Search field', ["Comments", "Sentences"])
    
    entry = exp.text_input('Search','')
    search_results = []
    for i in range(n_search):
        search_results.append(exp.empty())
        
    if entry != '':
        if col_search == "Comments":
            result = search(entry, df_comments['Comments'], df_comments['Embedding'], transformer, n=n_search)
        elif col_search == "Sentences":
            result = search(entry, df_sentences['Sentences'], df_sentences['Embedding'], transformer, n=n_search)
            
        for i,r in enumerate(result):
            search_results[i].write('> '+r)
            


    ### First overview ###
    avg = df_comments['Polarity'].mean()
    st.subheader(f'Mean satisfaction (over 5): {round(5*(avg+1)/2,2)}')
    

    ### TOPICS ###
    st.header("Topics covered")
    if MODE == 'Sentences':
        df = df_sentences
        topics = topics_sentences
    elif MODE == 'Comments':
        df = df_comments
        topics = topics_comments

    

    for i in topics:
        exp = st.beta_expander(f'Topic {i} : {" ".join(topics[i])}')
        df_top = df[df["Topic"] == i]
        df_top['significance_score'] = get_significance_score(df_top, MODE)
        best_sig_comment = df_top[df_top['significance_score'] == max(df_top['significance_score'])].head(1).reset_index()[MODE][0]
        best_pol_comment = df_top[df_top['Polarity'] == max(df_top['Polarity'])].head(1).reset_index()[MODE][0]
        worst_pol_comment = df_top[df_top['Polarity'] == min(df_top['Polarity'])].head(1).reset_index()[MODE][0]
        
        exp.markdown("**Most significant comment**")
        exp.write(best_sig_comment)
        exp.markdown("\n**Most positive comment**")
        exp.write(best_pol_comment)
        exp.markdown("\n**Most negative comment**")
        exp.write(worst_pol_comment)
    


    ### GRAPHICS ###    
    st.header("Visualizations")
    if MODE == 'Sentences':
        st.subheader("Satisfaction repartition")
        st.plotly_chart(pie_chart(df_sentences['Polarity']), use_container_width=True)
        st.plotly_chart(polarity_distrib(df_sentences['Polarity'], df_sentences['Sentences']), use_container_width=True)
        st.plotly_chart(show_comments_polarity(df_sentences, df_mode='Sentences'), use_container_width=True)
        st.subheader("Topics analysis")
        col1, col2 = st.beta_columns(2)
        col1.plotly_chart(plot_radar_topics(df_sentences, topics_sentences, mode='importance'), use_container_width=True)
        col2.plotly_chart(plot_radar_topics(df_sentences, topics_sentences, mode='quality'), use_container_width=True)
        st.plotly_chart(show_comments_topics(df_sentences, df_mode='Sentences'), use_container_width=True)
        st.plotly_chart(plot_topic_polarity(df_sentences, topics_sentences, mode='count'), use_container_width=True)
    elif MODE == 'Comments':
        st.subheader("Satisfaction repartition")
        st.plotly_chart(pie_chart(df_comments['Polarity']), use_container_width=True)
        st.plotly_chart(polarity_distrib(df_comments['Polarity'], df_comments['Comments']), use_container_width=True)
        st.plotly_chart(show_comments_polarity(df_comments, df_mode='Comments'), use_container_width=True)
        st.subheader("Topics analysis")
        col1, col2 = st.beta_columns(2)
        col2.plotly_chart(plot_radar_topics(df_comments, topics_comments, mode='importance'), use_container_width=True)
        col1.plotly_chart(plot_radar_topics(df_comments, topics_comments, mode='quality'), use_container_width=True)
        st.plotly_chart(show_comments_topics(df_comments, df_mode='Comments'), use_container_width=True)
        st.plotly_chart(plot_topic_polarity(df_comments, topics_comments, mode='count'), use_container_width=True)




    ### SUMMARY ###
    st.header('Reviews extractive summarization')
    text = '[SEP]'.join(list(df_sentences['Sentences']))
    summary = ExtractiveSummaryDouble(text, 4,
                                      text_embedding=text_embedding,
                                      sentences_embeddings=sentences_embeddings,
                                      metric='cosine')
    for s in summary:
        st.write(">  "+s)










