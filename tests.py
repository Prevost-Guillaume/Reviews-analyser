import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
from sklearn.decomposition import PCA


### GET DATA
print('Load data')
# Load models
transformer = SentenceTransformer("models/bert-base-nli-mean-tokens")
polarity_classifier = load_sklearn_polarity_classifier(file='models/polarity_classifier_sklearn_imdb.sav')


#### Load comments
##print('Scrapping comments')
##url = 'https://www.tripadvisor.in/Hotel_Review-g187147-d228694-Reviews-Hotel_Malte_Astotel-Paris_Ile_de_France.html'
##url = 'https://www.tripadvisor.in/Hotel_Review-g187147-d239642-Reviews-Hotel_des_Batignolles-Paris_Ile_de_France.html'
##comments = scrape_tripadvisor(url, n=1000)
##text = '[SEP]'.join(comments)
##
##
##
##
##
##### ENCODDING ###
### Encode text
##print('\nencode text')
##text = text.lower()
##text_embedding = transformer.encode(text)
##
### Encode sentences
##print('encode sentences')
##sentences = sent_tokenize(text.replace('[sep]','.'))
##sentences_embeddings = transformer.encode(sentences)
##
### Encode comments
##print('encode comments')
##comments = text.split('[sep]')
##comments_embedding = transformer.encode(comments)
##
##
##
##### BUILD DATAFRAMES ###
##data_embedding = [{'Embedding':comments_embedding[i], 'Comments':comments[i]} for i in range(len(comments_embedding))]
##df_comments = pd.DataFrame(data_embedding)
##
##data_embedding = [{'Embedding':sentences_embeddings[i], 'Sentences':sentences[i]} for i in range(len(sentences_embeddings))]
##df_sentences = pd.DataFrame(data_embedding)
##
##
##### TOPICS EXTRACTION ###
##print('\nGet comments topics')
##lda_topics, lda_model, corpus = topic_extraction(df_comments['Comments'], num_topics=5, tags=['NOUN'])#, 'ADJ'])
##topics_comments = filter_topics_keywords_naive(lda_topics)
##
##df_topic = classify_sentences(ldamodel=lda_model, corpus=corpus, texts=df_comments['Comments'])
##df_topic = df_topic.reset_index()
##df_topic.columns = ['Document_No', 'Topic', 'Topic_score', 'Keywords', 'Comments']
##df_topic = df_topic[['Comments', 'Topic', 'Topic_score']]
##df_comments = pd.concat([df_comments, df_topic[['Topic', 'Topic_score']]], axis=1)
##
##
##print('\nGet sentences topics')
##lda_topics, lda_model, corpus = topic_extraction(df_sentences['Sentences'], num_topics=5, tags=['NOUN'])#, 'ADJ'])
##topics_sentences = filter_topics_keywords_naive(lda_topics)
##
##df_topic = classify_sentences(ldamodel=lda_model, corpus=corpus, texts=df_sentences['Sentences'])
##df_topic = df_topic.reset_index()
##df_topic.columns = ['Document_No', 'Topic', 'Topic_score', 'Keywords', 'Comments']
##df_topic = df_topic[['Comments', 'Topic', 'Topic_score']]
##
##df_sentences = pd.concat([df_sentences, df_topic[['Topic', 'Topic_score']]], axis=1)
##
##
##### POLARITY ###
##print('\nGet comments polarity')
##df_comments['Polarity'] = getPolarity([[float(j) for j in i] for i in list(df_comments['Embedding'])], polarity_classifier, mode='proba')
##
##print('\nGet sentences polarity')
##df_sentences['Polarity'] = getPolarity([[float(j) for j in i] for i in list(df_sentences['Embedding'])], polarity_classifier, mode='proba')
##
##
##df_comments.to_pickle('df_comments.p')
##df_sentences.to_pickle('df_sentences.p')
##pickle.dump(topics_sentences, open('topics_sentences.p','wb'))
##pickle.dump(topics_comments, open('topics_comments.p','wb'))
##print('stored')  
##









print('load all this shit')
df_comments = pd.read_pickle('df_comments.p')
df_sentences = pd.read_pickle('df_sentences.p')
topics_sentences = pickle.load(open('topics_sentences.p','rb'))
topics_comments = pickle.load(open('topics_comments.p','rb'))

print(df_comments.columns)
print(df_sentences.columns)
print(topics_comments)
print(topics_sentences)



##df_sentences.columns = ['Embedding', 'Sentences', 'Topic', 'Topic_score', 'Topic_','Topic_score_', 'Polarity']
##df_sentences = df_sentences.drop(['Topic_','Topic_score_'], axis=1)

plot_radar_topics(df_sentences, topics_sentences, mode='importance')
plot_radar_topics(df_comments, topics_comments, mode='importance')
show_comments_topics(df_sentences, df_mode='Sentences')
show_comments_polarity(df_sentences, df_mode='Sentences')
plot_topic_polarity(df_sentences, topics_sentences, mode='a')
plot_topic_polarity(df_comments, topics_comments, mode='a')
polarity_distrib(df_sentences['Polarity'], df_sentences['Sentences'])
pie_chart(df_sentences['Polarity'])
pie_chart(df_comments['Polarity'])

input('done')


### SUMMARY ###
print('\nsummarize text')
text = '[SEP]'.join(list(df_comments['Comments']))
summary = ExtractiveSummaryDouble(text, 4,
                                  text_embedding=text_embedding,
                                  sentences_embeddings=sentences_embeddings,
                                  metric='cosine')
print('summary 1 : ',summary)


text = '[SEP]'.join(list(df_sentences['Sentences']))
summary = ExtractiveSummaryDouble(text, 4,
                                  text_embedding=text_embedding,
                                  sentences_embeddings=sentences_embeddings,
                                  metric='cosine')
print('summary 2 : ',summary)











