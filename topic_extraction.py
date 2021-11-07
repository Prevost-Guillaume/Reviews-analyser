import pandas as pd
import numpy as np
import pickle
import math
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk import FreqDist

import re
import spacy

import gensim
from gensim import corpora

# libraries for visualization
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint

#nltk.download('stopwords')

#############################################################################################################################################
#                                                                                                                                           #
#                                                             TOPICS NUMBER                                                                 #
#                                                                                                                                           #
#############################################################################################################################################
  
def get_intersection(topics):
    inter = 0
    all_words_count = {}
    all_words_sum_score = {}
    
    for i in topics:
        words = dict(topics[i])
        for word in words:
            try:
              all_words_count[word] += 1
            except:
              all_words_count[word] = 1
            try:
              all_words_sum_score[word] += words[word]
            except:
              all_words_sum_score[word] = words[word]
    for word in all_words_count:
        if all_words_count[word] >= 2:
          inter += (all_words_sum_score[word]/all_words_count[word])
    return inter


def find_best_topics_number(texts, tags=['NOUN', 'ADJ']):
    """On commence à 2 et on augmente jusqu'à mminimiser l'intersection pondérée des mots des differents sujets"""
    min_intersect = 1000
    best_n = None
    for i in range(2,10):
        top_topics, lda_model, doc_term_matrix = topic_extraction(texts, num_topics=i, tags=['NOUN', 'ADJ'])
        pprint(top_topics)
        print(get_intersection(top_topics))
        #input()


#############################################################################################################################################
#                                                                                                                                           #
#                                                           TOPICS EXTRACTION                                                               #
#                                                                                                                                           #
#############################################################################################################################################  

# function to plot most frequent terms
def freq_words(x, terms = 30):
  all_words = ' '.join([text for text in x])
  all_words = all_words.split()

  fdist = FreqDist(all_words)
  words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})

  # selecting top 20 most frequent words
  d = words_df.nlargest(columns="count", n = terms) 
  plt.figure(figsize=(20,5))
  ax = sns.barplot(data=d, x= "word", y = "count")
  ax.set(ylabel = 'Count')
  plt.show()

# function to remove stopwords
def remove_stopwords(rev, stop_words):
    rev_new = " ".join([i for i in rev if i not in stop_words])
    return rev_new


def lemmatization(texts, nlp, tags=['NOUN', 'ADJ']): # filter noun and adjective
       output = []
       for sent in texts:
             doc = nlp(" ".join(sent)) 
             output.append([token.lemma_ for token in doc if token.pos_ in tags])
       return output

def topic_extraction(texts, num_topics=5, tags=['NOUN', 'ADJ']):
    """Extract num_topics topics from comments"""

    # Put texts in a panda dataframe
    df = pd.DataFrame()
    df['reviewText'] = texts
    
    # remove short words (length < 3)
    df['reviewText'] = df['reviewText'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))
    df['reviewText'] = df['reviewText'].apply(lambda x: x.replace(',','').replace('.','').replace('!',''))

    # remove stopwords from the text, and make the text lowercase
    stop_words = stopwords.words('english')
    reviews = [remove_stopwords(r.split(), stop_words).lower() for r in df['reviewText']]


    # Tokenize, lemmantize and join reviews
    nlp = spacy.load("models/en_core_web_sm-3.0.0/en_core_web_sm/en_core_web_sm-3.0.0")
    
    tokenized_reviews = pd.Series(reviews).apply(lambda x: x.split())

    reviews_2 = lemmatization(tokenized_reviews, nlp, tags=tags)

    reviews_3 = []
    for i in range(len(reviews_2)):
      reviews_3.append(' '.join(reviews_2[i]))

    df['reviews'] = reviews_3

    #freq_words(df['reviews'], 35)


    ### LDA MODEL
    dictionary = corpora.Dictionary(reviews_2)
    doc_term_matrix = [dictionary.doc2bow(rev) for rev in reviews_2]

    # Creating the object for LDA model using gensim library
    LDA = gensim.models.ldamodel.LdaModel

    # Build LDA model
    lda_model = LDA(corpus=doc_term_matrix, id2word=dictionary, num_topics=num_topics,
                  random_state=100, chunksize=1000, passes=50)

    #pprint(lda_model.print_topics())
    top_topics = lda_model.top_topics(doc_term_matrix)
    top_topics = {i:lda_model.show_topic(i) for i in range(num_topics)}
    #pprint(top_topics)

    return top_topics, lda_model, doc_term_matrix
##




#############################################################################################################################################
#                                                                                                                                           #
#                                                     SENTENCES CLASSIFICATION                                                              #
#                                                                                                                                           #
#############################################################################################################################################
  
def classify_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)




#############################################################################################################################################
#                                                                                                                                           #
#                                                             POST PROCESSING                                                               #
#                                                                                                                                           #
#############################################################################################################################################
def aemi(word1, word2, texts_split):
    """Return probability than (word1,word2) is a good bigram"""
    p12 = 0
    p1_2 = 0
    p12_ = 0
    p1 = 0
    p2 = 0
    p1_ = 0
    p2_ = 0
    n = len(texts_split)
    orientation = 0
    for t in texts_split:
        if word1 in t:
            p1 += 1
        if word2 in t:
            p2 += 1
        if word1 not in t:
            p1_ += 1
        if word2 not in t:
            p2_ += 1
        if word1 in t and word2 in t:
            p12 += 1
            if t.index(word1) > t.index(word2):
                orientation -= 1
            else:
                orientation += 1
        if word1 in t and word2 not in t:
            p12_ += 1
        if word1 not in t and word2 in t:
            p1_2 += 1
    p12 /= n
    p1_2 /= n
    p12_ /= n
    p1 /= n
    p2 /= n
    p1_ /= n
    p2_ /= n
    if p12/(p1*p2) != 0:
        r1 = p12*math.log(p12/(p1*p2))
    else:
        r1 = -1000
        #return -10, 1
    if p12_/(p1*p2_) != 0:
        r2 = p12_*math.log(p12_/(p1*p2_))
    else:
        r2 = -1000
        #return -10, 1
    if p1_2/(p1_*p2) != 0:
        r3 = p1_2*math.log(p1_2/(p1_*p2))
    else:
        r3 = -1000
        #return -10, 1
    
    return r1-r2-r3, orientation


def find_bigrams(lda_topics, texts, tags=['NOUN','ADJ']):
    """Find representative bigrams"""
    ### Clean text
    ## Add lemmantization ? Remove punkt
    nlp = spacy.load("models/en_core_web_sm-3.0.0/en_core_web_sm/en_core_web_sm-3.0.0")

    stop_words = stopwords.words('english')
    texts_clean = [remove_stopwords(r.split(), stop_words).lower() for r in texts]

    tokenized_reviews = pd.Series(texts_clean).apply(lambda x: x.split())
    texts_clean = lemmatization(tokenized_reviews, nlp, tags=tags)
    reviews_3 = []
    for i in range(len(texts_clean)):
      reviews_3.append(' '.join(texts_clean[i]))

    texts_split = [r.split(' ') for r in reviews_3]

    ### Create dict of unique words
    vocab_text = {}
    vocab_topic = {}
    for t in texts_split:
      for w in t:
        vocab_text[w] = True
    for t in lda_topics:
        for w in lda_topics[t]:
            vocab_topic[w[0]] = True

    ### Find bigrams
    print('search bigramm')
    bigrams = []
    for word1 in vocab_topic:
        best_big = None
        max_aemi = -1000
        for word2 in vocab_text:
            if word2 != word1:
                aemi_, orientation = aemi(word1, word2, texts_split)
                if aemi_ > max_aemi:
                    max_aemi = aemi_
                    if orientation >= 0:
                        best_big = (word1, word2)
                    else:
                        best_big = (word2, word1)
        print(best_big, max_aemi)
        bigrams.append(best_big)

    
    return bigrams

def filter_topics_keywords(lda_topics, texts, n=5, tags=['NOUN','ADJ']):
    '''Return n clean keywords of each topic'''

    bigrams = find_bigrams(lda_topics, texts, tags=['NOUN','ADJ'])
    print('bigrams : ')
    print(bigrams)
    
    # Build clean topics keywords
    scores_lda = [{} for _ in range(len(lda_topics))]     # Lda scores of words
    scores_words = [{} for _ in range(len(lda_topics))]   # Our score of words
    words = []                                            # List of words
    for topic_id in lda_topics:
        for word,score in lda_topics[topic_id]:
            scores_lda[topic_id][word] = score
            words.append(word)

    for i,topic_dic in enumerate(scores_lda):
        for word in topic_dic:
            score = topic_dic[word]
            scores_words[i][word] = scores_lda[i][word]/(words.count(word)**2)

    topics = []
    for t in range(len(scores_words)):
        sorted_words = sorted(scores_words[t].items(), key=lambda item: item[1])
        sorted_words = [i[0] for i in sorted_words]
        topics.append(sorted_words[-n:])

    topics = {i:t for i,t in enumerate(topics)}
    
    return topics



def filter_topics_keywords_naive(lda_topics, n=1):
    '''Return n clean keywords of each topic'''

    
    # Build clean topics keywords
    scores_lda = [{} for _ in range(len(lda_topics))]     # Lda scores of words
    scores_words = [{} for _ in range(len(lda_topics))]   # Our score of words
    words = []                                            # List of words
    for topic_id in lda_topics:
        for word,score in lda_topics[topic_id]:
            scores_lda[topic_id][word] = score
            words.append(word)

    for i,topic_dic in enumerate(scores_lda):
        for word in topic_dic:
            score = topic_dic[word]
            scores_words[i][word] = scores_lda[i][word]/(words.count(word)**2)

    topics = []
    for t in range(len(scores_words)):
        sorted_words = sorted(scores_words[t].items(), key=lambda item: item[1])
        sorted_words = [i[0] for i in sorted_words]
        topics.append(sorted_words[-n:])

    topics = {i:t for i,t in enumerate(topics)}
    
    return topics


        
        


if __name__ == '__main__':

    ##    df = pd.read_csv('datasets/IMDB/IMDB Dataset.csv')
    ##    comments = df['review'].tolist()[:200]

    df = pd.read_csv('datasets/Womens Clothing/Womens Clothing E-Commerce Reviews.csv')
    df = df[['Clothing ID','Review Text']].dropna()
    print(df['Clothing ID'].value_counts(sort=True).head())
    df = df[df['Clothing ID'] == 1078]
    comments = df['Review Text'].tolist()[:2000]

##    #find_best_topics_number(df['Review Text'], tags=['NOUN'])
##    lda_topics, lda_model, corpus = topic_extraction(df['Review Text'], num_topics=5, tags=['NOUN'])#, 'ADJ'])
##
##    pickle.dump((lda_topics, lda_model, corpus), open('lda_outputs.p','wb'))

    lda_topics, lda_model, corpus = pickle.load(open('lda_outputs.p','rb'))

##    df_topic_sents_keywords = classify_sentences(ldamodel=lda_model, corpus=corpus, texts=comments)
##    # Format
##    df_dominant_topic = df_topic_sents_keywords.reset_index()
##    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
##    df_dominant_topic = df_dominant_topic.drop(['Document_No'], axis=1)
####    print(df_dominant_topic[['Dominant_Topic', 'Keywords']].head(20).to_string())
####    # Show
####    print(df_dominant_topic.head(1).to_string())
####    input()

    topics = filter_topics_keywords(lda_topics, df['Review Text'])

    pprint(lda_topics)
    pprint(topics)





