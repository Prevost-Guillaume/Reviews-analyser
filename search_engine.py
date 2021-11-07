import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize
from summarization import ExtractiveSummarySimilarity, ExtractiveSummaryClusturing, ExtractiveSummaryDouble
from sentence_transformers import SentenceTransformer
from polarization import load_sklearn_polarity_classifier, getPolarity
from topic_extraction import *
from vizualisation import *
from operator import itemgetter

from scipy.spatial import distance

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA



def dist(a,b):
    abs_sum = [(a[i]-b[i])**2 for i in range(len(a))]
    return sum(abs_sum)**0.5

def search(entry, texts, texts_embeddings, transformer, n=1):
    entry_embedding = transformer.encode(entry)
    text_dist = [[texts[i], dist(texts_embeddings[i],entry_embedding)] for i in range(len(texts_embeddings))]
    sorted_texts_dist = sorted(text_dist, key=itemgetter(1))
    return [i[0] for i in sorted_texts_dist][:n]
    

if __name__ == '__main__':
    df_comments = pd.read_pickle('df_comments.p')
    df_sentences = pd.read_pickle('df_sentences.p')
    transformer = SentenceTransformer("models/bert-base-nli-mean-tokens")

    print(df_comments.columns)
    print(df_sentences.columns)

    while True:
        entry = input('> ')
        #r = search(entry, df_comments['Comments'], df_comments['Embedding'], transformer, n=10)
        r = search(entry, df_sentences['Sentences'], df_comments['Embedding'], transformer, n=10)
        print('\n'.join(r[:10]))



