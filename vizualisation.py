import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from polarization import load_sklearn_polarity_classifier, getPolarity

from statistics import *

import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA

import pickle




def polarity_distrib(polarities, comments):
    """Show distribution of polarity"""
    hist_data = [[p for p in polarities if p>0],[p for p in polarities if p<0]]
    text_data = [[comments[i] for i in range(len(polarities)) if polarities[i]>0],[comments[i] for i in range(len(polarities)) if polarities[i]<0]]
    group_labels = ['positive', 'negative'] # name of the dataset

    fig = ff.create_distplot(hist_data, group_labels, bin_size=.1, rug_text=text_data)
    fig.layout.paper_bgcolor = '#fff'
    fig.layout.plot_bgcolor = '#FFFFFF'
    fig.layout.title = 'Reviews polarity distribution'
    return fig
    


def get_polarity_stats(polarities):
    stats = {}

    stats['mean'] = mean(polarities)
    stats['median'] = median(polarities)
    stats['std'] = stdev(polarities)

    return stats


def pie_chart_matplotlib(polarities):
    pol_int = [p>0 for p in polarities]
    
    labels = 'negative', 'positive'
    sizes = [pol_int.count(0), pol_int.count(1)]
    colors = ['red', 'green']
    
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90, colors=colors)
    ax1.axis('equal')

    plt.show()


def pie_chart(polarities):
    """Plot a pie chart of comments polarity"""
    pol_int = [p>0 for p in polarities]
    pol_int = [1 if p else 0 for p in pol_int]
    labels = ["Positive" if p == 1 else "Negative" for p in pol_int]
    df = pd.DataFrame()
    df['pol'] = pol_int
    df['1'] = [1 for _ in pol_int]
    df['label'] = labels
    
    fig = px.pie(df, values='1', names='label', color_discrete_map={'Positive':'blue','Negative':'red'}, title='Reviews polarity')
    fig.layout.paper_bgcolor = '#fff'
    fig.layout.plot_bgcolor = '#FFFFFF'
    return fig



def get_implication(polarities):
    p = [abs(p) for p in polarities]
    return sum(p)/len(p)



def show_comments_polarity(df, df_mode='Comments'):
    """Plot a 2D scatter graph with comments, colored by polarity"""
    pca = PCA(n_components=2)
    x = [[float(j) for j in i] for i in df['Embedding']]

    x = list(pca.fit_transform(np.array(x)))
    fig = px.scatter(x=[i[0] for i in x], y=[i[1] for i in x],
                     color=[float(i) for i in df['Polarity']],
                     hover_name=df[df_mode],
                     title='Some awesome visualization')
    fig.layout.paper_bgcolor = '#FFFFFF'
    fig.layout.plot_bgcolor = '#FFFFFF'
    return fig




def show_comments_topics(df, df_mode='Comments'):
    """Plot a 2D scatter graph with comments, colored by topics, sized by confidencs"""
    pca = PCA(n_components=2)
    x = [[float(j) for j in i] for i in df['Embedding']]

    x = list(pca.fit_transform(np.array(x)))
    fig = px.scatter(x=[i[0] for i in x], y=[i[1] for i in x],
                     color=[str(round(i)) for i in df['Topic']],
                     size=[float(i) for i in df['Topic_score']**2],
                     hover_name=[str(i) for i in df[df_mode]],
                     title='Main topics of reviews')
    fig.layout.paper_bgcolor = '#FFFFFF'
    fig.layout.plot_bgcolor = '#FFFFFF'
    fig.layout.xaxis.showticklabels = False
    fig.layout.yaxis.showticklabels = False
    return fig





def plot_topic_polarity_(df, topics_dic, mode='count'):

    # define data set
    if mode == 'count':
        df['Polarity_int'] = 2*(df['Polarity']>0) - 1
    else:
        df['Polarity_int'] = df['Polarity']

    df_good, df_bad = df[df['Polarity']>=0], df[df['Polarity']<0]

    if mode == 'count':
        df_good = df_good[['Polarity_int', 'Topic']].groupby(['Topic']).sum()
        df_bad = df_bad[['Polarity_int', 'Topic']].groupby(['Topic']).sum()
    else:
        df_good = df_good[['Polarity_int', 'Topic']].groupby(['Topic']).mean()
        df_bad = df_bad[['Polarity_int', 'Topic']].groupby(['Topic']).mean()
    df_good['Polarity_int_good'] = df_good['Polarity_int']
    df_bad['Polarity_int_bad'] = df_bad['Polarity_int']

    df_plot = pd.concat([df_bad, df_good], axis=1)

    df_plot['label1'] = ['topic_pos_'+str(topics_dic[i][0]) for i in df_plot.index]
    df_plot['label2'] = ['topic_neg_'+str(topics_dic[i][0]) for i in df_plot.index]
    df_plot['value1'] = df_plot['Polarity_int_bad']
    df_plot['value2'] = df_plot['Polarity_int_good']
    df_plot = df_plot[['label1','label2','value1','value2']]


    # create subplots
    fig = make_subplots(rows=1, cols=2, specs=[[{}, {}]], shared_xaxes=True,
                        shared_yaxes=True, horizontal_spacing=0)
    fig.layout.paper_bgcolor = '#FFFFFF'
    fig.layout.plot_bgcolor = '#FFFFFF'

    fig.append_trace(go.Bar(y=df_plot.index, x=df_plot.value1, orientation='h', width=0.4, showlegend=False, marker_color='#4472c4'), 1, 1)
    fig.append_trace(go.Bar(y=df_plot.index, x=df_plot.value2, orientation='h', width=0.4, showlegend=False, marker_color='#ed7d31'), 1, 2)
    fig.update_yaxes(showticklabels=False) # hide all yticks

    annotations = []
    for i, row in df_plot.iterrows():
        if row.label1 != '':
            annotations.append({
                'xref': 'x1',
                'yref': 'y1',
                'y': i,
                'x': row.value1,
                'text': round(row.value1,2),
                'xanchor': 'right',
                'showarrow': False})
            annotations.append({
                'xref': 'x1',
                'yref': 'y1',
                'y': i-0.3,
                'x': -1,
                'text': row.label1,
                'xanchor': 'right',
                'showarrow': False})            
        if row.label2 != '':
            annotations.append({
                'xref': 'x2',
                'yref': 'y2',
                'y': i,
                'x': row.value2,
                'text': row.value2,
                'xanchor': 'left',
                'showarrow': False})  
            annotations.append({
                'xref': 'x2',
                'yref': 'y2',
                'y': i-0.3,
                'x': 1,
                'text': row.label2,
                'xanchor': 'left',
                'showarrow': False})

    fig.update_layout(annotations=annotations)
    return fig




def plot_topic_polarity(df, topics_dic, mode='count'):

    # define data set
    if mode == 'count':
        df['Polarity_int'] = 2*(df['Polarity']>0) - 1
    else:
        df['Polarity_int'] = df['Polarity']

    df_good, df_bad = df[df['Polarity']>=0], df[df['Polarity']<0]

    if mode == 'count':
        df_good = df_good[['Polarity_int', 'Topic']].groupby(['Topic']).sum()
        df_bad = df_bad[['Polarity_int', 'Topic']].groupby(['Topic']).sum()
    else:
        df_good = df_good[['Polarity_int', 'Topic']].groupby(['Topic']).mean()
        df_bad = df_bad[['Polarity_int', 'Topic']].groupby(['Topic']).mean()
    df_good['Polarity_int_good'] = df_good['Polarity_int']
    df_bad['Polarity_int_bad'] = df_bad['Polarity_int']

    df_plot = pd.concat([df_bad, df_good], axis=1)

    df_plot['label1'] = ['topic_pos_'+str(topics_dic[i][0]) for i in df_plot.index]
    df_plot['label2'] = ['topic_neg_'+str(topics_dic[i][0]) for i in df_plot.index]
    df_plot['value1'] = df_plot['Polarity_int_bad']
    df_plot['value2'] = df_plot['Polarity_int_good']
    df_plot = df_plot[['label1','label2','value1','value2']]


    # create subplots
    titles = []
    for i in df_plot.index:
        titles.append("Negative reviews on "+topics_dic[i][0])
        titles.append("Positive reviews on "+topics_dic[i][0])
        
    fig = make_subplots(rows=len(df_plot.index), cols=2, specs=[[{}, {}] for i in range(len(df_plot.index))], shared_xaxes=True,
                        shared_yaxes=True, horizontal_spacing=0,subplot_titles=titles)

    for i in df_plot.index:
        fig.add_trace(go.Bar(y=[0], x=[df_plot.value1[i]], orientation='h', width=0.4, showlegend=False, marker_color='#4472c4'), int(i+1), 1)
        fig.add_trace(go.Bar(y=[0], x=[df_plot.value2[i]], orientation='h', width=0.4, showlegend=False, marker_color='#ed7d31'), int(i+1), 2)
    fig.update_yaxes(showticklabels=False) # hide all yticks
    fig.layout.plot_bgcolor = '#FFFFFF'
    return fig





def plot_radar_topics(df, topics_dic, mode='quality'):
    df['1'] = 1
    df_good, df_bad = df[df['Polarity']>=0], df[df['Polarity']<0]
    df_good = df_good[['Polarity', 'Topic']].groupby(['Topic']).sum()
    df_bad = df_bad[['Polarity', 'Topic']].groupby(['Topic']).sum()
    df_good['Polarity_good'] = df_good['Polarity'].fillna(0)
    df_bad['Polarity_bad'] = df_bad['Polarity'].fillna(0)
    df_plot = pd.concat([df_bad, df_good], axis=1)
    df_plot['topic_name'] = [' '.join(topics_dic[i]) for i in df_plot.index]

    if mode == 'quality':
        df_plot['ratio'] = df_plot['Polarity_good']/(df_plot['Polarity_good']-df_plot['Polarity_bad'])
        fig = px.line_polar(df_plot, r='ratio', theta='topic_name', line_close=True, title='satisfaction of each topic')
        fig.update_traces(fill='toself')
    elif mode == 'importance':
        df_plot['ratio'] = df[['Topic','1']].groupby(['Topic']).sum()['1']/len(df['Polarity'])
        #df_plot['ratio'] = (df_plot['Polarity_good']-df_plot['Polarity_bad'])/len(df['Polarity'])
        fig = px.line_polar(df_plot, r='ratio', theta='topic_name', line_close=True, title='Importance of topics in reviews')
        fig.update_traces(fill='toself')
    return fig  







if __name__ == '__main__':
    transformer = SentenceTransformer("models/bert-base-nli-mean-tokens")
    polarity_classifier = load_sklearn_polarity_classifier(file='models/polarity_classifier_sklearn_imdb.sav')

    # Load reviews
##    df = pd.read_csv('datasets/Womens Clothing/Womens Clothing E-Commerce Reviews.csv')
##    df = df[['Clothing ID','Review Text']].dropna()
##    print(df['Clothing ID'].value_counts(sort=True).head())
##    df = df[df['Clothing ID'] == 1078]
##
##    comments = df['Review Text'].tolist()[:2000]


    df = pd.read_csv('datasets/IMDB/IMDB Dataset.csv')
    comments = df['review'].tolist()[:500]

##    comments = [c.lower() for c in comments]
##    comments_embedding = transformer.encode(comments)
##    
##
##    # Show vizualisations
##    polarities = getPolarity(comments_embedding, polarity_classifier, mode='proba')
##    pickle.dump(polarities, open('polarities_viz.p','wb'))

    polarities = pickle.load(open('polarities_viz.p','rb'))

    plot_radar_topics(df, topics_dic, mode='quality')
    











