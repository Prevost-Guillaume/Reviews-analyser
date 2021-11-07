import warnings
warnings.simplefilter("ignore")

import numpy as np
from sklearn.cluster import KMeans
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer



def cos_sim(a,b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def eucl_dist(a,b):
    return np.linalg.norm(a-b)



def ExtractiveSummaryClusturing(text, length, sentences_embeddings=None, metric='cosine'):
    '''Return the [length] most significative sentences from the text
    Most significative sentences are chosen as centroids of KMeans clusters'''


    # Split sentences
    sentences = sent_tokenize(text.replace('[sep]','.').replace('[SEP]','.'))
    
    # Encode sentences
    if sentences_embeddings is None:
        sentences_embeddings = model.encode(sentences)

    # Clustering sentences
    model_cluster = KMeans(n_clusters=length, random_state=0)
    preds = model_cluster.fit_predict(sentences_embeddings)

##    for i in range(max(list(preds))+1):
##        preds_i = [sentences[j] for j in range(len(preds)) if preds[j] == i]
##
##        print("\n".join(preds_i))
##        print('\n')
    
    centers = list(model_cluster.cluster_centers_)

    # Find n closest sentences from clusters
    summary = []
    best_sents = ['None' for _ in range(length)]
    if metric == 'euclidian':
        min_dists = [10000 for _ in range(length)]

        for sentence, sentence_embedding in zip(sentences, sentences_embeddings):
            for i,c in enumerate(centers):
                d = eucl_dist(sentence_embedding, c)
                if d < min_dists[i]:
                    min_dists[i] = d
                    best_sents[i] = sentence
        
    else:
        max_sims = [-1 for _ in range(length)]

        for sentence, sentence_embedding in zip(sentences, sentences_embeddings):
            for i,c in enumerate(centers):
                d = cos_sim(sentence_embedding, c)
                if d > max_sims[i]:
                    max_sims[i] = d
                    best_sents[i] = sentence
                
    return best_sents




def ExtractiveSummarySimilarity(text, length, text_embedding=None, sentences_embeddings=None, model=None, metric='cosine'):
    '''Return the [length] most significative sentences from the text
    Most significative sentences are chosen as the most similar sentences to the entire text'''

    # Split sentences
    sentences = sent_tokenize(text)
    
    #text = '[SEP]'.join(sentences)
    if text_embedding is None:
        text = text.lower()
        text_embedding = model.encode(text)
    
    # Encode sentences
    if sentences_embeddings is None:
        sentences_embeddings = model.encode(sentences)

    # A OPTIMISER
    sentences_sims = {s : cos_sim(text_embedding, sentences_embeddings[i]) for i,s in enumerate(list(sentences))}
    sorted_sims = sorted([sentences_sims[s] for s in sentences_sims])[-length:]

    summary = []
    for sim in sorted_sims[::-1]:
        summary.append(list(sentences_sims.keys())[list(sentences_sims.values()).index(sim)])
                    
    return "\n".join(summary)



def ExtractiveSummaryDouble(text, length, text_embedding=None, sentences_embeddings=None, model=None, metric='cosine'):
    '''Return the [length] most significative sentences from the text
    Most significative sentences are chosen as the most similar sentences to each centroids of KMeans clusters'''

    if text_embedding is None:
        text = text.lower()
        text_embedding = model.encode(text)

    # Split sentences
    sentences = sentences = text.lower().replace('!','.').split()#text.split('[SEP]')

    # Encode sentences
    if sentences_embeddings is None:
        sentences_embeddings = model.encode(sentences)
    
    # Clustering sentences
    model_cluster = KMeans(n_clusters=length, random_state=0)
    preds = model_cluster.fit_predict(sentences_embeddings)

    summary = []
    for i in range(max(list(preds))+1):
        sentences_i = [sentences[j] for j in range(len(preds)) if preds[j] == i]
        sentences_embeddings_i = [sentences_embeddings[j] for j in range(len(preds)) if preds[j] == i]

        max_sim = -1
        best_sent = 'None'
        for i,s in enumerate(sentences_i):
            d = cos_sim(text_embedding, sentences_embeddings_i[i])
            if d > max_sim:
                max_sim = d
                best_sent = s
        summary.append(best_sent)


                
    return summary






if __name__ == '__main__':
    import pandas as pd

    text = '''A young girl, Fantine, has been abandoned by her lover; unfortunately, she has a child, little Cosette. To provide for them, Fantine is willing to do any job, but rejected everywhere as a mother's daughter, she is forced to engage in prostitution. An argument of poor Fantine with a fool who throws snow on her back brings her into the presence of the dreaded Javert, the police made man. Javert naturally gives the girl the wrong side of the law, but then he comes up against M. Madeleine, the town's mayor, who, having entered the office by chance, has heard Fantine's whole lamentable confession, and who, taken with pity, takes it upon himself to have her released. This impossible trait, a mayor saving a public girl, exasperates Javert and confirms the suspicions that other facts have already provoked in his mind. Isn't M. Madeleine hiding another personality under a false name? Javert lets this doubt be known; this greatly disturbs M. Madeleine, since he is none other than Jean Valjean himself, and he thus sees himself on the verge of losing all the fruits of ten years of probity. Another incident disturbs him even more deeply: he learns that an unfortunate man, arrested under the false name of Jean Valjean, is now being tried in a court of assizes. The unfortunate man wonders whether he should let the innocent man be condemned, a condemnation that will secure his future and strengthen his borrowed personality, and without making up his mind, driven by a kind of instinct, he goes to the court of assizes. There he sees the unfortunate man, the image of the old Valjean, stammering in a daze with recriminations that convince no one: he is going to be condemned. M. Madeleine rises and declares that he is Jean Valjean; he makes himself known to his fellow prisoners, who have been called in to be confronted with the false Valjean, and he is happily taken in by the merciless Javert. However, he is left free momentarily and he takes advantage of this respite to witness the agony of Fantine, who is dying on a hospital bed. He swears to her, whose death he blames himself for having caused by chasing her out of his workshop, to adopt her daughter, little Cosette, and he manages to escape to Paris, where he withdraws 600,000 francs from the Laffitte bank and buries them in a wood.'''
    text = '''The Federal Reserve Bank of New York president, John C. Williams, made clear on Thursday evening that officials viewed the emergency rate cut they approved earlier this week as part of an international push to cushion the economy as the coronavirus threatens global growth.
Mr. Williams, one of the Fed’s three key leaders, spoke in New York two days after the Fed slashed borrowing costs by half a point in its first emergency move since the depths of the 2008 financial crisis. The move came shortly after a call between finance ministers and central bankers from the Group of 7, which also includes Britain, Canada, France, Germany, Italy and Japan.
“Tuesday’s phone call between G7 finance ministers and central bank governors, the subsequent statement, and policy actions by central banks are clear indications of the close alignment at the international level,” Mr. Williams said in a speech to the Foreign Policy Association.
Rate cuts followed in Canada, Asia and the Middle East on Wednesday. The Bank of Japan and European Central Bank — which already have interest rates set below zero — have yet to further cut borrowing costs, but they have pledged to support their economies.
Mr. Williams’s statement is significant, in part because global policymakers were criticized for failing to satisfy market expectations for a coordinated rate cut among major economies. Stock prices temporarily rallied after the Fed’s announcement, but quickly sank again.
Central banks face challenges in offsetting the economic shock of the coronavirus.
Many were already working hard to stoke stronger economic growth, so they have limited room for further action. That makes the kind of carefully orchestrated, lock step rate cut central banks undertook in October 2008 all but impossible.
Interest rate cuts can also do little to soften the near-term hit from the virus, which is forcing the closure of offices and worker quarantines and delaying shipments of goods as infections spread across the globe.
“It’s up to individual countries, individual fiscal policies and individual central banks to do what they were going to do,” Fed Chair Jerome H. Powell said after the cut, noting that different nations had “different situations.”
Mr. Williams reiterated Mr. Powell’s pledge that the Fed would continue monitoring risks in the “weeks and months” ahead. Economists widely expect another quarter-point rate cut at the Fed’s March 18 meeting.
The New York Fed president, whose reserve bank is partly responsible for ensuring financial markets are functioning properly, also promised that the Fed stood ready to act as needed to make sure that everything is working smoothly.
Since September, when an obscure but crucial corner of money markets experienced unusual volatility, the Fed has been temporarily intervening in the market to keep it calm. The goal is to keep cash flowing in the market for overnight and short-term loans between banks and other financial institutions. The central bank has also been buying short-term government debt.
“We remain flexible and ready to make adjustments to our operations as needed to ensure that monetary policy is effectively implemented and transmitted to financial markets and the broader economy,” Mr. Williams said Thursday.'''

    model = SentenceTransformer("models/bert-base-nli-mean-tokens")
    sentences = text.lower().replace('!','.').split()
    sentences_embeddings = model.encode(sentences)
    
    summary1 = ExtractiveSummarySimilarity(text, 2, sentences_embeddings=sentences_embeddings, model=model, metric='cosine')
    print(summary1,'\n\n')
    summary2 = ExtractiveSummaryClusturing(text, 2, sentences_embeddings=sentences_embeddings, metric='cosine')
    print(summary2,'\n\n')
    summary3 = ExtractiveSummaryDouble(text, 2, sentences_embeddings=sentences_embeddings, model=model, metric='cosine')
    print(summary3,'\n\n')




















