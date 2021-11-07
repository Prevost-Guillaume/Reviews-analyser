import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC

from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Dropout, BatchNormalization

from keras.layers import Dense
from keras.models import load_model
from keras.models import Model
from keras.layers import Input, concatenate
from keras.regularizers import l2

from keras.wrappers.scikit_learn import KerasClassifier
import time

#####################################################################################################################################
#                                                                                                                                   #
#                                                       POLARITY FUNCTION                                                           #
#                                                                                                                                   #
#####################################################################################################################################
def load_sklearn_polarity_classifier(file='models/polarity_classifier_sklearn_imdb.sav'):
    return pickle.load(open(file, 'rb'))

    
def getPolarity(sents_embedding, classifier, mode='int'):
    """Return polarity of text"""
    
    if mode == 'int':
        pred = classifier.predict(sents_embedding)
    elif mode == 'proba':
        pred = classifier.predict_proba(sents_embedding)
        pred = [p[1]-p[0] for p in list(pred)]

    return pred



#####################################################################################################################################
#                                                                                                                                   #
#                                                           LOAD DATA                                                               #
#                                                                                                                                   #
#####################################################################################################################################


def load_data_movie_review():
    df = pd.read_csv('datasets/movie_review/movie_review.csv')
    df = df[['text','tag']]
    df = df.dropna()
    df = shuffle(df)
    print('dataset length : ',len(df['tag']))

    # Get Xy_train and Xy_test
    X = []
    n = len(df['text'])
    for i,s in enumerate(df['text']):
        X.append(transformer.encode(s))
        if i%100 == 0:
            print(100*i/n)
            
    X = np.array(X)
    y = df['tag'].map({'pos':1,'neg':0})#y = pd.get_dummies(df['tag'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print('X_train : ',X_train.shape)
    print('y_train : ',y_train.shape)

    pickle.dump([X_train, X_test, y_train, y_test], open('Xy_classifier_movie_review.p', 'wb'))
    return


    
def load_data_imdb():
    df = pd.read_csv('datasets/IMDB/IMDB Dataset.csv')
    df = df.dropna()
    df = shuffle(df)[100:]
    print('dataset length : ',len(df['sentiment']))
    
    X = []
    n = len(df['review'])
    for i,s in enumerate(df['review']):
        X.append(transformer.encode(s))
        if i%100 == 0:
            print(100*i/n)
            
    X = np.array(X)
    y = df['sentiment'].map({'positive':1,'negative':0})
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print('X_train : ',X_train.shape)
    print('y_train : ',y_train.shape)

    pickle.dump([X_train, X_test, y_train, y_test], open('Xy_classifier_imdb.p', 'wb'))



#####################################################################################################################################
#                                                                                                                                   #
#                                                       MODELS TRAINING                                                             #
#                                                                                                                                   #
#####################################################################################################################################

def create_keras_model(layer1_units=64, layer2_units=64, dropout_rate=0, l2_regularization=0):
    model = Sequential()
    model.add(Dense(units=layer1_units, activation='relu', input_dim=768))
    model.add(Dense(units=layer2_units, activation='relu', kernel_regularizer=l2(l2_regularization)))
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(units=2, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


def trainClassifier_sklearn(transformer, fileout='polarity_classifier.sav'):
    X_train, X_test, y_train, y_test = pickle.load(open('Xy_classifier_imdb.p', 'rb'))
    

    # Train classifier
    #classifier = make_pipeline(StandardScaler(),LinearSVC(C=0.001, random_state=0))
    classifier = make_pipeline(StandardScaler(),SVC(kernel='linear',probability=True, C=0.001, random_state=0))
    
    classifier.fit(X_train, y_train)

    print('save model')
    pickle.dump(classifier, open(fileout, 'wb'))

    # Evaluate classifier
    y_pred = classifier.predict(X_test)
    print('score train : ',classifier.score(X_train, y_train))
    print('score test : ',classifier.score(X_test, y_test))

    return classifier



 

def trainClassifier_keras(transformer, fileout='polarity_classifier.sav'):
    X_train, X_test, y_train, y_test = pickle.load(open('Xy_classifier_imdb.p', 'rb'))
    y_train = np.array([[y==0, y==1] for y in list(y_train)])
    y_test = np.array([[y==0, y==1] for y in list(y_test)])
    
    
    # create classifier pipeline
    classifier_keras = KerasClassifier(
        create_keras_model, 
        batch_size=32, 
        layer1_units=128,
        layer2_units=32,
        dropout_rate=0.4,
        l2_regularization=1e-5,
        epochs=40, 
        verbose=False)
    
    classifier = make_pipeline(StandardScaler(),classifier_keras)


    # Train and save classifier
    classifier.fit(X_train, y_train)

    #print('save model')
    #pickle.dump(classifier, open(fileout, 'wb'))


    # Evaluate classifier
    y_pred = classifier.predict(X_test)
    print('score train : ',classifier.score(X_train, y_train))
    print('score test : ',classifier.score(X_test, y_test))

    return classifier
    



    

#####################################################################################################################################
#                                                                                                                                   #
#                                                               MAIN                                                                #
#                                                                                                                                   #
#####################################################################################################################################



if __name__ == '__main__':
    transformer = SentenceTransformer("models/bert-base-nli-mean-tokens")
    #load_data_imdb()
    trainClassifier_sklearn(transformer, fileout='models/polarity_classifier.sav')

    #loaded_model = pickle.load(open(fileout, 'rb'))


    
