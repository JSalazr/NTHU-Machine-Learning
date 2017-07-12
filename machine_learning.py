from smart_open import smart_open
import pandas as pd
import numpy as np
from numpy import random
import gensim
import nltk
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from nltk.corpus import stopwords
from helpers import *
import nltk
nltk.download('punkt')

# read reviews
df = pd.read_csv('training_data.csv')
df = df.dropna()

# count the number of words
df['text'].apply(lambda x: len(x.split(' '))).sum()
print df
my_tags = ['m', 'f']

print df.gender.value_counts()
train_data, test_data = train_test_split(df, test_size=0.25, random_state=42)
print(len(test_data), len(train_data))

print "\n---BAG OF WORDS---"
count_vectorizer = CountVectorizer(
    analyzer="word", tokenizer=nltk.word_tokenize,
    preprocessor=None, stop_words='english', max_features=4000)

train_data_features = count_vectorizer.fit_transform(train_data['text'])
logreg_model = linear_model.LogisticRegression(n_jobs=1, C=1e5)
logreg_model = logreg_model.fit(train_data_features, train_data['gender'])
print word_embeddings.predict(count_vectorizer, logreg_model, test_data, my_tags)

print "\n---N-GRAMS---"
n_gram_vectorizer = CountVectorizer(
    analyzer="char",
    ngram_range=([2,5]),
    tokenizer=None,
    preprocessor=None,
    max_features=4000)

charn_model = linear_model.LogisticRegression(n_jobs=1, C=1e5)

train_data_features = n_gram_vectorizer.fit_transform(train_data['text'])

charn_model = charn_model.fit(train_data_features, train_data['gender'])
print word_embeddings.predict(n_gram_vectorizer, charn_model, test_data, my_tags)

print "\n---TF-IDF---"
tf_vect = TfidfVectorizer(
    min_df=2, tokenizer=nltk.word_tokenize,
    preprocessor=None, stop_words='english')
train_data_features = tf_vect.fit_transform(train_data['text'])

tfidf_model = linear_model.LogisticRegression(n_jobs=1, C=1e5)
tfidf_model = tfidf_model.fit(train_data_features, train_data['gender'])
print word_embeddings.predict(tf_vect, tfidf_model, test_data, my_tags)
