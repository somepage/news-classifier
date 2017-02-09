import re
import pickle
import numpy as np
import functools
from nltk.stem.snowball import RussianStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib


def predictor(sent):
    sent = stemming_sent(sent)
    tfid = joblib.load("tfidf.pkl")
    clf = joblib.load("model.pkl")
    sent = tfid.transform([sent]).toarray()
    #sent = sent[:,:62572]
    n_cls = clf.predict(sent)
    return n_cls


def stemming_sent(sent):
    pattern = re.compile('[a-zA-Zа-яА-Я]+')
    words = pattern.findall(sent)
    stemmer = RussianStemmer()
    words = list(map(lambda word: stemmer.stem(word), words))
    new_sent = functools.reduce(lambda x, y: x + ' ' + y, words)
    return new_sent


def learning():
    with open('text_base.pickle', 'rb') as f:  # text contains words from title with high probability
        text_base = pickle.load(f)

    categories = list(text_base.keys())
    X = []  # sentences
    Y = []  # category of sentence
    for category in categories:
        for text in text_base[category]:
            if text:
                X.append(text)
                Y.append(category)



    vect_stemming = np.vectorize(stemming_sent)
    X = vect_stemming(X)
    x_train, x_test, y_train, y_test = train_test_split(X, Y,
                                                        train_size=0.75)  # 75 % for training set and 25 % for test
    del X

    tf_idf_v = TfidfVectorizer()
    tf_idf_v.fit(x_train)
    joblib.dump(tf_idf_v, "tfidf.pkl")
    x_train = tf_idf_v.transform(x_train).toarray()
    nb = MultinomialNB()
    nb.fit(x_train, y_train)
    joblib.dump(nb, 'model.pkl')
    x_test = tf_idf_v.transform(x_test)
    print(nb.score(x_test, y_test))  # return accuracy of prediction (83 % in this model)
