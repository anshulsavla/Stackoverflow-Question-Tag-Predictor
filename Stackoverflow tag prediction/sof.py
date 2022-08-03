import numpy as np
import pandas as pd
import streamlit as st
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words=set(nltk.corpus.stopwords.words('english'))
from sklearn.preprocessing import MultiLabelBinarizer

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer , TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
#from skmultilearn.problem_transform import LabelPowerset
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import hamming_loss
from sklearn.model_selection import train_test_split
import re
import pickle

multilabel_model = pickle.load(open("C:\\Users\\User-1\\sof_first_model.sav" , "rb"))
tfidf_vectorizer = pickle.load(open("C:\\Users\\User-1\\tfidf_object.pickle", "rb"))
mlb = pickle.load(open("C:\\Users\\User-1\\multilabel_binarizer.pickle", "rb"))

# defining model function -
def model_func(question):
    # processing the question -
    question = question.replace('<p>', ' ')

    f1 = lambda x: re.sub('(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)', ' ', x)
    question = f1(question)

    question = question.replace('</p>', ' ')
    question = question.replace('\n', ' ')
    question = question.replace('</a>', ' ')

    f2 = lambda x: x.lower()
    question = f2(question)

    f3 = lambda x: ' '.join([w for w in x.split() if not w in stop_words])
    question = f3(question)

    question = question.replace('<a href=" ">', '')

    f4 = lambda x: ' '.join([w for w in x.split() if len(w) > 3])
    question = f4(question)

    # applying tfidf -
    tr_question = tfidf_vectorizer.transform(pd.Series([question]))

    # predicting the tags for the input question -
    return multilabel_model.predict(tr_question)





st.title('POST YOUR QUESTION')
TweetText = st.text_input("Query")
#TweetText = [TweetText]

if TweetText is not None:
    st.write(TweetText)
    label=model_func(TweetText)
    tag = mlb.inverse_transform(label)
    st.write(tag)
    
    
    st.button('Press')
