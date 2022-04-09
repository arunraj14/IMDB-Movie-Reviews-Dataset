# -*- coding: utf-8 -*-
"""
Created on Fri Apr 08 2022

@author: Arunraj
"""

from flask import Flask, request
import numpy as np
import pickle
import pandas as pd
import nltk as nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
import flasgger
from flasgger import Swagger

app=Flask(__name__)
Swagger(app)

# unpickling the model
pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)

# label for prediction Output
pred = {1: 'Positive', 0: 'Negative'}

# data prepreocessing steps
def preprocessing(df):
    porter=PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('from')
    df['review'] = df['review'].str.lower()
    df['review'] = df['review'].apply(lambda x: BeautifulSoup(x, "html.parser").get_text())
    df['review'] = df['review'].replace(to_replace=r"[^A-Za-z0-9]+", value=r" ", regex=True)
    df['review'] = df['review'].apply(lambda x: ' '.join([word for word in x.split() if not word.isdigit()]))
    df['review'] = df['review'].apply(lambda x: ' '.join([word for word in str(x).split() if word not in (all_stopwords)]))
    df['review'] = df['review'].apply(lambda x :' '.join([porter.stem(word) for word in x.split()]))
    return df['review'].values

@app.route('/')
def welcome():
    return "Please use localhost:8000/apidocs Or 127.0.0.1:8000/apidocs/ for Prediction"

@app.route('/predict',methods=["Get"])
def predict_note_authentication():
    
    """Let's predict movie review sentiments 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: review
        in: query
        type: string
        required: true
    responses:
        200:
            description: The output values
        
    """
    review=request.args.get("review")
    rev_df = pd.DataFrame([review],columns=['review'])
    rev_pre = preprocessing(rev_df)
    prediction=classifier.predict(rev_pre)    
    return "Hello The answer is "+str(pred[prediction[0]])

@app.route('/predict_file',methods=["POST"])
def predict_note_file():
    """Let's predict movie review sentiments 
    This is using docstrings for specifications.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      
    responses:
        200:
            description: The output values
        
    """
    rev_df=pd.read_csv(request.files.get("file"),header=None)
    rev_df.rename({0:'review'},axis=1,inplace=True)
    if rev_df.iloc[0]['review'] == ('review' or 'Review'):
        rev_df = rev_df.drop(0)
    rev_pre = preprocessing(rev_df)
    prediction=classifier.predict(rev_pre)    
        
    return str(list(prediction))

if __name__=='__main__':
    app.run(host='0.0.0.0',port=8000)
       