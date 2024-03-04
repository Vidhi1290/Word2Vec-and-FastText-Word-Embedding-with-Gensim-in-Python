#!/usr/bin/env python
# coding: utf-8

# #  Importing Libraries

# In[1]:


import streamlit as st  #importing streamlit liabrary


# In[2]:


import pandas as pd

import numpy as np
import gensim
from gensim.models import Word2Vec
from gensim.models import FastText
from sklearn.decomposition import PCA
from matplotlib import pyplot


# In[3]:


import matplotlib.pyplot as plt
import plotly.graph_objects as go     # our main display package
import string # used for preprocessing
import re # used for preprocessing
import nltk # the Natural Language Toolkit, used for preprocessing
import numpy as np # used for managing NaNs
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords # used for preprocessing
from nltk.stem import WordNetLemmatizer # used for preprocessing
from sklearn.model_selection import train_test_split
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')


# # Importing datasets 

# In[4]:


df=pd.read_csv('Dimension-covid.csv')   #for preprocessing
df1=pd.read_csv('Dimension-covid.csv')  #for returning results


# # Preprocessing data 

# In[5]:


# function to remove all urls
def remove_urls(text):    
    new_text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text).split())
    return new_text

# make all text lowercase
def text_lowercase(text):
    return text.lower()

# remove numbers
def remove_numbers(text):
    result = re.sub(r'\d+', '', text)
    return result

# remove punctuation
def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

# tokenize
def tokenize(text):
    text = word_tokenize(text)
    return text

# remove stopwords
stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    text = [i for i in text if not i in stop_words]
    return text

# lemmatize Words 
lemmatizer = WordNetLemmatizer()
def lemmatize(text):
    text = [lemmatizer.lemmatize(token) for token in text]
    return text

#Creating one function so that all functions can be applied at once
def preprocessing(text):
    
    text = text_lowercase(text)
    text = remove_urls(text)
    text = remove_numbers(text)
    text = remove_punctuation(text)
    text = tokenize(text)
    text = remove_stopwords(text)
    text = lemmatize(text)
    text = ' '.join(text)
    return text


skipgram = Word2Vec.load('skipgramx11.bin')
FastText=Word2Vec.load('FastText.bin')



# In[12]:


vector_size=100   #defining vector size for each word



def get_mean_vector(word2vec_model, words):
    # remove out-of-vocabulary words
    words = [word for word in tokenize(words) if word in list(word2vec_model.wv.index_to_key)]
    if len(words) >= 1:
        return np.mean(word2vec_model.wv[words], axis=0)
    else:
        return np.array([0]*100)




K=pd.read_csv('skipgram-vec.csv')   

K2=[]                          
for i in range(df.shape[0]):
    K2.append(K[str(i)].values)



KK=pd.read_csv('FastText-vec.csv')

K1=[]
for i in range(df.shape[0]):
    K1.append(KK[str(i)].values)



from numpy import dot
from numpy.linalg import norm
def cos_sim(a,b):

    return dot(a, b)/(norm(a)*norm(b)) 






pd.set_option("display.max_colwidth", -1)       #this function will display full text from each column




#streamlit function 
def main():
    # Load data and models
    data = df1     #our data which we have to display
      
    

    st.title("Clinical Trial Search engine")      #title of our app
    st.write('Select Model')       #text below title

    
    Vectors = st.selectbox("Model",options=['Skipgram' , 'Fasttext'])
    if Vectors=='Skipgram':
        K=K2
        word2vec_model=skipgram
    elif Vectors=='Fasttext':
        K=K1
        word2vec_model=FastText

    st.write('Type your query here')

    query = st.text_input("Search box")   #getting input from user

    def preprocessing_input(query):
            
            query=preprocessing(query)
            query=query.replace('\n',' ')
            K=get_mean_vector(word2vec_model,query)
   
        
            return K   

    def top_n(query,p,df1):
        
        
        query=preprocessing_input(query)   
                                    
        x=[]
    
        for i in range(len(p)):
            
            x.append(cos_sim(query,p[i]))
        tmp=list(x)    
        res = sorted(range(len(x)), key = lambda sub: x[sub])[-10:]
        sim=[tmp[i] for i in reversed(res)]
        print(sim)

        L=[]
        for i in reversed(res):
           
    
            L.append(i)
        return df1.iloc[L, [1,2,5,6]],sim  
    
    model = top_n
    if query:
        
        P,sim =model(str(query),K,data)     #storing our output dataframe in P
        #Plotly function to display our dataframe in form of plotly table
        fig = go.Figure(data=[go.Table(header=dict(values=['ID', 'Title','Abstract','Publication Date','Score']),cells=dict(values=[list(P['Trial ID'].values),list(P['Title'].values), list(P['Abstract'].values),list(P['Publication date'].values),list(np.around(sim,4))],align=['center','right']))])
        #displying our plotly table
        fig.update_layout(height=1700,width=700,margin=dict(l=0, r=10, t=20, b=20))
        
        st.plotly_chart(fig) 
        # Get individual results
    

if __name__ == "__main__":
    main()

