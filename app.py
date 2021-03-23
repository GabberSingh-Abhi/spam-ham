import streamlit as st
st.title('Spam Ham Classification')
import pandas as pd
df=pd.read_table('spam.tsv')
x=df.iloc[:,1]
y=df.iloc[:,0]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0,stratify=y)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
text_model=Pipeline([('tfidf',TfidfVectorizer()),('model',SVC())])
text_model.fit(x_train,y_train)
select=st.text_input('Enter yoyr message')
op=text_model.predict([select])
st.title(op[0])
 
