import pandas as pd
import numpy as np
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn import metrics
import streamlit as st

# Download NLTK data
nltk.download('punkt')

# Function to preprocess the data
def preprocess_data(df):
    tokenized_list = []
    stemmer = PorterStemmer()

    for i in range(len(df)):
        cur_text = df.iloc[i]['Text']
        cur_text = cur_text.replace('\n', '')
        cur_tokenized = word_tokenize(cur_text)
        
        # Remove punctuations
        cur_tokenized = [word.lower() for word in cur_tokenized if word.lower() not in string.punctuation]
        
        # Stemming
        cur_tokenized = [stemmer.stem(word) for word in cur_tokenized]
        tokenized_list.append(cur_tokenized)
    
    df['Processed_Text'] = tokenized_list
    return df

# Load and preprocess the data
@st.cache
def load_data():
    try:
        df = pd.read_csv('data.csv', header=None)
        col = [0, 1]
        new_df = df[col].copy()  # Create a copy to avoid modifying the original DataFrame
        new_df = new_df[pd.notnull(new_df[1])]
        new_df.columns = ['Emotion', 'Text']
        return preprocess_data(new_df)
    except Exception as e:
        st.error("Error loading data: {}".format(str(e)))
        return None

# Function to return tokenized list for vectorization
def return_phrase(input_list):
    return input_list

# Function to transform sentence for prediction
def transform_sentence(sent):
    tokenized_stemmed = []
    stemmer = PorterStemmer()
    sent_tokenized = word_tokenize(sent)
    sent_tokenized = [word.lower() for word in sent_tokenized if word.lower() not in string.punctuation]
    sent_tokenized = [stemmer.stem(word) for word in sent_tokenized]
    tokenized_stemmed.append(sent_tokenized)
    return tokenized_stemmed

# Load data
data = load_data()

# Splitting the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(data['Processed_Text'], data['Emotion'], test_size=.3, random_state=1)

# Extracting features for both models
my_vectorizer = TfidfVectorizer(analyzer='word', tokenizer=return_phrase, preprocessor=return_phrase, token_pattern=None, ngram_range=(1, 3))
my_vectorizer.fit(X_train)
transformed_train = my_vectorizer.transform(X_train).toarray()
transformed_test = my_vectorizer.transform(X_test).toarray()

# Training Multinomial Naive Bayes model
cur_alpha = 0.33
nb_classifier = MultinomialNB(alpha=cur_alpha)
nb_classifier.fit(transformed_train, Y_train)

# Training SVM Model
cur_c = 2
svm_classifier = svm.LinearSVC(C=cur_c)
svm_classifier.fit(transformed_train, Y_train)

# Streamlit App
st.title('Text Emotion Detection')

# User input for prediction
sentence = st.text_input('Enter a sentence:', '')

# Predictions
if st.button('Predict Emotion (Naive Bayes)'):
    if sentence:
        pred = nb_classifier.predict(my_vectorizer.transform(transform_sentence(sentence)).toarray())
        st.write('Predicted Emotion (Naive Bayes):', pred[0])

if st.button('Predict Emotion (SVM)'):
    if sentence:
        pred = svm_classifier.predict(my_vectorizer.transform(transform_sentence(sentence)).toarray())
        st.write('Predicted Emotion (SVM):', pred[0])

