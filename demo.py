from unicodedata import name
import gradio as gr
from joblib import load

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
nlp = spacy.load('en_core_web_lg')

from sklearn.feature_extraction.text import TfidfVectorizer

import string

import pandas as pd
import numpy as np

course = pd.read_csv('kmeans.csv', sep=';')

# drop course_id, tokenized, clean_document
course.drop(['course_id', 'tokenized', 'clean_document'], axis=1, inplace=True)

model = load("kmeans.joblib")
vectorizer = load("vectorizer.joblib")

def kmean_predict(course_name, level):
    course_name = nlp(course_name)
    course_name = [token.text for token in course_name if token.text not in STOP_WORDS]
    course_name = [token for token in course_name if token not in string.punctuation]
    clean_course_name = " ".join([token.lower() for token in course_name])
    course_name_vector = vectorizer.transform([clean_course_name])
    predicted_cluster = model.predict(course_name_vector)[0]
    similar_courses = course[course['cluster'] == predicted_cluster]
    similar_courses = similar_courses[similar_courses['level'] == level]
    return similar_courses.sample(n=5, replace=True)


gr.Interface(kmean_predict,inputs=["text",gr.Dropdown(['All Levels', 'Beginner', 'Intermediate', 'Expert'])], outputs=["dataframe"]).launch()