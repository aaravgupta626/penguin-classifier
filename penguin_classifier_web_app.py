import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier

csv_file = 'penguin.csv'
df = pd.read_csv(csv_file)

df.head()

df = df.dropna()

df['label'] = df['species'].map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo':2})


df['sex'] = df['sex'].map({'Male':0,'Female':1})

df['island'] = df['island'].map({'Biscoe': 0, 'Dream': 1, 'Torgersen':2})


X = df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)


svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)


rf_clf = RandomForestClassifier(n_jobs = -1)
rf_clf.fit(X_train, y_train)

def prediction(model, island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex):
	predicted = model.predict([[island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex]])
	if predicted[0] == 0:
		return 'Adelie'
	elif predicted[0] == 1:
		return 'Chinstrap'
	else:
		return 'Gentoo'

st.title("Penguin Species Classification Web Application")
bill_length_mm = st.sidebar.slider('Bill Length (mm)', float(df['bill_length_mm'].min()), float(df['bill_length_mm'].max()))
bill_depth_mm = st.sidebar.slider('Bill Depth (mm)', float(df['bill_depth_mm'].min()), float(df['bill_depth_mm'].max()))
flipper_length_mm = st.sidebar.slider('Flipper Length (mm)', float(df['flipper_length_mm'].min()), float(df['flipper_length_mm'].max()))
body_mass_g = st.sidebar.slider('Body Mass (g)', float(df['body_mass_g'].min()), float(df['body_mass_g'].max()))

sex = st.sidebar.selectbox('Choose Sex:', ['Male', 'Female'])
if sex == 'Male':
	sex = 0
else:
	sex = 1

island = st.sidebar.selectbox('Choose Island:', ['Biscoe', 'Dream', 'Torgersen'])
if island == 'Biscoe':
	island = 0
elif island == 'Dream':
	island = 1
else:
	island = 2

classifier = st.sidebar.selectbox('Select a classifier:', ('Logistic Regression', 'Support Vector Machine', 'Random Forest Classifier'))

button = st.sidebar.button('Predict')

if button:
	if classifier == 'Logistic Regression':
		predict = prediction(log_reg, island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex)
		score = log_reg.score(X_train, y_train)
	elif classifier == 'Support Vector Machine':
		predict = prediction(svc_model, island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex)
		score = svc_model.score(X_train, y_train)
	else:
		predict = prediction(rf_clf, island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex)
		score = rf_clf.score(X_train, y_train)

	st.write('The species predicted is:', predict)
	st.write('The accuracy score of this model is:', score)
