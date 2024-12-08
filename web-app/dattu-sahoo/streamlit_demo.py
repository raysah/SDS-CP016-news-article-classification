import streamlit as st
import numpy as np
import pickle


# Interface
st.markdown('## Iris Species Prediction')
sepal_length = st.number_input('sepal length (cm)')
sepal_width = st.number_input('sepal width (cm)')
petal_length = st.number_input('petal length (cm)')
petal_width = st.number_input('petal width (cm)')

# Predict button
if st.button('Predict'):
    # loading in the model to predict on the data
    # model = joblib.load('models/iris_model.pkl')

    filename = r'/ Users / dots / PycharmProjects / SDS - CP016 - news - article - classification / web - app / dattu - sahoo / iris_model.pkl'
    pickle_in = open(filename, 'rb')
    classifier = pickle.load(pickle_in)
    X = np.array([sepal_length, sepal_width, petal_length, petal_width])
    if any(X <= 0):
        st.markdown('### Inputs must be greater than 0')
    else:
        st.markdown(f'### Prediction is {classifier.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]}')
