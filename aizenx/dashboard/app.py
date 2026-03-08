import streamlit as st
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier


st.title("AizenX Explainability Dashboard")

data = load_breast_cancer()

X = data.data
y = data.target

model = RandomForestClassifier()
model.fit(X,y)

st.write("Model trained")

index = st.slider("Select sample",0,len(X)-1)

sample = X[index]

prediction = model.predict([sample])[0]

st.write("Prediction:", prediction)

st.write("Feature values")

for i,f in enumerate(data.feature_names):
    st.write(f, sample[i])