import numpy as np
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier


st.write(''' # Predicción del Vinos ''')
st.image("vino.jpeg", caption="El vino es la única obra de arte que se puede beber. — Louis Fernando Olaverri")

st.header('Datos de captura')

def user_input_features():
  # Entrada
  fixed = st.number_input('Acidez fija :', min_value=4.0, max_value=16.0, value=7.4, step=0.1)
  volatile = st.number_input('Acidez volátil :', min_value=0.1, max_value=2.0, value=0.70, step=0.01)
  citric = st.number_input('Ácido cítrico :', min_value=0.0, max_value=1.0, value=0.00, step=0.01)
  residual = st.number_input('Azúcar residual :', min_value=0.9, max_value=15.0, value=1.9, step=0.1)
  chlorides = st.number_input('Cloruros :', min_value=0.01, max_value=0.6, value=0.076, step=0.001)
  free_sulfur = st.number_input('Dióxido de azufre libre :', min_value=1.0, max_value=72.0, value=11.0, step=1.0)
  total_sulfur = st.number_input('Dióxido de azufre total :', min_value=6.0, max_value=289.0, value=34.0, step=1.0)
  density = st.number_input('Densidad :', min_value=0.990, max_value=1.005, value=0.9978, step=0.0001)
  pH = st.number_input('Nivel de pH:', min_value=2.7, max_value=4.0, value=3.51, step=0.01)
  sulphates = st.number_input('Sulfatos :', min_value=0.3, max_value=2.0, value=0.56, step=0.01)
  alcohol = st.number_input('Grados de alcohol :', min_value=8.0, max_value=15.0, value=9.4, step=0.1)

  user_input_data = {'fixed acidity': [fixed],
        'volatile acidity': [volatile],
        'citric acid': [citric],
        'residual sugar': [residual],
        'chlorides': [chlorides],
        'free sulfur dioxide': [free_sulfur],
        'total sulfur dioxide': [total_sulfur],
        'density': [density],
        'pH': [pH],
        'sulphates': [sulphates],
        'alcohol': [alcohol]}

  features = pd.DataFrame(user_input_data, index=[0])
  return features

df = user_input_features()

vinos = pd.read_csv('vino_rojo.csv', encoding='latin-1', sep=';')

X = vinos.drop(columns=['quality'])
 
Y = (vinos['quality'] >= 6).astype(int)
 
classifier = DecisionTreeClassifier(max_depth=8, criterion='entropy', min_samples_leaf=10, max_features=7, random_state=0)
classifier.fit(X, Y)
 
prediction = classifier.predict(df)

st.subheader('Predicción')
 
if prediction[0] == 0:
  st.error('El modelo predice: **Baja Calidad** ')
elif prediction[0] == 1:
  st.success('El modelo predice: **Buena Calidad** ')
else:
  st.write('Sin predicción')
