import numpy as np
import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

st.write(''' # Predicción de Vinos 🍷 ''')
st.image("vino.jpeg", caption="El vino es la única obra de arte que se puede beber. — Louis Fernando Olaverri")

st.header('Datos de captura')

def user_input_features():
    # Entradas de la interfaz
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

    user_input_data = {
        'fixed acidity': [fixed], 'volatile acidity': [volatile], 'citric acid': [citric],
        'residual sugar': [residual], 'chlorides': [chlorides], 'free sulfur dioxide': [free_sulfur],
        'total sulfur dioxide': [total_sulfur], 'density': [density], 'pH': [pH],
        'sulphates': [sulphates], 'alcohol': [alcohol]
    }
    return pd.DataFrame(user_input_data, index=[0])

df = user_input_features()

# Leemos tu archivo (como usaste df.to_csv por defecto en Colab, está separado por COMAS)
vinos = pd.read_csv('vino_rojo.csv', encoding='latin-1')

X = vinos.drop(columns=['quality'])

# --- CORRECCIÓN MATEMÁTICA VITAL ---
# Como tu archivo ya contiene 0 y 1, lo asignamos DIRECTAMENTE sin condicionantes
Y = vinos['quality']
 
# Entrenamos al clasificador con tus parámetros reales optimizados de tu GridSearch
classifier = DecisionTreeClassifier(max_depth=6, criterion='gini', min_samples_leaf=10, max_features=None, random_state=0)
classifier.fit(X, Y)
 
# Predicción basada en la pantalla
prediction = classifier.predict(df)

st.subheader('Predicción')
 
if prediction[0] == 0:
    st.error('El modelo predice: **Baja Calidad** 🍇')
elif prediction[0] == 1:
    st.success('El modelo predice: **Buena Calidad** ⭐🍷')
