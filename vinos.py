import numpy as np
import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Título y presentación de la app
st.write(''' # Predicción de Vinos 🍷 ''')
st.image("vino.jpeg", caption="El vino es la única obra de arte que se puede beber. — Louis Fernando Olaverri")

st.header('Datos de captura')

# Función para recolectar las entradas del usuario desde la interfaz
def user_input_features():
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
        'fixed acidity': [fixed],
        'volatile acidity': [volatile],
        'citric acid': [citric],
        'residual sugar': [residual],
        'chlorides': [chlorides],
        'free sulfur dioxide': [free_sulfur],
        'total sulfur dioxide': [total_sulfur],
        'density': [density],
        'pH': [pH],
        'sulphates': [sulphates],
        'alcohol': [alcohol]
    }

    features = pd.DataFrame(user_input_data, index=[0])
    return features

# Guardamos los datos de entrada del usuario en el DataFrame 'df'
df = user_input_features()

# 1. Cargamos el dataset autodetectando si el separador es coma (,) o punto y coma (;)
vinos = pd.read_csv('vino_rojo.csv', encoding='latin-1', sep=None, engine='python')

# 2. Separamos las características (X) e Y (calidad binaria)
X = vinos.drop(columns=['quality'])
Y = (vinos['quality'] >= 6).astype(int)
 
# 3. Entrenamos el clasificador optimizado
classifier = DecisionTreeClassifier(max_depth=8, criterion='entropy', min_samples_leaf=10, max_features=None, random_state=0)
classifier.fit(X, Y)

# 4. Inicializamos las variables de sesión para la memoria de la app
if 'modo_demo' not in st.session_state:
    st.session_state['modo_demo'] = False

# Opciones de la barra lateral
st.sidebar.header("Opciones de Demostración")

# 5. Botón para activar el modo demostración de vino excelente
if st.sidebar.button("Cargar Vino Excelente Garantizado 🍷"):
    st.session_state['modo_demo'] = True

# Botón extra para regresar a evaluar manualmente con los números de la pantalla
if st.sidebar.button("Regresar a Modo Manual ⚙️"):
    st.session_state['modo_demo'] = False

# 6. Evaluamos la predicción final dependiendo del modo activo
if st.session_state['modo_demo']:
    # Si la demo está activa, inyectamos los datos del vino excelente
    vino_premiado = pd.DataFrame({
        'fixed acidity': [10.3], 'volatile acidity': [0.32], 'citric acid': [0.45],
        'residual sugar': [6.4], 'chlorides': [0.073], 'free sulfur dioxide': [5.0],
        'total sulfur dioxide': [13.0], 'density': [0.9976], 'pH': [3.23],
        'sulphates': [0.82], 'alcohol': [12.6]
    }, index=[0])
    prediction = classifier.predict(vino_premiado)
    st.sidebar.success("¡Modo Demo: Vino Excelente Cargado!")
else:
    # Si no, predice normalmente basándose en las entradas de la pantalla
    prediction = classifier.predict(df)

# Despliegue de la predicción final en la interfaz principal
st.subheader('Predicción')
 
if prediction[0] == 0:
    st.error('El modelo predice: **Baja Calidad** 🍇')
elif prediction[0] == 1:
    st.success('El modelo predice: **Buena Calidad** ⭐🍷')
else:
    st.write('Sin predicción')
