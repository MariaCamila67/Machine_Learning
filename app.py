import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Cargar el modelo y el scaler
linear_model = joblib.load('linear_regression_model.joblib')
scaler = joblib.load('minmax_scaler.joblib')

st.title('Predicción de Precios de Viviendas en California')
st.write('Introduce los valores de las características para predecir el precio medio de una vivienda.')

# Entradas de usuario
med_inc = st.number_input('Ingreso Medio por Bloque (MedInc)', min_value=0.0, value=3.0, step=0.1)
ave_rooms = st.number_input('Número Promedio de Habitaciones (AveRooms)', min_value=0.0, value=5.0, step=0.1)
ave_bedrms = st.number_input('Número Promedio de Dormitorios (AveBedrms)', min_value=0.0, value=1.0, step=0.1)
ave_occup = st.number_input('Ocupación Promedio del Hogar (AveOccup)', min_value=0.0, value=2.5, step=0.1)
latitude = st.number_input('Latitud (Latitude)', min_value=32.0, max_value=42.0, value=37.0, step=0.1)

if st.button('Predecir Precio'):
    # Crear un DataFrame con los datos de entrada
    input_data = pd.DataFrame([{
        'MedInc': med_inc,
        'AveRooms': ave_rooms,
        'AveBedrms': ave_bedrms,
        'AveOccup': ave_occup,
        'Latitude': latitude
    }])

    # Escalar los datos de entrada
    # Asegurarse de que las columnas coincidan con las utilizadas durante el entrenamiento
    X_cols = ['MedInc', 'AveRooms', 'AveBedrms', 'AveOccup', 'Latitude']
    input_scaled = scaler.transform(input_data[X_cols])
    
    # Realizar la predicción
    prediction = linear_model.predict(input_scaled)
    
    # Mostrar la predicción
    st.success(f'El precio medio predicho de la vivienda es: ${prediction[0]*100000:.2f}')
