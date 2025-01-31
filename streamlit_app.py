import streamlit as st
import joblib
import os

# Cargar el modelo
model = joblib.load('svm_model.pkl')

# Título de la aplicación
st.title("Titanic Survival Predictor")

# Entradas del usuario
st.write("Ingresa los datos para predecir si sobrevivirías en el Titanic:")
pclass = st.selectbox("Clase (Pclass)", [1, 2, 3])
sex = st.selectbox("Sexo", ["Male", "Female"])
age = st.number_input("Edad", min_value=0, max_value=100)
sibsp = st.number_input("Número de hermanos/cónyuges (SibSp)", min_value=0)
parch = st.number_input("Número de padres/hijos (Parch)", min_value=0)
# Input para la tarifa como texto
fare_input = st.text_input("Tarifa (Fare)", value="0.0")

# Validar y convertir la tarifa
try:
    fare = float(fare_input.replace(",", "."))  # Reemplazar coma por punto y convertir a float
    print(fare)
except ValueError:
    st.error("Por favor, ingresa un número válido para la tarifa.")
    fare = 0.0


embarked = st.selectbox("Puerto de embarque (Embarked)", ["S", "C", "Q"])

# Convertir entradas a formato numérico
sex = 0 if sex == "Male" else 1
embarked = 0 if embarked == "S" else 1 if embarked == "C" else 2

# Predecir si el pasajero sobreviviría
if st.button("Predecir"):
    features = [pclass, sex, age, sibsp, parch, fare, embarked]
    prediction = model.predict([features])
    st.write("Sobrevivirías" if prediction[0] == 1 else "No sobrevivirías")
