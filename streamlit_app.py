import streamlit as st
import joblib

# Cargar el modelo
model = joblib.load('svm_model.pkl')

# Nombre de la app
st.title("Titanic Survival Predictor")

# Inputs del usuario
st.write("Ingresa los datos para predecir si sobrevivirías en el Titanic:")
pclass = st.selectbox("Clase (Pclass)", [1, 2, 3])
sex = st.selectbox("Sexo", ["Male", "Female"])
age = st.number_input("Edad", min_value=0, max_value=100)
sibsp = st.number_input("Número de hermanos/cónyuges (SibSp)", min_value=0)
parch = st.number_input("Número de padres/hijos (Parch)", min_value=0)

# Input para la tarifa como texto
fare_input = st.text_input("Tarifa (Fare)", value="0.0")

# Valida y convierte la tarifa
try:
    fare = float(fare_input.replace(",", "."))  # Reemplaza la coma por punto (para que sea el formato adecuado) y lo convierte a float
except ValueError:
    st.error("Por favor, ingresa un número válido para la tarifa.")
    fare = 0.0

embarked = st.selectbox("Puerto de embarque (Embarked)", ["S", "C", "Q"])

# Convierte las entradas a formato numerico
sex = 0 if sex == "Male" else 1
embarked = 0 if embarked == "S" else 1 if embarked == "C" else 2

# Predice si el pasajero sobreviviría
if st.button("Predecir"):
    features = [pclass, sex, age, sibsp, parch, fare, embarked]
    
    # Comprueba los valores de entrada en la interfaz de Streamlit
    st.write("Valores de entrada para el modelo:")
    st.write(f"Pclass: {pclass}, Sex: {sex}, Age: {age}, SibSp: {sibsp}, Parch: {parch}, Fare: {fare}, Embarked: {embarked}")
    
    # Realiza la prediccion
    prediction = model.predict([features])
    
    # Muestra el resultado
    st.write("Sobrevivirías" if prediction[0] == 1 else "No sobrevivirías")
