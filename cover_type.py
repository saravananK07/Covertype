import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import pickle

# Load the trained model
model = load_model(r'C:\Users\HP\Downloads\Covertype DNN\dnn_regression_model.h5')

# Load the scaler
with open(r'C:\Users\HP\Downloads\Covertype DNN\scaler_selected.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit App
st.title("Forest Cover Type Prediction")

st.header("Input Features")

# User inputs
elevation = st.number_input("Elevation", value=3000)
wilderness_area4 = st.number_input("Wilderness Area 4", value=0)
horizontal_distance_to_roadways = st.number_input("Horizontal Distance to Roadways", value=100)
horizontal_distance_to_fire_points = st.number_input("Horizontal Distance to Fire Points", value=200)
wilderness_area2 = st.selectbox("Wilderness Area 2", [0, 1])

# Prepare input data
input_data = np.array([[elevation, wilderness_area4, horizontal_distance_to_roadways,
                        horizontal_distance_to_fire_points, wilderness_area2]])

# Scale input data
input_data_scaled = scaler.transform(input_data)

# Predict and display the result
if st.button("Predict"):
    prediction_probs = model.predict(input_data_scaled)
    predicted_class = (prediction_probs > 0.5).astype(int)

    st.subheader("Prediction")
    if predicted_class[0] == 1:
        st.write("The input data is classified as 'Spruce/Fir'.")
    else:
        st.write("The input data is classified as 'Not Spruce/Fir'.")

