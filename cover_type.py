import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib

model = load_model('dnn_regression_model.h5')

scaler = joblib.load('scaler.joblib')  # Adjust the path if necessary

st.title("Forest Cover Type Prediction")


st.header("Input Features")

#Ensure all features are included

elevation = st.number_input("Elevation", value=3000)
wilderness_area4 = st.number_input("Wilderness Area 4", value=0)  # Adjusted to be binary (0 or 1)
horizontal_distance_to_roadways = st.number_input("Horizontal Distance to Roadways", value=100)
horizontal_distance_to_fire_points = st.number_input("Horizontal Distance to Fire Points", value=200)
wilderness_area2 = st.selectbox("Wilderness Area 2", [0, 1])  # Binary input (0 or 1)


input_data = np.array([[elevation, wilderness_area4, horizontal_distance_to_roadways,
                        horizontal_distance_to_fire_points, wilderness_area2]])
 
input_data_scaled = scaler.transform(input_data)  

st.write("Input data shape:", input_data.shape)
st.write("Scaler expects features:", scaler.n_features_in_)
 # 
if st.button("Predict"):
    prediction_probs = model.predict(input_data_scaled)
    predicted_class = np.argmax(prediction_probs, axis=1)

    st.subheader("Prediction")
    if predicted_class[0] == 1:
        st.write("The input data is classified as 'Spruce/Fir'.")
    else:
        st.write("The input data is classified as 'Not Spruce/Fir'.")

    
