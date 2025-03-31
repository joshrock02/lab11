import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import load_model

# Load preprocessing pipeline
with open("preprocessing_pipeline.pkl", "rb") as f:
    preprocessor = pickle.load(f)

# Load trained model
model = load_model("tf_bridge_model.h5")

# Streamlit app
def main():
    st.title("Bridge Load Capacity Prediction")
    
    # User input fields
    span_ft = st.number_input("Span (ft)", min_value=10, max_value=1000, step=1)
    deck_width_ft = st.number_input("Deck Width (ft)", min_value=10, max_value=100, step=1)
    age_years = st.number_input("Age (years)", min_value=0, max_value=200, step=1)
    num_lanes = st.number_input("Number of Lanes", min_value=1, max_value=10, step=1)
    condition_rating = st.slider("Condition Rating (1-5)", min_value=1, max_value=5, step=1)
    material = st.selectbox("Material", ["Steel", "Concrete", "Composite"])
    
    # Create input dataframe
    input_data = pd.DataFrame({
        'Span_ft': [span_ft],
        'Deck_Width_ft': [deck_width_ft],
        'Age_Years': [age_years],
        'Num_Lanes': [num_lanes],
        'Condition_Rating': [condition_rating],
        'Material': [material]
    })
    
    # Preprocess input
    input_data_processed = preprocessor.transform(input_data)
    
    # Predict
    if st.button("Predict Load Capacity"):
        prediction = model.predict(input_data_processed)
        st.write(f"### Predicted Maximum Load Capacity: {prediction[0][0]:.2f} tons")
    
if __name__ == "__main__":
    main()

