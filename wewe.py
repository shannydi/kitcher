import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Load your RandomForest model
model = joblib.load("random_forest_regressor.pkl")

# Load your dataset
data = pd.read_csv("algae.csv")

# Define the list of variables
variables = ['Light', 'Nitrate', 'Iron', 'Phosphate', 'Temperature', 'pH', 'CO2']

# Set up page configurations
st.set_page_config(page_title="Algae Population Predictor", layout="wide")

# Custom CSS for green theme
st.markdown("""
    <style>
    .main {
        background-color: #e8f5e9;
    }
    </style>
    """, unsafe_allow_html=True)

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a Page:", ["Home Page", "Analysis Page", "Prediction Page"])

# Page 1: Home Page
if page == "Home Page":
    st.title("Understanding Algal Bloom")
    st.markdown("""
    Algal blooms are rapid increases in the population of algae in aquatic systems. 
    These events can significantly impact water quality, ecosystem stability, 
    and biodiversity. Factors like light availability, nutrient levels, and water 
    temperature play crucial roles in the proliferation of algae.
    """)
    # Display the image
    image = Image.open("algae_image.jpg")
    st.image(image, caption="Algal Bloom", use_column_width=True)

# Page 2: Analysis Page
elif page == "Analysis Page":
    st.title("Data Analysis of Environmental Factors")
    st.markdown("Explore the relationships between different environmental factors that influence algal populations.")
    
    # Dropdown for variable selection
    col1, col2 = st.columns(2)
    with col1:
        var1 = st.selectbox("Select the first variable", variables)
    with col2:
        var2 = st.selectbox("Select the second variable", variables, index=1)
    
    # Button to generate plot
    if st.button("Generate Plot"):
       fig, ax = plt.subplots()
       sns.kdeplot(x=var1, y=var2, data=data, ax=ax, fill=True, thresh=0, levels=100, cmap="viridis")
       plt.xlabel(var1)
       plt.ylabel(var2)
       plt.title(f"Density Plot of {var1} vs {var2}")
       st.pyplot(fig)

# Page 3: Prediction Page
elif page == "Prediction Page":
    st.title("Predict Algal Population")
    st.markdown("Input environmental variables to predict algal population.")

    # Input fields for features
    inputs = {feature: st.number_input(f"{feature}:", float(data[feature].min()), float(data[feature].max()), step=0.01) for feature in variables}

    # Predict button
    if st.button("Predict"):
        # Make prediction
        features = np.array([list(inputs.values())]).reshape(1, -1)
        prediction = model.predict(features)
        st.success(f"The predicted algae population is {prediction[0]:.2f}")