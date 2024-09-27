import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pickle

mpl.style.use('ggplot')

car = pd.read_csv('Cleaned_Car_data.csv')

st.title("Car Price Prediction")

# Sidebar options to choose inputs
st.sidebar.header("Select Input Parameters")

# Create dropdown menus for Company and Car Name
companies = car['company'].unique()
selected_company = st.sidebar.selectbox("Company", companies)

car_names = car['name'].unique()
selected_car_name = st.sidebar.selectbox("Car Name", car_names)

year = st.sidebar.number_input("Year", 1900, 2023, 2019)
kms_driven = st.sidebar.number_input("Kilometers Driven", 0)
fuel_type = st.sidebar.selectbox("Fuel Type", car['fuel_type'].unique())

input_data = pd.DataFrame({'name': [selected_car_name], 'company': [selected_company], 'year': [
                          year], 'kms_driven': [kms_driven], 'fuel_type': [fuel_type]})

# Display the input data
st.subheader("Input Data")
st.write(input_data)

# Load the trained model
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))

# Predict car price
if st.button("Predict Car Price"):
    prediction = model.predict(input_data)
    st.subheader("Predicted Car Price")
    st.write(f"Rs. {prediction[0]:,.2f}")

if st.checkbox("Show Cleaned Car Data"):
    st.subheader("Cleaned Car Data")
    st.write(car)

# Company vs. Price
if st.checkbox("Boxplot - Company vs. Price"):
    st.subheader("Boxplot - Company vs. Price")
    fig, ax = plt.subplots(figsize=(15, 7))
    sns.boxplot(x='company', y='Price', data=car, ax=ax)
    st.pyplot(fig)

# Year vs. Price
if st.checkbox("Swarmplot - Year vs. Price"):
    st.subheader("Swarmplot - Year vs. Price")
    fig, ax = plt.subplots(figsize=(20, 10))
    sns.swarmplot(x='year', y='Price', data=car, ax=ax)
    st.pyplot(fig)
