import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Load the diabetes dataset
diabetes_dataset = pd.read_csv('diabetes.csv')

# Display the dataset using Streamlit
st.title("Diabetes Prediction App")
st.subheader("Diabetes Dataset")
st.dataframe(diabetes_dataset)

# Summary statistics
st.subheader("Dataset Summary")
st.write(diabetes_dataset.describe())

# Class distribution
st.subheader("Class Distribution")
st.write(diabetes_dataset['Outcome'].value_counts())

# Mean values by Outcome
st.subheader("Mean Values by Outcome")
st.write(diabetes_dataset.groupby('Outcome').mean())

# Separating the data and labels
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

# Standardize the data
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    standardized_data, Y, test_size=0.2, stratify=Y, random_state=2)

# Train the SVM classifier
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Training data accuracy
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

# Test data accuracy
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

st.subheader("Model Evaluation")
st.write(f'Accuracy score on the training data: {training_data_accuracy:.2f}')
st.write(f'Accuracy score on the test data: {test_data_accuracy:.2f}')

# Input data for prediction
st.sidebar.header("Diabetes Prediction")
st.sidebar.subheader("Enter New Data")

input_data = st.sidebar.text_input(
    "Enter data as a comma-separated string (e.g., 0, 166, 72, 19, 175, 25.8, 0.587, 51)")

if st.sidebar.button("Predict"):
    input_data = input_data.split(',')
    input_data = [float(x) for x in input_data]
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    std_data = scaler.transform(input_data_reshaped)
    prediction = classifier.predict(std_data)

    if prediction[0] == 0:
        st.sidebar.write('The person is not diabetic')
    else:
        st.sidebar.write('The person is diabetic')
