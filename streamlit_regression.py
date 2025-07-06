import tensorflow as tf
import pickle
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import pandas as pd
import streamlit as st
import numpy as np

# Load model and preprocessing tools
model = tf.keras.models.load_model('regression_model.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app UI
st.title('Salary Prediction')

# Input fields
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 100)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])
exited = st.selectbox('Exited', [0, 1])

# Prepare input DataFrame
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Geography': [geography],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited': [exited]
})

# One-hot encode Geography
geo_encoded = onehot_encoder_geo.transform(input_data[['Geography']])
# Convert sparse matrix to dense array
geo_encoded = geo_encoded.toarray()
# Get expected geo column names
geo_feature_names = onehot_encoder_geo.get_feature_names_out(['Geography'])
# Create a dataframe for the encoded geography
geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo_feature_names)
# Ensure all expected geo columns exist
for col in geo_feature_names:
    if col not in geo_encoded_df.columns:
        geo_encoded_df[col] = 0

# Reorder geo columns to ensure alignment
geo_encoded_df = geo_encoded_df[geo_feature_names]

# Final input: drop raw Geography, add encoded
input_data = pd.concat([input_data.drop('Geography', axis=1), geo_encoded_df], axis=1)



# Convert column names to strings (to avoid sklearn feature name error)
input_data.columns = input_data.columns.astype(str)


# Match training column order if scaler has feature names
try:
    input_data = input_data[scaler.feature_names_in_]
except AttributeError:
    st.warning("Scaler does not have feature names. Proceeding as-is.")

# Scale the input
input_data_scaled = scaler.transform(input_data)

# Predict
try:
    predicted_salary = model.predict(input_data_scaled)
    predicted_salary = predicted_salary[0][0]
    st.success(f'Predicted Salary: ${predicted_salary:,.2f}')
except Exception as e:
    st.error(f"Prediction failed: {e}")
