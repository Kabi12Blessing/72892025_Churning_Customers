import streamlit as st
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load your Keras model
model_path = "C:\\Users\\hanna\\Documents\\Customer_churn\\best_model.joblib"

from joblib import load


# Load the model from the saved file
best_model = load(model_path)

# You can use loaded_model for predictions



# Load the scalers
num_features_scaler_path = "C:\\Users\\hanna\\Documents\\Customer_churn\\scaler_model.joblib"


num_features_scaler = joblib.load(num_features_scaler_path)

label_path= "C:\\Users\\hanna\\Documents\\Customer_churn\\label_encoder.joblib"
label_encoder = load(label_path)
st.title("Neural network model using TensorFlow's Keras API for Churn prediction")

def main():
    tenure = st.number_input('Tenure', min_value=0)
    monthly_charges = st.number_input('Monthly Charges', min_value=0.0)
    total_charges = st.number_input('Total Charges', min_value=0.0)
    online_security = st.selectbox('Online Security', ['Yes', 'No', 'No internet service'])
    tech_support = st.selectbox('Tech Support', ['Yes', 'No', 'No internet service'])
    payment_method_selection = st.selectbox('Select Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer'])  # Replace with actual options
    internet_service_selection = st.selectbox('Select Internet Service', ['DSL', 'Fiber optic'])  # Replace with actual options

    if st.button("Predict"):
    # Define your categorical columns here
        categorical_columns = [online_security, tech_support]

        categorical_features_encoded = label_encoder.fit_transform(categorical_columns)  
        
        # Handle 'PaymentMethod' and 'InternetService' as one-hot encoded features directly
        data = {
            'PaymentMethod': [payment_method_selection],
            'InternetService': [internet_service_selection ]
        }
        df = pd.DataFrame(data)
        encoded_df = pd.get_dummies(df, columns=['PaymentMethod', 'InternetService'])

        # Flatten the encoded_df to make it a 1D array
        flattened_encoded_df = encoded_df.values.flatten()

        # Combine all features, including label-encoded columns and one-hot encoded columns
        input_features = np.concatenate([
            np.array([tenure, monthly_charges, total_charges, *categorical_features_encoded]),
            flattened_encoded_df
        ]).reshape(1, -1)
        # Scale numerical features
    

        num_features_scaled = num_features_scaler.transform(input_features)
        print("Number of features in num_features_scaler:", len(num_features_scaler.scale_))
        prediction = best_model.predict(num_features_scaled)
        label_mapping = {1: 'Yes', 0: 'No'}
        predicted_churn_label = int(prediction[0])
        predicted_churn = label_mapping[predicted_churn_label]
        st.write(f"Predicted Churn: {predicted_churn}")

if __name__ == "__main__":
    main()
