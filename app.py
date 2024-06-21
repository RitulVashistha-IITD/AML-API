import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Load the trained model and scaler
pickle_in = open("stacked_model.pkl", "rb")
model = pickle.load(pickle_in)

scaler = pickle.load(open("scaler.pkl", 'rb'))

# List of feature names for scaling
feature_names = ['CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']

def preprocess_data(data):
    # Drop unnecessary columns
    data = data.drop(['step', 'isFlaggedFraud'], axis=1)
    
    # Fill missing values with median
    data = data.fillna(data.select_dtypes(include=['number']).median())
    
    # One-hot encode categorical columns
    data = pd.get_dummies(data, columns=['type'])

    # Ensure all columns present in the training set are in the data
    for col in feature_names:
        if col not in data.columns:
            data[col] = 0
    
    # Reorder columns to match the order in feature_names
    data = data[feature_names]
    
    return data

def scale_numerical_cols(data):
    # Extract numerical columns
    numerical_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    numerical_data = data[numerical_cols]
    
    # Scale numerical columns
    scaled_data = scaler.transform(numerical_data)
    
    # Replace the original numerical columns with the scaled ones, in the correct order
    data[numerical_cols] = scaled_data
    
    return data

def app():
    st.title('Money Laundering Detection')
    st.write('Identifying fraudulent transactions using a machine learning model')
    
    # File uploader for the input 
    uploaded_file = st.file_uploader('Upload your data:', type=['csv'])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        
        # Preprocess the data
        data_processed = preprocess_data(data)
        
        # Scale numerical columns
        data_scaled = scale_numerical_cols(data_processed)
        
        # Ensure data has the correct number of features
        if data_scaled.shape[1] != len(feature_names):
            st.error(f"Expected {len(feature_names)} features, but got {data_scaled.shape[1]} instead.")
            return
        
        # Make predictions for data
        predictions = model.predict(data_scaled)
        
        # Convert predictions to fraud/not fraud
        predictions = ['Fraud' if p == 1 else 'Not Fraud' for p in predictions]
        
        # Prepare result dataframe
        result_df = pd.DataFrame({
            'Row': range(1, len(predictions) + 1),
            'NameOrig': data['nameOrig'],
            'NameDest': data['nameDest'],
            'Prediction': predictions,
            'Amount': data['amount']
        })
        
        # Show the table with predictions
        st.subheader('Predictions')
        st.table(result_df)
        
        # Show bar plot for fraud vs non-fraud transactions
        st.subheader('Fraud vs Non-Fraud Transactions')
        fraud_counts = result_df['Prediction'].value_counts()
        st.bar_chart(fraud_counts)
        
        # Show KDE plot or histogram for transaction amounts
        st.subheader('Transaction Amount Distribution')
        fig, ax = plt.subplots()
        sns.histplot(result_df[result_df['Prediction'] == 'Fraud']['Amount'], label='Fraud', kde=True, ax=ax)
        sns.histplot(result_df[result_df['Prediction'] == 'Not Fraud']['Amount'], label='Not Fraud', kde=True, ax=ax)
        ax.set_xlabel('Amount')
        ax.set_ylabel('Count')
        ax.legend()
        st.pyplot(fig)
        

if __name__=='__main__':
    app()



# import pickle
# import numpy as np
# import pandas as pd
# import streamlit as st
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score

# # Load the trained model and scaler
# pickle_in = open("stacked_model.pkl", "rb")
# model = pickle.load(pickle_in)

# scaler = pickle.load(open("scaler.pkl", 'rb'))

# # List of feature names for scaling
# feature_names = ['CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER','amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest'
#                  ]

# def preprocess_data(data):
#     # Drop unnecessary columns
#     data = data.drop(['step', 'nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1)
    
#     # Fill missing values with median
#     data = data.fillna(data.select_dtypes(include=['number']).median())
    
#     # One-hot encode categorical columns
#     data = pd.get_dummies(data, columns=['type'])

#     # Ensure all columns present in the training set are in the data
#     for col in feature_names:
#         if col not in data.columns:
#             data[col] = 0
    
#     # Reorder columns to match the order in feature_names
#     data = data[feature_names]
    
#     return data

# def scale_numerical_cols(data):
#     # Extract numerical columns
#     numerical_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
#     numerical_data = data[numerical_cols]
    
#     # Scale numerical columns
#     scaled_data = scaler.transform(numerical_data)
    
#     # Replace the original numerical columns with the scaled ones, in the correct order
#     data[numerical_cols] = scaled_data
    
#     return data

# def app():
#     st.title('Money Laundering Detection')
#     st.write('Identifying fraudulent transactions using a machine learning model')
    
#     # File uploader for the input 
#     uploaded_file = st.file_uploader('Upload your data:', type=['csv'])
    
#     if uploaded_file is not None:
#         data = pd.read_csv(uploaded_file)

        
#         # Preprocess the data
#         data_processed = preprocess_data(data)
        
#         # Scale numerical columns
#         data_scaled = scale_numerical_cols(data_processed)
        
#         # Ensure data has the correct number of features
#         if data_scaled.shape[1] != len(feature_names):
#             st.error(f"Expected {len(feature_names)} features, but got {data_scaled.shape[1]} instead.")
#             return
        
#         # Make predictions for data
#         predictions = model.predict(data_scaled)
        
#         # Convert predictions to fraud/not fraud
#         predictions = ['Fraud' if p == 1 else 'Not Fraud' for p in predictions]


        
#         # Prepare result dataframe
        
#         result_df = pd.DataFrame({
#             'Row': range(1, len(predictions) + 1),
#             'NameOrig': data['nameOrig'],
#             'NameDest': data['nameDest'],
#             'Prediction': predictions
#         })

 
        
#         # Show result
#         st.table(result_df)
        

# if __name__=='__main__':
#     app()

