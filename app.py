import streamlit as st
import pickle
import pandas as pd
import numpy as np

MAX_VALUE = 10_000_000
MIN_VALUE = 0

@st.cache_data
def load_artifacts():
    """Load ML artifacts with error handling"""
    try:
        artifacts = {}
        files = [
            "model", "scaler", "label_encoders",
            "feature_names", "numerical_cols", "categorical_cols"
        ]
        
        for file in files:
            with open(f"models/{file}.pkl", "rb") as f:
                artifacts[file] = pickle.load(f)
        
        return (
            artifacts["model"],
            artifacts["scaler"],
            artifacts["label_encoders"],
            artifacts["feature_names"],
            artifacts["numerical_cols"],
            artifacts["categorical_cols"]
        )
    except Exception as e:
        st.error(f"Error loading artifacts: {str(e)}")
        st.stop()

model, scaler, encoders, feature_names, num_cols, cat_cols = load_artifacts()

st.title("🏦 Loan Approval Predictor")

with st.form("loan_form"):
    col1, col2 = st.columns(2)
    
    inputs = {}
    with col1:
        inputs["no_of_dependents"] = st.number_input("Dependents", 0, 10, 0)
        inputs["education"] = st.selectbox("Education", ["Graduate", "Not Graduate"])
        inputs["self_employed"] = st.selectbox("Employment", ["Yes", "No"])
        inputs["income_annum"] = st.number_input("Annual Income ($)", 
                                               MIN_VALUE, MAX_VALUE, 50_000)
        inputs["loan_amount"] = st.number_input("Loan Amount ($)", 
                                              MIN_VALUE, MAX_VALUE, 10_000)
    
    with col2:
        inputs["loan_term"] = st.slider("Term (months)", 6, 360, 12)
        inputs["cibil_score"] = st.slider("Credit Score", 300, 900, 700)
        inputs["residential_assets_value"] = st.number_input("Home Assets ($)", 
                                                           MIN_VALUE, MAX_VALUE, 0)
        inputs["commercial_assets_value"] = st.number_input("Business Assets ($)", 
                                                          MIN_VALUE, MAX_VALUE, 0)
        inputs["luxury_assets_value"] = st.number_input("Luxury Assets ($)", 
                                                      MIN_VALUE, MAX_VALUE, 0)
        inputs["bank_asset_value"] = st.number_input("Bank Balance ($)", 
                                                   MIN_VALUE, MAX_VALUE, 0)
    
    submitted = st.form_submit_button("Check Approval")
    
    if submitted:
        try:
            input_df = pd.DataFrame([inputs])
            
            # Process categorical features
            for col in cat_cols:
                input_df[col] = input_df[col].str.strip().str.lower()
                le = encoders[col]
                
                if input_df[col].iloc[0] not in le.classes_:
                    st.error(f"Invalid value for {col}: {input_df[col].iloc[0]}")
                    st.stop()
                    
                input_df[col] = le.transform(input_df[col])
            
            # Scale numerical features
            scaled_numerical = scaler.transform(input_df[num_cols])
            processed_numerical = pd.DataFrame(scaled_numerical, columns=num_cols)
            
            # Combine features
            processed_df = pd.concat([
                processed_numerical,
                input_df[cat_cols].reset_index(drop=True)
            ], axis=1)[feature_names]
            
            # Predict and display
            prediction = model.predict(processed_df)[0]
            result = "✅ Approved" if prediction == 1 else "❌ Denied"
            
            # Display results
            st.subheader("Result")
            st.success(result)
            
            # Show effects without DeltaGenerator output
            if prediction == 1:
                st.balloons()
            else:
                st.snow()
            
            # Disclaimer
            st.caption("Note: Predictive model only - not for real financial decisions")

        except Exception as e:
            st.error(f"⚠️ Processing error: {str(e)}")
            st.info("Please check your inputs and try again")