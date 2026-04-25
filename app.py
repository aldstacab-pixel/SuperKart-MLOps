import os
import io
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "aldon-cabral/SuperKart-RF")

st.set_page_config(page_title="SuperKart Sales Forecast", layout="centered")

st.title("🛒 SuperKart Sales Forecasting")
st.markdown("Predict **Product_Store_Sales_Total** using our updated Random Forest pipeline.")

# -- Model loader -----------------------------------------------------------
@st.cache_resource
def load_model_from_hub():
    hf_model = os.environ.get("HF_MODEL") or "aldon-cabral/SuperKart-RF"
    hf_token = os.environ.get("HF_TOKEN")
    try:
        model_path = hf_hub_download(
            repo_id=hf_model, 
            filename="model.joblib", 
            token=hf_token if hf_token else None
        )
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Failed to load model from Hub: {e}")
        return None

model = load_model_from_hub()

# -- Internal Cleaning Logic (Must match training script) -------------------
def preprocess_input(df):
    """Applies the same cleaning used during training."""
    df = df.copy()
    
    # 1. Fix Sugar Content labels
    if 'Product_Sugar_Content' in df.columns:
        df['Product_Sugar_Content'] = df['Product_Sugar_Content'].replace({'reg': 'Regular'})
    
    # 2. Convert Year to Age (Crucial for model accuracy)
    if 'Store_Establishment_Year' in df.columns:
        df['Store_Age'] = 2024 - df['Store_Establishment_Year']
        df = df.drop(columns=['Store_Establishment_Year'])
        
    # 3. Drop IDs that the model wasn't trained on
    for col in ['Product_Id', 'Store_Id']:
        if col in df.columns:
            df = df.drop(columns=[col])
            
    return df

# -- UI Logic ---------------------------------------------------------------
mode = st.radio("Mode", ["Single record", "Batch CSV upload"])

pred_df = None

if mode == "Single record":
    with st.expander("Enter Product and Store Details", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            p_mrp = st.number_input("Product_MRP", min_value=10.0, max_value=500.0, value=147.0)
            p_weight = st.number_input("Product_Weight", min_value=1.0, max_value=50.0, value=12.5)
            p_area = st.number_input("Product_Allocated_Area", 0.0, 1.0, 0.06)
            p_type = st.selectbox("Product_Type", [
                'Frozen Foods', 'Dairy', 'Canned', 'Baking Goods', 'Health and Hygiene',
                'Snack Foods', 'Meat', 'Household', 'Hard Drinks', 'Fruits and Vegetables'
            ])
            p_sugar = st.selectbox("Product_Sugar_Content", ['Low Sugar', 'Regular', 'No Sugar'])

        with col2:
            s_year = st.number_input("Store_Establishment_Year", 1980, 2024, 2000)
            s_size = st.selectbox("Store_Size", ['Small', 'Medium', 'High'])
            s_city = st.selectbox("Store_Location_City_Type", ['Tier 1', 'Tier 2', 'Tier 3'])
            s_type = st.selectbox("Store_Type", ['Supermarket Type1', 'Supermarket Type2', 'Departmental Store', 'Food Mart'])

    if st.button("Predict"):
        # Construct raw DataFrame from inputs
        raw_data = {
            "Product_Weight": [p_weight],
            "Product_Sugar_Content": [p_sugar],
            "Product_Allocated_Area": [p_area],
            "Product_Type": [p_type],
            "Product_MRP": [p_mrp],
            "Store_Establishment_Year": [s_year],
            "Store_Size": [s_size],
            "Store_Location_City_Type": [s_city],
            "Store_Type": [s_type]
        }
        input_df = pd.DataFrame(raw_data)
        
        # Apply cleaning/engineering
        cleaned_df = preprocess_input(input_df)
        
        try:
            preds = model.predict(cleaned_df)
            st.success(f"Estimated Total Sales: ${preds[0]:,.2f}")
            st.write("Processed features sent to model:")
            st.dataframe(cleaned_df)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

else:
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            cleaned_batch = preprocess_input(df)
            
            preds = model.predict(cleaned_batch)
            df["Predicted_Sales"] = preds
            
            st.write(f"Predicted {len(df)} rows")
            st.dataframe(df.head(20))
            pred_df = df
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")

if pred_df is not None:
    towrite = io.BytesIO()
    pred_df.to_csv(towrite, index=False)
    towrite.seek(0)
    st.download_button("Download Results", towrite, "predictions.csv", "text/csv")