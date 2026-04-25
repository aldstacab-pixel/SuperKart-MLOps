import os
import io
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
import os

# load .env from repo root
load_dotenv()   # reads .env and sets os.environ entries

HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "aldon-cabral/SuperKart-RF")


st.set_page_config(page_title="SuperKart Sales Forecast", layout="centered")

st.title("🛒 SuperKart Sales Forecasting")
st.markdown("Use the form below to predict Product_Store_Sales_Total for one record, or upload a CSV for batch predictions.")


# -- Model loader -----------------------------------------------------------
@st.cache_resource
def load_model_from_hub():
    """Download model.joblib from Hugging Face model hub. Set HF_MODEL env var to e.g. 'username/SuperKart-RF'.
    If HF_MODEL isn't set, the function will try a sensible default. Returns the deserialized sklearn pipeline.
    """
    hf_model = os.environ.get("HF_MODEL") or "aldon-cabral/SuperKart-RF"
    hf_token = os.environ.get("HF_TOKEN")
    try:
        if hf_token:
            model_path = hf_hub_download(repo_id=hf_model, filename="model.joblib", token=hf_token)
        else:
            model_path = hf_hub_download(repo_id=hf_model, filename="model.joblib")
        return joblib.load(model_path)
    except Exception as e:
        st.warning(f"Could not load model from Hugging Face: {e}")
        return None


model = load_model_from_hub()

if model is None:
    st.error("Model not available. Train & push your model to the Hugging Face Model Hub and set the HF_MODEL env var (or use the default repo name).")
    st.caption("Set HF_MODEL and (optionally) HF_TOKEN in the Space or your environment.")
    st.stop()


# -- Helper: default options for categorical fields --------------------------------
SUGAR_OPTIONS = ["Low Sugar", "Regular", "No Sugar"]
TYPE_OPTIONS = [
    "Meat", "Snack Foods", "Hard Drinks", "Dairy", "Canned", "Soft Drinks",
    "Health and Hygiene", "Baking Goods", "Bread", "Breakfast", "Frozen Foods",
    "Fruits and Vegetables", "Household", "Seafood", "Starchy Foods", "Others"
]
STORE_SIZE = ["High", "Medium", "Low"]
CITY_TIERS = ["Tier 1", "Tier 2", "Tier 3"]
STORE_TYPES = ["Departmental Store", "Supermarket Type 1", "Supermarket Type 2", "Food Mart"]


def single_input_form():
    with st.form(key="single_input"):
        st.subheader("Single record input")
        col1, col2 = st.columns(2)
        with col1:
            product_mrp = st.number_input("Product_MRP", value=100.0, min_value=0.0)
            product_weight = st.number_input("Product_Weight", value=10.0, min_value=0.0)
            product_alloc_area = st.number_input("Product_Allocated_Area", value=0.05, min_value=0.0, max_value=1.0, step=0.01)
            product_sugar = st.selectbox("Product_Sugar_Content", SUGAR_OPTIONS)
        with col2:
            product_type = st.selectbox("Product_Type", TYPE_OPTIONS)
            store_est_year = st.number_input("Store_Establishment_Year", value=2010, min_value=1900, max_value=2026)
            store_size = st.selectbox("Store_Size", STORE_SIZE)
            store_city = st.selectbox("Store_Location_City_Type", CITY_TIERS)
            store_type = st.selectbox("Store_Type", STORE_TYPES)

        submitted = st.form_submit_button("Predict single record")

    if submitted:
        row = {
            "Product_MRP": product_mrp,
            "Product_Weight": product_weight,
            "Product_Allocated_Area": product_alloc_area,
            "Product_Sugar_Content": product_sugar,
            "Product_Type": product_type,
            "Store_Establishment_Year": store_est_year,
            "Store_Size": store_size,
            "Store_Location_City_Type": store_city,
            "Store_Type": store_type,
        }
        input_df = pd.DataFrame([row])
        return input_df
    return None


st.write("---")

st.header("Predict")
col_a, col_b = st.columns([2, 1])
with col_a:
    choice = st.radio("Mode", ["Single record", "Batch CSV upload"], index=0)
with col_b:
    st.caption("Model repo: " + (os.environ.get("HF_MODEL") or "aldon-cabral/SuperKart-RF"))


pred_df = None
if choice == "Single record":
    input_df = single_input_form()
    if input_df is not None:
        try:
            preds = model.predict(input_df)
            input_df["Pred_Product_Store_Sales_Total"] = preds
            st.success(f"Estimated Total Sales: ${preds[0]:,.2f}")
            st.dataframe(input_df)
            pred_df = input_df
        except Exception as e:
            st.error(f"Prediction failed: {e}")
else:
    uploaded = st.file_uploader("Upload CSV with records (columns must match training features)", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            # drop id/unused columns if present
            for c in ["Product_Id", "Store_Id"]:
                if c in df.columns:
                    df = df.drop(columns=[c])
            preds = model.predict(df)
            df["Pred_Product_Store_Sales_Total"] = preds
            st.write(f"Predicted {len(df)} rows")
            st.dataframe(df.head(50))
            pred_df = df
        except Exception as e:
            st.error(f"Failed to read or predict uploaded file: {e}")


if pred_df is not None:
    # allow download
    towrite = io.BytesIO()
    pred_df.to_csv(towrite, index=False)
    towrite.seek(0)
    st.download_button(label="Download predictions as CSV", data=towrite, file_name="predictions.csv", mime="text/csv")

st.markdown("---")
st.markdown("Notes: the app expects the same feature columns used during training (excluding the target and ID columns). If you trained with additional preprocessing, save and push the full sklearn pipeline (preprocessing + model) as `model.joblib` to the model repo so the app can call `predict()` directly.")