import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page settings
st.set_page_config(
    page_title="Uganda Food Price Predictor",
    page_icon="🌽",
    layout="wide"
)

# Load model and reference data
model = joblib.load("random_forest_food_price_model.pkl")
ref_df = pd.read_csv("food_price_input_reference.csv")

# Convert date column
ref_df["date"] = pd.to_datetime(ref_df["date"])

# App title
st.title("Uganda Retail Food Price Prediction App")
st.write(
    "This app predicts the next month retail price of a food commodity "
    "in a Ugandan market using a trained Random Forest regression model."
)

st.sidebar.header("Select Market and Commodity")

# User selections
commodity = st.sidebar.selectbox(
    "Select Commodity",
    sorted(ref_df["commodity"].unique())
)

market = st.sidebar.selectbox(
    "Select Market",
    sorted(ref_df["market"].unique())
)

# Filter reference data
filtered_df = ref_df[
    (ref_df["commodity"] == commodity) &
    (ref_df["market"] == market)
].copy()

if filtered_df.empty:
    st.warning("No historical data found for this commodity and market combination.")
    st.stop()

# Get latest available record for selected commodity-market pair
latest_record = filtered_df.sort_values("date").iloc[-1]

st.subheader("Latest Available Market Record")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Commodity", latest_record["commodity"])
    st.metric("Market", latest_record["market"])

with col2:
    st.metric("Region", latest_record["admin1"])
    st.metric("District", latest_record["admin2"])

with col3:
    st.metric("Latest Date", str(latest_record["date"].date()))
    st.metric("Current Price", f"UGX {latest_record['price']:,.0f}")

st.subheader("Adjust Input Values")

col1, col2 = st.columns(2)

with col1:
    current_price = st.number_input(
        "Current Retail Price (UGX)",
        min_value=0.0,
        value=float(latest_record["price"]),
        step=100.0
    )

    previous_price = st.number_input(
        "Previous Price (UGX)",
        min_value=0.0,
        value=float(latest_record["previous_price"]),
        step=100.0
    )

with col2:
    price_3_month_avg = st.number_input(
        "3-Month Average Price (UGX)",
        min_value=0.0,
        value=float(latest_record["price_3_month_avg"]),
        step=100.0
    )

    price_6_month_avg = st.number_input(
        "6-Month Average Price (UGX)",
        min_value=0.0,
        value=float(latest_record["price_6_month_avg"]),
        step=100.0
    )

# Calculate price changes
price_change = current_price - previous_price

if previous_price == 0:
    price_pct_change = 0
else:
    price_pct_change = price_change / previous_price

# Create input dataframe
input_data = pd.DataFrame([{
    "year": int(latest_record["year"]),
    "month": int(latest_record["month"]),
    "quarter": int(latest_record["quarter"]),
    "admin1": latest_record["admin1"],
    "admin2": latest_record["admin2"],
    "market": latest_record["market"],
    "latitude": float(latest_record["latitude"]),
    "longitude": float(latest_record["longitude"]),
    "category": latest_record["category"],
    "commodity": latest_record["commodity"],
    "unit": latest_record["unit"],
    "price": current_price,
    "previous_price": previous_price,
    "price_change": price_change,
    "price_pct_change": price_pct_change,
    "price_3_month_avg": price_3_month_avg,
    "price_6_month_avg": price_6_month_avg
}])

st.subheader("Model Input Preview")
st.dataframe(input_data)

# Prediction button
if st.button("Predict Next Month Price"):
    prediction = model.predict(input_data)[0]

    st.success(
        f"Predicted next month retail price for {commodity} in {market}: "
        f"UGX {prediction:,.0f}"
    )

    difference = prediction - current_price

    if difference > 0:
        st.info(f"The model predicts a price increase of UGX {difference:,.0f}.")
    elif difference < 0:
        st.info(f"The model predicts a price decrease of UGX {abs(difference):,.0f}.")
    else:
        st.info("The model predicts no price change.")

st.subheader("About the Model")

st.write(
    "The final model used in this app is a Random Forest Regressor. "
    "It was selected because it performed better than the baseline model "
    "and the neural network model during evaluation."
)

st.write("Model features include:")
st.write(
    "- Time features: year, month, quarter\n"
    "- Location features: region, district, market, latitude, longitude\n"
    "- Commodity features: category, commodity, unit\n"
    "- Historical price features: current price, previous price, price change, percentage change, 3-month average and 6-month average"
)