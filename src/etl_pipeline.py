import os
import pandas as pd
import numpy as np


def run_etl(
    raw_path="data/raw/wfp_food_prices_uga.csv",
    processed_path="data/processed/wfp_uganda_food_prices_ml_ready.csv",
    app_reference_path="app/food_price_input_reference.csv"
):
    print("Starting ETL pipeline...")

    # Load raw data
    df = pd.read_csv(raw_path)

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Convert date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Convert numeric columns
    numeric_cols = [
        "market_id", "latitude", "longitude", "commodity_id",
        "price", "usdprice"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Strip text columns
    text_cols = df.select_dtypes(include="object").columns

    for col in text_cols:
        df[col] = df[col].astype(str).str.strip()

    # Remove duplicates
    df = df.drop_duplicates()

    # Remove invalid records
    df = df.dropna(subset=["date", "price"])
    df = df[df["price"] > 0]

    # Keep retail food prices only
    if "pricetype" in df.columns:
        df = df[df["pricetype"].str.lower() == "retail"]

    if "category" in df.columns:
        df = df[df["category"].str.lower() != "non-food"]

    # Create time features
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter

    # Group by commodity and market
    group_cols = ["commodity_id", "market_id"]

    df = df.sort_values(group_cols + ["date"])

    # Create lag and rolling features
    df["previous_price"] = df.groupby(group_cols)["price"].shift(1)
    df["price_change"] = df["price"] - df["previous_price"]
    df["price_pct_change"] = df["price_change"] / df["previous_price"]

    df["price_3_month_avg"] = (
        df.groupby(group_cols)["price"]
        .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    )

    df["price_6_month_avg"] = (
        df.groupby(group_cols)["price"]
        .transform(lambda x: x.rolling(window=6, min_periods=1).mean())
    )

    # Create target variable
    df["next_month_price"] = df.groupby(group_cols)["price"].shift(-1)

    # Remove rows that cannot be used for ML
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(
        subset=[
            "previous_price",
            "price_change",
            "price_pct_change",
            "price_3_month_avg",
            "price_6_month_avg",
            "next_month_price"
        ]
    )

    # Select final ML columns
    ml_columns = [
        "date",
        "year",
        "month",
        "quarter",
        "admin1",
        "admin2",
        "market",
        "market_id",
        "latitude",
        "longitude",
        "category",
        "commodity",
        "commodity_id",
        "unit",
        "price",
        "previous_price",
        "price_change",
        "price_pct_change",
        "price_3_month_avg",
        "price_6_month_avg",
        "next_month_price"
    ]

    ml_df = df[ml_columns].copy()

    # Create folders if they do not exist
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    os.makedirs(os.path.dirname(app_reference_path), exist_ok=True)

    # Save processed ML dataset
    ml_df.to_csv(processed_path, index=False)

    # Save reference file for the app
    app_reference_columns = [
        "date",
        "year",
        "month",
        "quarter",
        "admin1",
        "admin2",
        "market",
        "latitude",
        "longitude",
        "category",
        "commodity",
        "unit",
        "price",
        "previous_price",
        "price_change",
        "price_pct_change",
        "price_3_month_avg",
        "price_6_month_avg"
    ]

    ml_df[app_reference_columns].to_csv(app_reference_path, index=False)

    print("ETL pipeline completed successfully.")
    print(f"Processed dataset saved to: {processed_path}")
    print(f"App reference data saved to: {app_reference_path}")
    print(f"Final shape: {ml_df.shape}")

    return ml_df


if __name__ == "__main__":
    run_etl()