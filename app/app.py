# ============================================================
# Uganda Retail Food Price Prediction Dashboard
# app.py
#
# This app:
# 1. Runs ETL automatically from the raw WFP CSV
# 2. Creates the processed ML-ready dataset
# 3. Creates the app reference dataset
# 4. Loads the trained Random Forest model
# 5. Displays a live Gradio dashboard and prediction demo
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib


# ============================================================
# 1. PROJECT PATHS
# ============================================================

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(APP_DIR)

RAW_DATA_PATH = os.path.join(
    PROJECT_DIR,
    "data",
    "raw",
    "wfp_food_prices_uga.csv"
)

PROCESSED_DATA_PATH = os.path.join(
    PROJECT_DIR,
    "data",
    "processed",
    "wfp_uganda_food_prices_ml_ready.csv"
)

REFERENCE_DATA_PATH = os.path.join(
    APP_DIR,
    "food_price_input_reference.csv"
)

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "random_forest_food_price_model.pkl"
)


# ============================================================
# 2. ETL PIPELINE
# ============================================================

def run_etl(
    raw_path=RAW_DATA_PATH,
    processed_path=PROCESSED_DATA_PATH,
    app_reference_path=REFERENCE_DATA_PATH
):
    """
    Extracts raw WFP Uganda food price data, cleans it, engineers features,
    creates the next-month price target, and saves processed outputs.
    """

    if not os.path.exists(raw_path):
        print("Raw data file not found. Skipping ETL.")
        print(f"Expected raw file at: {raw_path}")
        return None

    print("Starting ETL pipeline...")

    # Extract
    df = pd.read_csv(raw_path)

    # Standardize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    required_columns = [
        "date",
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
        "pricetype",
        "price"
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"Raw dataset is missing required columns: {missing_columns}"
        )

    # Convert date column
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Convert numeric columns
    numeric_cols = [
        "market_id",
        "latitude",
        "longitude",
        "commodity_id",
        "price"
    ]

    if "usdprice" in df.columns:
        numeric_cols.append("usdprice")

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clean text columns
    text_cols = df.select_dtypes(include="object").columns

    for col in text_cols:
        df[col] = df[col].astype(str).str.strip()

    # Remove duplicates
    df = df.drop_duplicates()

    # Remove invalid records
    df = df.dropna(subset=["date", "price"])
    df = df[df["price"] > 0]

    # Keep retail prices only
    df = df[df["pricetype"].str.lower() == "retail"]

    # Remove non-food category
    df = df[df["category"].str.lower() != "non-food"]

    # Create time features
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter

    # Sort data before lag/rolling calculations
    group_cols = ["commodity_id", "market_id"]
    df = df.sort_values(group_cols + ["date"])

    # Feature engineering
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

    # Target variable: next recorded price for same commodity and market
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

    # Final ML dataset columns
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

    # Create folders
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    os.makedirs(os.path.dirname(app_reference_path), exist_ok=True)

    # Save full processed ML dataset
    ml_df.to_csv(processed_path, index=False)

    # Save app reference dataset
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
    print(f"Processed data saved to: {processed_path}")
    print(f"App reference data saved to: {app_reference_path}")
    print(f"Final ML dataset shape: {ml_df.shape}")

    return ml_df


# ============================================================
# 3. RUN ETL AUTOMATICALLY WHEN APP STARTS
# ============================================================

run_etl()


# ============================================================
# 4. LOAD MODEL AND APP REFERENCE DATA
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found. Expected file at: {MODEL_PATH}"
    )

if not os.path.exists(REFERENCE_DATA_PATH):
    raise FileNotFoundError(
        f"Reference data file not found. Expected file at: {REFERENCE_DATA_PATH}"
    )

model = joblib.load(MODEL_PATH)

ref_df = pd.read_csv(REFERENCE_DATA_PATH)
ref_df["date"] = pd.to_datetime(ref_df["date"], errors="coerce")


# ============================================================
# 5. CLEAN LOADED REFERENCE DATA
# ============================================================

numeric_columns = [
    "year",
    "month",
    "quarter",
    "latitude",
    "longitude",
    "price",
    "previous_price",
    "price_change",
    "price_pct_change",
    "price_3_month_avg",
    "price_6_month_avg"
]

for col in numeric_columns:
    if col in ref_df.columns:
        ref_df[col] = pd.to_numeric(ref_df[col], errors="coerce")

ref_df = ref_df.dropna(subset=["date", "year", "price"])


# ============================================================
# 6. MODEL FEATURES
# ============================================================

model_features = [
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

missing_features = [col for col in model_features if col not in ref_df.columns]

if missing_features:
    raise ValueError(
        f"The app reference data is missing these model columns: {missing_features}"
    )


# ============================================================
# 7. FILTER OPTIONS
# ============================================================

all_categories = ["All"] + sorted(ref_df["category"].dropna().unique().tolist())
all_commodities = ["All"] + sorted(ref_df["commodity"].dropna().unique().tolist())
all_markets = ["All"] + sorted(ref_df["market"].dropna().unique().tolist())
all_years = sorted(ref_df["year"].dropna().astype(int).unique().tolist())

commodities = sorted(ref_df["commodity"].dropna().unique().tolist())
markets = sorted(ref_df["market"].dropna().unique().tolist())

DEFAULT_CATEGORY = "All"
DEFAULT_COMMODITY = "All"
DEFAULT_MARKET = "All"
DEFAULT_START_YEAR = min(all_years)
DEFAULT_END_YEAR = max(all_years)

REFRESH_SECONDS = 60


# ============================================================
# 8. DASHBOARD STYLING
# ============================================================

custom_css = """
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 20px;
}

.kpi-card {
    background: #f8f9fa;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #e6e6e6;
    text-align: center;
}

.kpi-value {
    font-size: 26px;
    font-weight: 700;
    color: #1f2937;
}

.kpi-label {
    font-size: 14px;
    color: #6b7280;
}

.dashboard-note {
    background: #f8fafc;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #e5e7eb;
}
"""


# ============================================================
# 9. HELPER FUNCTIONS
# ============================================================

def make_empty_plot(message):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        fontsize=12
    )
    ax.axis("off")
    plt.tight_layout()
    return fig


def filter_dashboard_data(category, commodity, market, start_year, end_year):
    filtered = ref_df.copy()

    if category != "All":
        filtered = filtered[filtered["category"] == category]

    if commodity != "All":
        filtered = filtered[filtered["commodity"] == commodity]

    if market != "All":
        filtered = filtered[filtered["market"] == market]

    filtered = filtered[
        (filtered["year"] >= int(start_year)) &
        (filtered["year"] <= int(end_year))
    ]

    return filtered


# ============================================================
# 10. KPI FUNCTION
# ============================================================

def dashboard_kpis_filtered(category, commodity, market, start_year, end_year):
    filtered = filter_dashboard_data(
        category,
        commodity,
        market,
        start_year,
        end_year
    )

    total_records = len(filtered)
    total_markets = filtered["market"].nunique()
    total_commodities = filtered["commodity"].nunique()

    if filtered.empty:
        date_range = "No data available for selected filters"
    else:
        date_range = (
            f"{filtered['date'].min().date()} "
            f"to {filtered['date'].max().date()}"
        )

    return f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-value">{total_records:,}</div>
            <div class="kpi-label">Filtered Records</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">{total_markets}</div>
            <div class="kpi-label">Markets</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">{total_commodities}</div>
            <div class="kpi-label">Commodities</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">UGX</div>
            <div class="kpi-label">Currency</div>
        </div>
    </div>

    <p><b>Date range:</b> {date_range}</p>
    """


# ============================================================
# 11. DASHBOARD VISUALIZATION FUNCTIONS
# ============================================================

def plot_top_commodities_filtered(
    category,
    commodity,
    market,
    start_year,
    end_year
):
    filtered = filter_dashboard_data(
        category,
        commodity,
        market,
        start_year,
        end_year
    )

    if filtered.empty:
        return make_empty_plot("No data available for selected filters")

    top_commodities = filtered["commodity"].value_counts().head(10)

    fig, ax = plt.subplots(figsize=(10, 6))
    top_commodities.plot(kind="bar", ax=ax)

    ax.set_title("Top 10 Commodities by Number of Records")
    ax.set_xlabel("Commodity")
    ax.set_ylabel("Number of Records")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    return fig


def plot_average_price_trend_filtered(
    category,
    commodity,
    market,
    start_year,
    end_year
):
    filtered = filter_dashboard_data(
        category,
        commodity,
        market,
        start_year,
        end_year
    )

    if filtered.empty:
        return make_empty_plot("No data available for selected filters")

    monthly_avg = (
        filtered
        .set_index("date")
        .resample("M")["price"]
        .mean()
        .dropna()
        .reset_index()
    )

    if monthly_avg.empty:
        return make_empty_plot("No monthly trend available for selected filters")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        monthly_avg["date"],
        monthly_avg["price"],
        marker="o",
        linewidth=2
    )

    ax.set_title("Average Retail Food Price Trend Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Average Retail Price in UGX")
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig


def plot_category_average_price_filtered(
    category,
    commodity,
    market,
    start_year,
    end_year
):
    filtered = filter_dashboard_data(
        category,
        commodity,
        market,
        start_year,
        end_year
    )

    if filtered.empty:
        return make_empty_plot("No data available for selected filters")

    category_avg = (
        filtered
        .groupby("category")["price"]
        .mean()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    category_avg.plot(kind="bar", ax=ax)

    ax.set_title("Average Retail Price by Food Category")
    ax.set_xlabel("Food Category")
    ax.set_ylabel("Average Retail Price in UGX")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    return fig


def plot_current_vs_predicted_filtered(
    category,
    commodity,
    market,
    start_year,
    end_year
):
    filtered = filter_dashboard_data(
        category,
        commodity,
        market,
        start_year,
        end_year
    ).copy()

    if filtered.empty:
        return make_empty_plot("No data available for selected filters")

    prediction_df = filtered.replace([np.inf, -np.inf], np.nan)
    prediction_df = prediction_df.dropna(subset=model_features)

    if prediction_df.empty:
        return make_empty_plot(
            "No complete rows available for prediction after filtering"
        )

    prediction_df["predicted_next_month_price"] = model.predict(
        prediction_df[model_features]
    )

    prediction_trend = (
        prediction_df
        .set_index("date")
        .resample("M")
        .agg(
            current_price=("price", "mean"),
            predicted_next_month_price=("predicted_next_month_price", "mean")
        )
        .dropna()
        .reset_index()
    )

    if prediction_trend.empty:
        return make_empty_plot(
            "No prediction trend available for selected filters"
        )

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        prediction_trend["date"],
        prediction_trend["current_price"],
        marker="o",
        linewidth=2,
        label="Current Price"
    )

    ax.plot(
        prediction_trend["date"],
        prediction_trend["predicted_next_month_price"],
        marker="o",
        linewidth=2,
        label="Predicted Next Month Price"
    )

    ax.set_title("Current Price vs Predicted Next Month Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Average Retail Price in UGX")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig


def refresh_dashboard_filtered(
    category,
    commodity,
    market,
    start_year,
    end_year
):
    return (
        dashboard_kpis_filtered(
            category,
            commodity,
            market,
            start_year,
            end_year
        ),
        plot_top_commodities_filtered(
            category,
            commodity,
            market,
            start_year,
            end_year
        ),
        plot_average_price_trend_filtered(
            category,
            commodity,
            market,
            start_year,
            end_year
        ),
        plot_category_average_price_filtered(
            category,
            commodity,
            market,
            start_year,
            end_year
        ),
        plot_current_vs_predicted_filtered(
            category,
            commodity,
            market,
            start_year,
            end_year
        )
    )


# ============================================================
# 12. PREDICTION DEMO FUNCTION
# ============================================================

def predict_food_price_with_plot(
    commodity,
    market,
    current_price,
    previous_price,
    price_3_month_avg,
    price_6_month_avg
):
    filtered_df = ref_df[
        (ref_df["commodity"] == commodity) &
        (ref_df["market"] == market)
    ].copy()

    if filtered_df.empty:
        return (
            "No historical data found for this commodity and market combination.",
            make_empty_plot("No historical data found")
        )

    latest_record = filtered_df.sort_values("date").iloc[-1]

    price_change = current_price - previous_price

    if previous_price == 0:
        price_pct_change = 0
    else:
        price_pct_change = price_change / previous_price

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

    prediction = model.predict(input_data)[0]
    difference = prediction - current_price

    if difference > 0:
        direction = (
            f"The model predicts a price increase of UGX {difference:,.0f}."
        )
    elif difference < 0:
        direction = (
            f"The model predicts a price decrease of UGX {abs(difference):,.0f}."
        )
    else:
        direction = "The model predicts no price change."

    result_text = (
        f"Predicted Next Month Retail Price: UGX {prediction:,.0f}\n\n"
        f"Commodity: {commodity}\n"
        f"Market: {market}\n"
        f"Current Price: UGX {current_price:,.0f}\n"
        f"Previous Price: UGX {previous_price:,.0f}\n"
        f"3-Month Average: UGX {price_3_month_avg:,.0f}\n"
        f"6-Month Average: UGX {price_6_month_avg:,.0f}\n\n"
        f"{direction}"
    )

    comparison_df = pd.DataFrame({
        "Price Type": [
            "Previous Price",
            "Current Price",
            "3-Month Avg",
            "6-Month Avg",
            "Predicted Next Month"
        ],
        "Price": [
            previous_price,
            current_price,
            price_3_month_avg,
            price_6_month_avg,
            prediction
        ]
    })

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(comparison_df["Price Type"], comparison_df["Price"])

    ax.set_title(f"Price Comparison for {commodity} in {market}")
    ax.set_ylabel("Retail Price in UGX")

    for i, value in enumerate(comparison_df["Price"]):
        ax.text(
            i,
            value,
            f"{value:,.0f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    return result_text, fig


# ============================================================
# 13. INITIAL DASHBOARD VALUES
# ============================================================

(
    initial_kpis,
    initial_top_plot,
    initial_trend_plot,
    initial_category_plot,
    initial_prediction_plot
) = refresh_dashboard_filtered(
    DEFAULT_CATEGORY,
    DEFAULT_COMMODITY,
    DEFAULT_MARKET,
    DEFAULT_START_YEAR,
    DEFAULT_END_YEAR
)


# ============================================================
# 14. BUILD GRADIO APP
# ============================================================

with gr.Blocks(
    title="Uganda Food Price Prediction Dashboard",
    css=custom_css
) as demo:

    gr.Markdown(
        """
        # Uganda Retail Food Price Prediction Dashboard

        This live dashboard presents exploratory insights from WFP Uganda market price data
        and provides a machine learning demo for predicting next month retail food prices.

        The app automatically runs an ETL pipeline at startup to prepare the latest data.
        """
    )

    dashboard_timer = gr.Timer(value=REFRESH_SECONDS)

    with gr.Tab("Dashboard Overview"):

        gr.Markdown("## Dashboard Filters")

        with gr.Row():
            category_filter = gr.Dropdown(
                choices=all_categories,
                value=DEFAULT_CATEGORY,
                label="Filter by Category"
            )

            commodity_filter = gr.Dropdown(
                choices=all_commodities,
                value=DEFAULT_COMMODITY,
                label="Filter by Commodity"
            )

            market_filter = gr.Dropdown(
                choices=all_markets,
                value=DEFAULT_MARKET,
                label="Filter by Market"
            )

        with gr.Row():
            start_year_filter = gr.Dropdown(
                choices=all_years,
                value=DEFAULT_START_YEAR,
                label="Start Year"
            )

            end_year_filter = gr.Dropdown(
                choices=all_years,
                value=DEFAULT_END_YEAR,
                label="End Year"
            )

        gr.Markdown("## Dataset Summary")
        kpi_output = gr.HTML(value=initial_kpis)

        gr.Markdown("## Data Visualizations")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Top 10 Commodities by Number of Records")
                top_plot = gr.Plot(value=initial_top_plot)

            with gr.Column():
                gr.Markdown("### Average Retail Food Price Trend")
                trend_plot = gr.Plot(value=initial_trend_plot)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Average Retail Price by Food Category")
                category_plot = gr.Plot(value=initial_category_plot)

            with gr.Column():
                gr.Markdown("### Current Price vs Predicted Price")
                prediction_trend_plot = gr.Plot(value=initial_prediction_plot)

        gr.Markdown(
            """
            <div class="dashboard-note">
            <b>Interpretation:</b> Use the filters to explore how food prices vary by
            category, commodity, market, and year. The Current Price vs Predicted Price
            chart shows that the dashboard is connected to the trained machine learning model.
            </div>
            """
        )

        filter_inputs = [
            category_filter,
            commodity_filter,
            market_filter,
            start_year_filter,
            end_year_filter
        ]

        dashboard_outputs = [
            kpi_output,
            top_plot,
            trend_plot,
            category_plot,
            prediction_trend_plot
        ]

        for filter_component in filter_inputs:
            filter_component.change(
                fn=refresh_dashboard_filtered,
                inputs=filter_inputs,
                outputs=dashboard_outputs
            )

    with gr.Tab("Prediction Demo"):

        gr.Markdown(
            """
            ## Predict Next Month Retail Food Price

            Select a commodity and market, then enter the current and historical price values.
            The trained Random Forest model will predict the expected retail price for the next month.
            """
        )

        with gr.Row():
            with gr.Column():
                commodity_input = gr.Dropdown(
                    choices=commodities,
                    value=commodities[0],
                    label="Select Commodity"
                )

                market_input = gr.Dropdown(
                    choices=markets,
                    value=markets[0],
                    label="Select Market"
                )

            with gr.Column():
                current_price_input = gr.Number(
                    label="Current Retail Price (UGX)",
                    value=float(ref_df["price"].median())
                )

                previous_price_input = gr.Number(
                    label="Previous Price (UGX)",
                    value=float(ref_df["previous_price"].median())
                )

        with gr.Row():
            with gr.Column():
                avg_3_input = gr.Number(
                    label="3-Month Average Price (UGX)",
                    value=float(ref_df["price_3_month_avg"].median())
                )

            with gr.Column():
                avg_6_input = gr.Number(
                    label="6-Month Average Price (UGX)",
                    value=float(ref_df["price_6_month_avg"].median())
                )

        predict_button = gr.Button(
            "Predict Next Month Price",
            variant="primary"
        )

        with gr.Row():
            with gr.Column():
                prediction_output = gr.Textbox(
                    label="Prediction Result",
                    lines=10
                )

            with gr.Column():
                prediction_plot = gr.Plot(
                    label="Prediction Visualization"
                )

        predict_button.click(
            fn=predict_food_price_with_plot,
            inputs=[
                commodity_input,
                market_input,
                current_price_input,
                previous_price_input,
                avg_3_input,
                avg_6_input
            ],
            outputs=[
                prediction_output,
                prediction_plot
            ]
        )

    with gr.Tab("ETL Pipeline"):

        gr.Markdown(
            f"""
            ## Automated ETL Pipeline

            This app runs an ETL pipeline automatically when it starts.

            ### Extract

            The app reads the raw WFP Uganda food price dataset from:

            `{RAW_DATA_PATH}`

            ### Transform

            The ETL process:

            - standardizes column names
            - converts date values
            - converts prices and coordinates to numeric values
            - removes duplicate records
            - removes invalid prices
            - keeps retail food prices only
            - removes non-food records
            - creates year, month, and quarter features
            - creates previous price, price change, and percentage change
            - creates 3-month and 6-month average price features
            - creates the target variable: `next_month_price`

            ### Load

            The processed files are saved to:

            `{PROCESSED_DATA_PATH}`

            `{REFERENCE_DATA_PATH}`

            This supports the automation requirement because the dashboard is not only static;
            it prepares the latest available dataset before running the visualizations and prediction demo.
            """
        )

    with gr.Tab("About the Project"):

        gr.Markdown(
            """
            ## Project Summary

            This project uses WFP Uganda food price data to predict next month retail food prices
            in Ugandan markets.

            ### Machine Learning Task

            This is a supervised regression problem.

            **Target variable:** `next_month_price`

            ### Final Model

            The Random Forest Regressor was selected as the final model because it performed
            best compared to the baseline model and neural network.

            ### Model Performance

            | Model | MAE | RMSE | R² |
            |---|---:|---:|---:|
            | Baseline | 557.70 | 1975.42 | 0.8586 |
            | Random Forest | 533.20 | 1697.70 | 0.8955 |
            | Neural Network | 691.14 | 1760.43 | 0.8877 |

            ### Business Value

            This dashboard can support food price monitoring by helping users explore market
            trends and estimate likely price movement for the next month.

            ### Limitations

            The model does not include external factors such as rainfall, fuel prices, inflation,
            exchange rates, transport costs, harvest seasons, conflict, or supply disruptions.
            """
        )

    demo.load(
        fn=refresh_dashboard_filtered,
        inputs=[
            category_filter,
            commodity_filter,
            market_filter,
            start_year_filter,
            end_year_filter
        ],
        outputs=[
            kpi_output,
            top_plot,
            trend_plot,
            category_plot,
            prediction_trend_plot
        ]
    )

    dashboard_timer.tick(
        fn=refresh_dashboard_filtered,
        inputs=[
            category_filter,
            commodity_filter,
            market_filter,
            start_year_filter,
            end_year_filter
        ],
        outputs=[
            kpi_output,
            top_plot,
            trend_plot,
            category_plot,
            prediction_trend_plot
        ]
    )


# ============================================================
# 15. LAUNCH APP
# ============================================================

if __name__ == "__main__":
    demo.launch(share=True)