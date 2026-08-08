# Uganda Food Price Prediction

## Project Overview

This project uses Uganda market food price data from the World Food Programme (WFP) to build a machine learning regression model for predicting retail food prices in Ugandan markets.

The goal of the project is to predict the next month retail price of a food commodity using historical price trends, market information, commodity information, and location-based features.

This project was completed as part of a Data Engineering and Analytics final project.

---

## Machine Learning Problem

This is a supervised machine learning regression problem.

The model predicts the next recorded retail price of the same food commodity in the same market.

**Target variable:**

```text
next_month_price
```

---

## Dataset Description

The dataset used is the WFP Uganda food prices dataset. The raw dataset contains market price records for different commodities across Ugandan markets.

### Raw Dataset Summary

| Item | Description |
|---|---|
| Dataset | WFP Uganda Market Food Prices |
| Country | Uganda |
| Rows | 32,171 |
| Columns | 16 |
| Markets | 43 |
| Commodities | 38 |
| Time Period | January 2006 to June 2026 |
| Currency | Uganda Shillings (UGX) |
| Source | World Food Programme food price data |

### Main Columns

The raw dataset includes:

- `date`
- `admin1`
- `admin2`
- `market`
- `market_id`
- `latitude`
- `longitude`
- `category`
- `commodity`
- `commodity_id`
- `unit`
- `priceflag`
- `pricetype`
- `currency`
- `price`
- `usdprice`

---

## Project Scope

The project focuses on predicting **retail food prices**.

During data preparation, the dataset was filtered to retain:

- Retail prices only
- Food items only
- Positive price values only

Wholesale records and non-food items were excluded because the project focuses on consumer-facing food price prediction.

After cleaning and feature engineering, the final machine learning dataset had:

```text
20,872 rows and 21 columns
```

---

## Data Cleaning and Preparation

The following cleaning and preparation steps were performed:

1. Loaded the raw WFP Uganda food prices dataset
2. Standardized column names
3. Converted the `date` column to datetime format
4. Converted price columns to numeric format
5. Checked missing values
6. Checked duplicate records
7. Filtered the dataset to retail food prices only
8. Removed non-food items
9. Created time-based features:
   - `year`
   - `month`
   - `quarter`
10. Created historical price features:
   - `previous_price`
   - `price_change`
   - `price_pct_change`
   - `price_3_month_avg`
   - `price_6_month_avg`
11. Created the target variable:
   - `next_month_price`
12. Removed rows with missing lag or target values
13. Saved the cleaned machine learning dataset

---

## Features Used for Modelling

The model used the following input features:

### Time Features

- `year`
- `month`
- `quarter`

### Location Features

- `admin1`
- `admin2`
- `market`
- `latitude`
- `longitude`

### Commodity Features

- `category`
- `commodity`
- `unit`

### Historical Price Features

- `price`
- `previous_price`
- `price_change`
- `price_pct_change`
- `price_3_month_avg`
- `price_6_month_avg`

---

## Exploratory Data Analysis

The exploratory data analysis included:

- Distribution of retail food prices
- Top commodities by number of records
- Number of records by food category
- Average price by food category
- Average retail food price trend over time
- Price trends by category
- Current price vs next month price
- Correlation analysis of numerical price features

### Key EDA Findings

- Food prices vary by commodity, market, category, and time.
- Current price, previous price, and moving averages have strong relationships with next month price.
- Some commodities have much higher prices due to differences in units and market value.
- The dataset contains some high-price outliers, which may be caused by commodity type, market shocks, inflation, or sudden supply changes.

---

## Machine Learning Models

Three models were compared:

1. Baseline model
2. Random Forest Regressor
3. Neural Network Regressor

### Baseline Model

The baseline model assumed:

```text
next month price = current month price
```

This was used as a simple benchmark.

### Classical Machine Learning Model

The classical machine learning model used was:

```text
Random Forest Regressor
```

### Neural Network Model

A feedforward neural network was trained using TensorFlow/Keras.

---

## Model Evaluation

The models were evaluated using:

- MAE: Mean Absolute Error
- RMSE: Root Mean Squared Error
- R² Score

### Model Performance

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Baseline | 557.70 | 1975.42 | 0.8586 |
| Random Forest | 533.20 | 1697.70 | 0.8955 |
| Neural Network | 691.14 | 1760.43 | 0.8877 |

---

## Best Model

The best model was the:

```text
Random Forest Regressor
```

It achieved the best overall performance because it had:

- the lowest MAE
- the lowest RMSE
- the highest R² score

The Random Forest model explained approximately 89.6% of the variation in next month retail food prices.

---

## Demo Interface

A simple Gradio demo interface was created for the final model.

The demo allows a user to:

1. Select a food commodity
2. Select a market
3. Enter current price
4. Enter previous price
5. Enter 3-month average price
6. Enter 6-month average price
7. Predict the next month retail food price

Gradio was used because it works well in Google Colab and provides a shareable demo link.

---

## Project Structure

```text
Uganda_Food_Price_Prediction/
│
├── data/
│   ├── raw/
│   │   └── wfp_food_prices_uga.csv
│   │
│   └── processed/
│       ├── wfp_uganda_food_prices_ml_ready.csv
│       └── final_food_price_predictions.csv
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_Data_Exploration_and_Visualization.ipynb
│   ├── 03_Machine_Learning_Preparation.ipynb
│   
│
├── models/
│   └── random_forest_food_price_model.pkl
│
├── reports/
│   └── figures/
│
├── app/
│   ├── app.py
│   └── food_price_input_reference.csv
│
├── presentation/
│   └── Uganda_Food_Price_Prediction_Sample_Presentation.pptx
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/floCarlee/Uganda_Food_Price_Prediction.git
```

### 2. Open the Project Folder

```bash
cd Uganda_Food_Price_Prediction
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Run the Notebooks

Run the notebooks in order:

1. `01_data_understanding.ipynb`
2. `02_data_cleaning_preparation.ipynb`
3. `03_eda_visualization.ipynb`
4. `04_machine_learning_preparation.ipynb`
5. `05_classical_ml_model.ipynb`
6. `06_neural_network_model.ipynb`
7. `07_demo_interface.ipynb`

### 5. Run the Demo App

```bash
python app/app.py
```

Alternatively, the Gradio interface can be launched directly from the demo notebook.

---

## Requirements

The main Python libraries used are:

- pandas
- numpy
- matplotlib
- scikit-learn
- tensorflow
- joblib
- gradio

---

## Key Findings

- Retail food prices in Uganda can be predicted using historical market price data.
- Current price and previous price are strong predictors of next month price.
- Moving averages help capture short-term price trends.
- The Random Forest model performed better than both the baseline and neural network models.
- Food prices differ significantly across commodities, categories, markets, and time periods.

---

## Limitations

The dataset does not include some external factors that may affect food prices, such as:

- rainfall
- fuel prices
- exchange rates
- inflation
- transport costs
- conflict or supply disruptions
- harvest seasons
- market demand changes

Including these factors in future work may improve model performance.

---

## Future Improvements

Future versions of the project could:

- Add weather data
- Add fuel price data
- Add inflation data
- Add exchange rate data
- Use advanced time-series models
- Tune model hyperparameters further
- Deploy the demo as a web application
- Build an interactive dashboard using Power BI or Tableau

---

## Conclusion

This project demonstrates an end-to-end data engineering and analytics workflow using Uganda market food price data.

The project includes data cleaning, feature engineering, exploratory data analysis, machine learning modelling, neural network modelling, model evaluation, and a working demo interface.

The final Random Forest model provides a practical way to estimate next month retail food prices in Ugandan markets.
