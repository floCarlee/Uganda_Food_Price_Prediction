# RefactoryFinalProject
Data engineering with ML
# Uganda Approved Budget Predictive Analytics Project

## Project Title
Predicting Uganda Approved Budget Allocations Using Multi-Year Government Budget Data

## Course Requirement Alignment
This project uses a unique Uganda budgeting dataset, contains more than 1,000 raw rows, and will build a predictive machine learning model using both a classical ML model and a neural network.

## Dataset
- File: `Approved budget datasets(2).xlsx`
- Source: Uganda Budget Information Portal / Ministry of Finance budget datasets
- Financial years: FY2018/19 to FY2023/24
- Worksheets: 6
- Total raw rows: 132,802
- Raw columns: 20 to 25 columns per sheet
- NDP coverage: NDP II and NDP III

## Proposed Machine Learning Goal
Predict the approved budget allocation amount for a budget line item using features such as financial year, NDP, sector, vote, programme, fund type, item, and classification.

## Target Variable
`approved_budget`

## Suggested Models
1. Classical ML model: Random Forest Regressor
2. Neural network: Feed-forward regression neural network

## Evaluation Metrics
- MAE: Mean Absolute Error
- RMSE: Root Mean Squared Error
- R² Score

## Project Stages
1. Dataset approval and GitHub repository setup
2. Data description
3. Data cleaning and schema harmonization
4. Data exploration and visualization
5. Feature engineering
6. Classical ML model
7. Neural network model
8. Evaluation and findings
9. Demo/interface using Streamlit
10. Final presentation slides
