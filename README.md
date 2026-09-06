# Life Expectancy Prediction and Analysis

A machine learning and statistical analysis project using the WHO Life Expectancy dataset to investigate the relationship between life expectancy and health, demographic, and socioeconomic factors.

## Overview

This project performs an end-to-end analysis of life expectancy across countries and years. The workflow includes exploratory data analysis, data preprocessing, dimensionality reduction, regression modeling, feature selection, model comparison, and diagnostic analysis.

## Dataset

The project uses the **WHO Life Expectancy dataset** containing:

- 2,938 country-year observations
- 193 countries
- 20 health, demographic, and socioeconomic predictors
- Target variable: **Life Expectancy**

The dataset covers the period **2000–2015**.

## Methodology

### 1. Data Preprocessing
- Inspected missing values and data distributions
- Handled missing observations using median imputation
- Prepared numerical predictors for modeling
- Standardized features where required

### 2. Exploratory Data Analysis
- Distribution analysis of life expectancy
- Correlation analysis
- Scatter plots between life expectancy and important predictors
- Missing-value analysis
- Trends in life expectancy over time
- Correlation heatmap

### 3. Dimensionality Reduction

**Principal Component Analysis (PCA)** was used to:
- Analyze the structure of the predictor space
- Examine explained variance
- Visualize relationships between variables
- Study the contribution of different features

### 4. Machine Learning Models

The following models were implemented and compared:

- Ordinary Least Squares (OLS)
- Ridge Regression
- Lasso Regression
- Random Forest Regression
- Gradient Boosting Regression

Models were evaluated using cross-validation and regression metrics including:

- RMSE
- MAE
- R²

### 5. Feature Selection and Interpretation

- Lasso regression was used for feature selection.
- Random Forest feature importance was used to identify influential predictors.
- Residual analysis was performed to examine model errors and diagnostic behaviour.

## Project Structure

```text
life-expectancy-ml/
│
├── MA5755_Analysis.ipynb
├── life_expectancy.py
├── Life Expectancy Data.csv
└── README.md
