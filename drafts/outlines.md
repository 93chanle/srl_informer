
# Introduction

## Explain the need for RL

## Explain why forecasting SRL is important

## Explain how RL market functions

- Daily bids
- 12 products of 6 time slots
- capacity prices & working prices
- ...

## Changes in regulations of the RL market

- Before 11.2020: time of published key paper on the same topic of SRL (Marten 2020)
- Major changes to the market

## The immergence of ML & DL in the field of energy forecasting

- Introduction to ML & DL
- Rise of ML-oriented solutions in forecasting different aspects of energy

## Concept of machine learning

- Approximating an unknown target function
- Hypothesis space
- Choosing a model
- Measuring accuracy & loss functions
- Training & validation

## Concept of deep learning 

- Neural network models
- Non-linear activations
- Back propagation

## Concept & characteristics of time series data

- Seasonalities
- Peaks & outliers
- Structural breaks (potential pitfalls to forecasting models(?))
- Forecasting horizons: short-, mid-, long-lerm forecasting

## Contribution

- Novel idea of incorporating predicted revenue as accuracy measure
- Applying Transformer model for time series to the field of energy forecasting, which conventionally was saturated with statistical analysis models like ARIMA and traditional deep learning recurrent models like LSTM
- Revising and expanding the work in the identified key papers (Marten 2020) which was published before multiple major changes in the regulation of the SRL market

# Literature review

## Regelleistung in the literature

## Transformer models for energy forecasting in the literature

## Boosting models for energy forecasting in the literature

## Key paper: SRL prediction (Marten 2020)

## Key paper: Informer model (Zhou 2021)

- Only summarizes, model details are discussed in Methodology

# The data

## SRL data

- Obtained from regelleistung.de
- 

## Analysis of data

- TS plots
- Autocorrelation (ACF, PACF)?
- Analysing peaks & outliers?
- Detecting seasonalities
- Trend decomposition
- ...

# Methodology

## Main ideas

- Applying different forecasting models to the respective data and comparing the results with baseline

## Problem formulation

- Given daily prices of $x_{t-k} ... x_{t}$, predict $x_{t+1}$
- Short-term

## Informer model

## Boosting (XGBoost)

## ARIMA model

## LSTM model

## Baseline

- Day-ahead forecasting

# Experiments

## Different ways of scaling the data

- MinMax scaling
- Standard scaling

## Informer: Different variation of loss functions

- RMSE loss
- Weighted RMSE loss (penalizes more when pred > true)
- Log RMSE loss

## Informer: Adding predicted revenue as accuracy metric

- Assuming 1 KW of sold electricity

## Informer and XGBoost: Hyperparameter tunning

### Types of HP

- Random search
- Grid search
- Bayesian hyperparameter tunning

### Working with Optuna

- Explain Optuna
- Define parameter search space
- Config the tuning

## Informer and XGBoost: Multivariate approach (TBC)

## Adding more data (TBC)

### Find and scrape following

- Weather data
- EPEX prices
- Gas price
- ...

# Results

# Discussion

## Complexity vs Computational cost comparison

- Number of weights & parameters of a model
- Track (average) computational time 

## Effects of data scaling in preprocessing on different models

- Informer, LSTM: DL models, different unit scales affect shape of gradient and difficulty of model learning
- XGBoost: tree-based model, monotonic transformation/scaling makes little senses here, model only picks cut-points to split a node
- ARIMA: statistical model, is not relevant

# Conclusion