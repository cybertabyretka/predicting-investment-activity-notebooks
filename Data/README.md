# Russian investment activity

## Overview
This dataset provides a comprehensive panel of economic, social, and financial indicators for all regions of the Russian Federation. It is specifically engineered for the task of forecasting the key economic metric "Investments in Fixed Capital" (Инвестиции в основной капитал) at the regional level.

## Target Variable
The primary goal is to predict target_next_year — the future value of fixed capital investments for each region.

## Dataset Features & Innovation
The dataset combines traditional features with advanced feature engineering:
 - **Core Temporal & Target**: year, log_value, delta_target, delta_target_percent.
 - **Regional Context**: Encoded regions and federal districts (okrug), regional and district-level statistical aggregates.
 - **Macroeconomic & Financial Indicators**: GDP (ВРП), central bank rates, inflation (Индексы потребительских цен), government debt, oil & gold prices, currency rates, and financial market indices (MCFTR, RVI, RGBITR).
 - **Socio-Economic Dimensions**: Population, income, wages, unemployment, demographics, retail trade, housing, and student numbers.
 - **Innovation & Corporate Data**: R&D spending, patent applications, innovation activity, and corporate accounts payable.
 - **Advanced Temporal Features**: Includes lagged values (lag_1, lag_2) and exponentially weighted moving averages (EWM) for key indicators to capture trends.
 - **Key Innovation**: Incorporates Markov Chain-derived state probabilities. The economic state of each region (e.g., state_sharp_growth, state_fall) and transition probabilities (e.g., prob_to_growth) are added as features, providing a novel, probabilistic view of regional economic dynamics.

## Potential Use Cases
 - **Time-Series Forecasting**: Build models (LSTM, Transformer, Gradient Boosting) to predict regional investment activity.
 - **Economic Analysis**: Study the impact of macroeconomic factors, oil prices, or innovation on regional development.
 - **Causal Inference**: Investigate relationships between socio-economic policies and investment outcomes.
 - **Methodological Research**: Explore the utility of Markov Chain states as features in economic forecasting.

## Dataset Structure
The data is structured in a long format (panel data), with each row representing a unique region-year observation.

## Acknowledgements
Data compiled from public and official statistical sources of the Russian Federation.

---
More information about the dataset can be found at the following link: https://www.kaggle.com/datasets/demirtry/russian-investment-activity