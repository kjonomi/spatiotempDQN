# Multivariate Spatio-Temporal Treatment Policy Optimization for Air Transportation

This repository contains the code and analysis for the manuscript:

**Multivariate Spatio-Temporal Treatment Policy Optimization for Air Transportation via Deep Reinforcement Learning and Clustering of Heterogeneous Treatment Effects**

The study develops an offline fitted Q-evaluation (FQE) framework that combines temporal representation learning, spatial information, multi-outcome policy evaluation, off-policy evaluation, and heterogeneous treatment-effect analysis.

## Overview

The proposed framework is designed for observational sequential decision problems in which:

- covariates evolve over time;
- observations have spatial dependence;
- treatment/action assignments are observational;
- multiple outcomes must be evaluated jointly; and
- treatment effects may vary across observational units.

The framework combines:

1. **Spatio-temporal state representation**
   - Historical weather trajectories are represented using temporal sequences.
   - Weather information from neighboring airports is incorporated as spatial context.
   - Causal temporal convolutions preserve chronological ordering.

2. **CNN--LSTM Q-network**
   - 1D causal convolution extracts local temporal patterns.
   - LSTM layers capture longer-range temporal dependence.
   - The network estimates a vector-valued Q-function for multiple outcomes.

3. **Offline Fitted Q-Evaluation (FQE)**
   - Q-values are estimated from previously collected observational data.
   - No online interaction with the transportation environment is required.
   - A prespecified weighted scalarization is used to construct the evaluation policy.

4. **Off-policy evaluation**
   - A multinomial propensity model represents observational treatment assignment.
   - Policy values are evaluated using IPS, SNIPS, and doubly robust (DR) estimators.
   - Effective sample size (ESS) is reported to characterize weighting stability.

5. **CATE-based heterogeneity analysis**
   - Model-implied treatment-effect contrasts are obtained from differences between Q-values.
   - Standardized contrasts are clustered using K-means.
   - Silhouette analysis is used to select the number of clusters.
   - PCA provides a low-dimensional visualization of estimated treatment-effect heterogeneity.

---

## Data

The empirical application uses data from the New York City airport system:

- **JFK** -- John F. Kennedy International Airport
- **LGA** -- LaGuardia Airport
- **EWR** -- Newark Liberty International Airport

The analysis combines:

- historical flight information from `nycflights13`;
- hourly meteorological variables;
- departure and arrival delays; and
- spatial information from neighboring airports.

The weather variables used in the analysis are:

```text
temp
dewp
humid
wind_dir
wind_speed
wind_gust
precip
pressure
visib
