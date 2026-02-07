# Offline Fitted Q-Evaluation with Temporal CNN–LSTM + Spatial Graph Features + CATE Clustering

This repository demonstrates the implementation of **Offline Fitted Q-Evaluation (FQE)** using **CNN-LSTM networks** with temporal and spatial covariates. The goal is to estimate **Conditional Average Treatment Effects (CATE)**, and cluster observations based on their CATE estimates. The model also evaluates the policy using **Importance Sampling (IPS)** off-policy evaluation.

The code also includes:
- **Spatio-temporal feature generation**
- **CATE clustering and visualization**
- **Policy distribution summary**

## Prerequisites

To run the code, you need to install the following R packages:

```r
pkgs <- c("keras","tensorflow","mclust","cluster","factoextra",
          "dplyr","ggplot2")
for (p in pkgs) if (!requireNamespace(p, quietly=TRUE)) install.packages(p)
