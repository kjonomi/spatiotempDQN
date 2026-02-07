# spatiotempDQN

**Offline Reinforcement Learning with Temporal CNN–LSTM,  
Spatial Graph Features, and CATE Clustering**

---

## Overview

`spatiotempDQN` implements an Offline Fitted Q-Evaluation (FQE) framework
using a temporal CNN–LSTM architecture with spatial feature aggregation
and heterogeneous treatment effect (CATE) clustering.

The package supports:

- Multi-action, multi-reward reinforcement learning
- Offline policy evaluation (IPS + FQE)
- Temporal convolution + LSTM modeling
- Spatial graph feature aggregation
- Conditional Average Treatment Effect (CATE) estimation
- K-means clustering for heterogeneity discovery
- Bootstrap confidence intervals
- Simulation studies
- Real-data applications (NYC Flights dataset)

---

## Installation

Install the development version from GitHub:

```r
install.packages("devtools")
devtools::install_github("kjonomi/spatiotempDQN", force = TRUE)
