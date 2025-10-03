# spatiotempDQN

**Full Reproducible Temporal Multivariate CNN-LSTM DQN with Spatio-Temporal Features and Clustering**

`spatiotempDQN` provides tools to simulate temporal + spatial features, train CNN-LSTM Deep Q-Networks (DQN) on multivariate rewards, and perform CATE-style clustering for policy analysis. It also includes real-data adaptation using NYC flights and hourly weather data.

---

## Installation

```r
# Install dependencies if not already installed
install.packages(c("keras", keras3", "tensorflow", "mclust", "factoextra", "cluster", "dplyr", "nycflights13", "nnet"))

# Install spatiotempDQN from local tar.gz
install.packages("spatiotempDQN_0.1.0.tar.gz", repos = NULL, type = "source")

# Load the package
library(spatiotempDQN)
library(keras)
library(keras3)

# 1. Synthetic Data Simulation
X_syn <- simulate_temporal_features(n_obs = 200, T_steps = 10, p = 5)

# 2. Build CNN-LSTM DQN
model <- build_cnn_lstm_dqn(input_shape = c(10, 5), n_actions = 3)

# 3. Train DQN
Y_syn <- matrix(rnorm(200*3), ncol = 3)
W_syn <- sample(0:2, 200, replace = TRUE)
model <- train_dqn_model(model, X_syn, W_syn, Y_syn, epochs = 2)

# 4. NYC Flights Data
nyc_data <- prepare_nycflights(T_steps = 10)
str(nyc_data)

# 5. CATE Clustering
clusters <- cate_clustering(nyc_data$X_train, nyc_data$Y_obs_train, n_clusters = 3)
table(clusters)
hist(clusters)
```
