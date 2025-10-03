# spatiotempDQN

**Full Reproducible Temporal Multivariate CNN-LSTM DQN with Spatio-Temporal Features and Clustering**

`spatiotempDQN` provides tools to simulate temporal + spatial features, train CNN-LSTM Deep Q-Networks (DQN) on multivariate rewards, and perform CATE-style clustering for policy analysis. It also includes real-data adaptation using NYC flights and hourly weather data.

---

## Installation

```r
# Install dependencies if not already installed
install.packages(c("keras", "tensorflow", "mclust", "factoextra", "cluster", "dplyr", "nycflights13", "nnet"))

# Install spatiotempDQN from local tar.gz
install.packages("spatiotempDQN.tar.gz", repos = NULL, type = "source")
library(spatiotempDQN)

# Simulate temporal + spatial features
sim_data <- simulate_temporal_spatial(n = 500, T_steps = 10, p = 6, n_actions = 3, n_rewards = 2)

# Build model
model <- build_cnn_lstm(T_steps = 10, p = sim_data$p_final, n_actions = 3, n_rewards = 2)

# Train DQN
res <- train_dqn(X_train = sim_data$X_train,
                 W_train = sim_data$W_train,
                 Y_obs_train = sim_data$Y_obs_train,
                 epochs = 40, batch_size = 64,
                 weighted_sampling = TRUE,
                 model_init = model)

# Compute CATE contrasts and clusters

library(spatiotempDQN)

# Prepare NYC flights + weather data
nyc_data <- prepare_nyc_flights(T_steps = 10)

# Build and train model
model_nyc <- build_cnn_lstm(T_steps = 10, p = nyc_data$p_final, n_actions = 3, n_rewards = 2)
res_nyc <- train_dqn(X_train = nyc_data$X_train,
                     W_train = nyc_data$W_train,
                     Y_obs_train = nyc_data$Y_obs_train,
                     epochs = 40, batch_size = 64,
                     weighted_sampling = TRUE,
                     model_init = model_nyc)

# Evaluate off-policy using IPS
val_ips <- ips_policy_value(res_nyc$model, nyc_data)
print(val_ips)

# CATE-like clustering
clusters_nyc <- compute_cate_clusters(res_nyc$model, nyc_data$X_test, nyc_data$reward_weights, k = 4)
print(clusters_nyc$summary)

clusters <- compute_cate_clusters(res$model, sim_data$X_test, sim_data$reward_weights, k = 4)
print(clusters$summary)
