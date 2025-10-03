# spatiotempDQN

**Full Reproducible Temporal Multivariate CNN-LSTM DQN with Spatio-Temporal Features and Clustering**

`spatiotempDQN` provides tools to simulate temporal + spatial features, train CNN-LSTM Deep Q-Networks (DQN) on multivariate rewards, and perform CATE-style clustering for policy analysis. It also includes real-data adaptation using NYC flights and hourly weather data.

---

## Installation

```r
# Install dependencies if not already installed
install.packages(c("keras", "tensorflow", "mclust", "factoextra", "cluster", "dplyr", "nycflights13", "nnet"))

# Install spatiotempDQN from local tar.gz
install.packages("spatiotempDQN_0.1.0.tar.gz", repos = NULL, type = "source")

# Load the package
library(spatiotempDQN)

# --- 1) Simulate temporal + spatial features ---
sim_data <- simulate_temporal_features(
  n_obs = 100,       # number of observations
  T_steps = 10,      # time steps
  p = 5              # original features per time step
)

# Extract input shape for model building
input_shape <- dim(sim_data$X_train)[2:3]  # T_steps x number of features

# --- 2) Build CNN-LSTM DQN model ---
model <- build_cnn_lstm_dqn(
  input_shape = input_shape,
  n_actions = 3,     # number of possible actions
  n_rewards = 2      # multivariate rewards
)

# --- 3) Train DQN ---
res <- train_dqn(
  X_train = sim_data$X_train,
  W_train = sim_data$W_train,
  Y_obs_train = sim_data$Y_obs_train,
  epochs = 40,
  batch_size = 64,
  weighted_sampling = TRUE,
  model_init = model
)

# --- 4) Compute CATE contrasts and perform clustering ---
cate_res <- compute_cate_clusters(
  model = res$model,
  X_test = sim_data$X_test,
  reward_weights = c(0.6, 0.4),
  n_clusters = 4
)

# View cluster profiles
print(cate_res$cluster_profiles)

# Optional: PCA plot of estimated CATEs
plot_cate_pca(cate_res)

# Load the package
library(spatiotempDQN)

# --- 1) Prepare NYC flights + weather data ---
nyc_data <- prepare_nyc_flights(T_steps = 10)

# --- 2) Build CNN-LSTM DQN model for NYC data ---
model_nyc <- build_cnn_lstm(
  T_steps = 10,
  p = nyc_data$p_final,   # number of features after spatial lag
  n_actions = 3,
  n_rewards = 2
)

# --- 3) Train DQN ---
res_nyc <- train_dqn(
  X_train = nyc_data$X_train,
  W_train = nyc_data$W_train,
  Y_obs_train = nyc_data$Y_obs_train,
  epochs = 40,
  batch_size = 64,
  weighted_sampling = TRUE,
  model_init = model_nyc
)

# --- 4) Evaluate off-policy policy value using IPS ---
val_ips <- ips_policy_value(
  model = res_nyc$model,
  nyc_data = nyc_data
)
print(val_ips)

# --- 5) Compute CATE-like contrasts and clustering ---
clusters_nyc <- compute_cate_clusters(
  model = res_nyc$model,
  X_test = nyc_data$X_test,
  reward_weights = nyc_data$reward_weights,
  n_clusters = 4
)
print(clusters_nyc$cluster_profiles)

# --- Optional: PCA plot of estimated CATEs ---
plot_cate_pca(clusters_nyc)

# --- 6) Compare with synthetic simulation clusters (if available) ---
clusters_sim <- compute_cate_clusters(
  model = res$model,
  X_test = sim_data$X_test,
  reward_weights = sim_data$reward_weights,
  n_clusters = 4
)
print(clusters_sim$cluster_profiles)

