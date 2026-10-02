############################################################
# REAL-DATA FITTED Q-EVALUATION (FQE) WITH TEMPORAL CNN-LSTM
# + SPATIAL GRAPH FEATURES + CATE CLUSTERING (NYC FLIGHTS)
#
# Keras 3 + Reticulate Compatible
############################################################

# ==========================================================
# 0. ENVIRONMENT SETUP
# ==========================================================

Sys.setenv(CUDA_VISIBLE_DEVICES = "-1")
Sys.setenv(TF_CPP_LOG_LEVEL = "2")

needed <- c(
  "keras", "tensorflow", "nycflights13", "dplyr", "tidyr", "lubridate",
  "purrr", "stringr", "nnet", "mclust", "cluster", "factoextra", 
  "ggplot2", "knitr", "kableExtra"
)

for (p in needed) {
  if (!requireNamespace(p, quietly = TRUE)) {
    install.packages(p, quiet = TRUE)
  }
  library(p, character.only = TRUE)
}

set.seed(123)
tf$random$set_seed(123)


# ==========================================================
# 1. PARAMETERS & CONFIGURATION
# ==========================================================

T_steps        <- 10
feat_vars      <- c("temp", "dewp", "humid", "wind_dir", "wind_speed", 
                    "wind_gust", "precip", "pressure", "visib")
p              <- length(feat_vars)
n_actions      <- 3
n_rewards      <- 2
train_frac     <- 0.70
reward_weights <- c(0.6, 0.4)
gamma          <- 0.90


# ==========================================================
# 2. DATA PREPARATION (NYC FLIGHTS & WEATHER)
# ==========================================================

cat("Preparing NYC Flights and Weather dataset...\n")

# Process Weather Data
wx <- nycflights13::weather %>%
  mutate(ts = make_datetime(year, month, day, hour)) %>%
  dplyr::select(c("origin", "ts", feat_vars)) %>%
  arrange(origin, ts) %>%
  group_by(origin) %>% 
  fill(all_of(feat_vars), .direction = "downup") %>% 
  ungroup()

# Process Flights Data
fl <- nycflights13::flights %>%
  mutate(
    sched_dep_hour = floor(sched_dep_time / 100),
    ts_dep         = make_datetime(year, month, day, sched_dep_hour)
  ) %>%
  filter(origin %in% c("JFK", "LGA", "EWR")) %>%
  filter(!is.na(dep_delay), !is.na(arr_delay), !is.na(ts_dep)) %>%
  dplyr::select(year, month, day, origin, dest, carrier, flight,
                ts_dep, dep_delay, arr_delay)

# Action Mapping: 0=Morning [5-12), 1=Afternoon [12-18), 2=Evening/Night
bucket_of_hour <- function(h) {
  if (h >= 5 && h < 12) return(0L)
  if (h >= 12 && h < 18) return(1L)
  return(2L)
}
fl <- fl %>% mutate(action = vapply(hour(ts_dep), bucket_of_hour, integer(1)))

# Rewards: Negative absolute delays
Y_obs <- cbind(-abs(fl$dep_delay), -abs(fl$arr_delay))

# Spatio-Temporal Sequences Lookup Function
neighbors <- list(JFK = c("LGA", "EWR"), LGA = c("JFK", "EWR"), EWR = c("JFK", "LGA"))

get_seq_row <- function(origin, ts_dep) {
  times <- ts_dep - hours(T_steps:1) + hours(1)
  
  wx_o <- wx %>% filter(origin == !!origin, ts %in% times) %>% arrange(ts)
  if (nrow(wx_o) != T_steps) return(NULL)
  Xo <- as.matrix(wx_o[, feat_vars])
  
  nb <- neighbors[[origin]]
  wx_n <- wx %>% 
    filter(origin %in% nb, ts %in% times) %>%
    group_by(ts) %>% 
    summarise(across(all_of(feat_vars), mean, na.rm = TRUE), .groups = "drop") %>%
    arrange(ts)
  
  if (nrow(wx_n) != T_steps) return(NULL)
  Xn <- as.matrix(wx_n[, feat_vars])
  
  list(Xo = Xo, Xn = Xn)
}

# Subsample dataset for performance
n_max  <- 8000
fl_sub <- fl %>% slice_sample(n = min(n_max, nrow(fl)))
Y_sub  <- Y_obs[as.numeric(rownames(fl_sub)), ]

rows <- vector("list", nrow(fl_sub))
keep <- rep(FALSE, nrow(fl_sub))

for (i in seq_len(nrow(fl_sub))) {
  gi <- get_seq_row(fl_sub$origin[i], fl_sub$ts_dep[i])
  if (!is.null(gi)) { rows[[i]] <- gi; keep[i] <- TRUE }
}

fl_sub <- fl_sub[keep, ]
Y_sub  <- Y_sub[keep, ]
rows   <- rows[keep]
n      <- nrow(fl_sub)

X_origin <- array(NA_real_, dim = c(n, T_steps, p))
X_spat   <- array(NA_real_, dim = c(n, T_steps, p))

for (i in seq_len(n)) {
  X_origin[i, , ] <- rows[[i]]$Xo
  X_spat[i, , ]   <- rows[[i]]$Xn
}

# Combine Origin + Spatial Features
X_combined <- array(NA_real_, dim = c(n, T_steps, 2 * p))
X_combined[, , 1:p] <- X_origin
X_combined[, , (p + 1):(2 * p)] <- X_spat
p_final <- dim(X_combined)[3]

# Standardization
for (j in seq_len(p_final)) {
  for (t in seq_len(T_steps)) {
    v   <- X_combined[, t, j]
    mu  <- mean(v, na.rm = TRUE)
    sdv <- sd(v, na.rm = TRUE)
    if (!is.finite(sdv) || sdv == 0) sdv <- 1
    X_combined[, t, j] <- (v - mu) / sdv
  }
}
X_combined[!is.finite(X_combined)] <- 0

# Extract Final Dataset Matrices
A_obs <- fl_sub$action

# Train/Test Split
idx_tr <- sample(seq_len(n), floor(train_frac * n))
idx_te <- setdiff(seq_len(n), idx_tr)

X_tr <- X_combined[idx_tr, , , drop = FALSE]
A_tr <- A_obs[idx_tr]
Y_tr <- Y_sub[idx_tr, , drop = FALSE]

X_te <- X_combined[idx_te, , , drop = FALSE]
A_te <- A_obs[idx_te]
Y_te <- Y_sub[idx_te, , drop = FALSE]

cat(sprintf("Dataset prepared: Train N = %d | Test N = %d | Features = %d\n", 
            length(idx_tr), length(idx_te), p_final))


# ==========================================================
# 3. CNN-LSTM Q-NETWORK BUILDER
# ==========================================================

build_cnnlstm_qnet <- function(T_steps, p_dim, n_actions, n_rewards, lr = 1e-4) {
  inputs <- keras::layer_input(shape = c(T_steps, p_dim))

  x <- inputs |>
    keras::layer_conv_1d(filters = 32, kernel_size = 3, activation = "relu", padding = "causal") |>
    keras::layer_lstm(units = 48, return_sequences = FALSE) |>
    keras::layer_dense(units = 48, activation = "relu")

  outputs <- x |>
    keras::layer_dense(units = n_actions * n_rewards, activation = "linear")

  model     <- keras::keras_model(inputs = inputs, outputs = outputs)
  optimizer <- keras::optimizer_adam(learning_rate = lr)

  model$compile(optimizer = optimizer, loss = "mse")
  return(model)
}


# ==========================================================
# 4. FITTED Q-EVALUATION (FQE) FITTER
# ==========================================================

fit_fqe <- function(X, A, Y, epochs = 20, batch_size = 64, gamma = 0.90) {
  N   <- dim(X)[1]
  T_s <- dim(X)[2]
  P   <- dim(X)[3]

  cat(sprintf("\nFQE initialization: N = %d, T = %d, P = %d\n", N, T_s, P))

  q_net      <- build_cnnlstm_qnet(T_s, P, n_actions, n_rewards)
  target_net <- build_cnnlstm_qnet(T_s, P, n_actions, n_rewards)
  target_net$set_weights(q_net$get_weights())

  for (ep in seq_len(epochs)) {
    cat(sprintf("FQE epoch %d/%d\n", ep, epochs))

    idx     <- sample(seq_len(N), size = N, replace = FALSE)
    batches <- split(idx, ceiling(seq_along(idx) / batch_size))

    for (b in batches) {
      nb  <- length(b)
      X_b <- array(X[b, , , drop = FALSE], dim = c(nb, T_s, P))
      A_b <- A[b] + 1L
      Y_b <- Y[b, , drop = FALSE]

      # Predict current Q
      Q_curr_raw <- q_net$predict(X_b, verbose = 0)
      Q_curr     <- array(as.numeric(Q_curr_raw), dim = c(nb, n_actions, n_rewards))

      # Construct next-state tensor (X_next)
      X_next <- array(0, dim = c(nb, T_s, P))
      if (T_s > 1) {
        X_next[, 1:(T_s - 1), 1:P] <- X_b[, 2:T_s, 1:P]
      }
      X_next[, T_s, 1:P] <- X_b[, T_s, 1:P] # Pad with final observation

      # Target predictions
      Q_next_raw <- target_net$predict(X_next, verbose = 0)
      Q_next     <- array(as.numeric(Q_next_raw), dim = c(nb, n_actions, n_rewards))

      # Choose greedy action based on scalarized multi-reward Q
      Q_next_scal  <- apply(Q_next, c(1, 2), function(x) sum(x * reward_weights))
      best_actions <- apply(Q_next_scal, 1, which.max)

      # Bellman Targets Update
      for (i in seq_len(nb)) {
        act            <- A_b[i]
        next_act       <- best_actions[i]
        obs_reward     <- as.numeric(Y_b[i, , drop = TRUE])
        next_val       <- as.numeric(Q_next[i, next_act, , drop = TRUE])
        
        bellman_target <- obs_reward + gamma * next_val
        Q_curr[i, act, ] <- bellman_target
      }

      # Flatten & Train Main Network
      target_flat <- matrix(Q_curr, nrow = nb, ncol = n_actions * n_rewards)
      q_net$fit(x = X_b, y = target_flat, epochs = 1L, batch_size = nb, verbose = 0L)
    }

    # Synchronize target network periodically
    if (ep %% 5 == 0 || ep == epochs) {
      target_net$set_weights(q_net$get_weights())
      cat(sprintf("  -> Target network updated at epoch %d\n", ep))
    }
  }

  cat("\nFQE training completed successfully.\n")
  return(q_net)
}


# ==========================================================
# 5. OFF-POLICY EVALUATION (OPE)
# ==========================================================

evaluate_ope <- function(X_test, A_test, Y_test, q_model, prop_model) {
  N <- dim(X_test)[1]

  # Evaluate Q Values
  Q_raw  <- q_model$predict(X_test, verbose = 0)
  Q_arr  <- array(as.numeric(Q_raw), dim = c(N, n_actions, n_rewards))
  Q_scal <- apply(Q_arr, c(1, 2), function(x) sum(x * reward_weights))

  # Target Evaluation Policy
  pi_hat <- apply(Q_scal, 1, which.max) - 1L

  # Propensity scores calculation
  X_flat <- apply(X_test, c(1, 3), mean)
  colnames(X_flat) <- paste0("V", seq_len(ncol(X_flat)))

  phat <- predict(prop_model, newdata = as.data.frame(X_flat), type = "probs")
  if (is.null(dim(phat))) phat <- matrix(phat, ncol = n_actions)
  phat <- pmax(pmin(phat, 0.99), 0.01)

  # Reward and Index setup
  r_scalar  <- as.numeric(Y_test %*% reward_weights)
  a_obs_idx <- cbind(seq_len(N), A_test + 1L)
  pi_e_obs  <- phat[a_obs_idx]

  # IPS & Truncation
  w_ips       <- ifelse(A_test == pi_hat, 1 / pi_e_obs, 0)
  trunc_cut   <- quantile(w_ips, probs = 0.95, na.rm = TRUE)
  w_ips_trunc <- pmin(w_ips, trunc_cut)

  # Estimators Calculation
  ess        <- if (sum(w_ips_trunc^2) > 0) sum(w_ips_trunc)^2 / sum(w_ips_trunc^2) else 0
  val_ips    <- mean(w_ips_trunc * r_scalar)
  val_snips  <- if (sum(w_ips_trunc) > 0) sum(w_ips_trunc * r_scalar) / sum(w_ips_trunc) else NA_real_

  # Doubly Robust Estimator
  v_hat_pi <- apply(Q_scal, 1, max)
  q_obs    <- Q_scal[a_obs_idx]
  val_dr   <- mean(v_hat_pi + w_ips_trunc * (r_scalar - q_obs))

  list(
    val_ips   = val_ips,
    val_snips = val_snips,
    val_dr    = val_dr,
    ess       = ess,
    pi_hat    = pi_hat,
    Q_scal    = Q_scal
  )
}


# ==========================================================
# 6. PIPELINE EXECUTION
# ==========================================================

# Fit FQE Model
cat("\nFitting CNN-LSTM FQE model on NYC Flights data...\n")
q_model <- fit_fqe(X = X_tr, A = A_tr, Y = Y_tr, epochs = 20, batch_size = 64, gamma = gamma)

# Fit Propensity Model
cat("\nFitting propensity score model...\n")
X_tr_flat <- apply(X_tr, c(1, 3), mean)
colnames(X_tr_flat) <- paste0("V", seq_len(ncol(X_tr_flat)))
prop_model <- nnet::multinom(factor(A_tr) ~ ., data = as.data.frame(X_tr_flat), trace = FALSE)

# Off-Policy Evaluation
cat("\nRunning Off-Policy Evaluation (OPE)...\n")
ope_res <- evaluate_ope(X_test = X_te, A_test = A_te, Y_test = Y_te, q_model = q_model, prop_model = prop_model)

cat(sprintf("\nOPE Results -> DR: %.4f | SNIPS: %.4f | IPS: %.4f | ESS: %.2f\n",
            ope_res$val_dr, ope_res$val_snips, ope_res$val_ips, ope_res$ess))


# ==========================================================
# 7. CATE CONTRASTS & CLUSTERING ANALYSIS
# ==========================================================

cate_hat <- cbind(
  tau10 = ope_res$Q_scal[, 2] - ope_res$Q_scal[, 1],
  tau20 = ope_res$Q_scal[, 3] - ope_res$Q_scal[, 1]
)

# Silhouette Analysis to select optimal K
sil_scores <- sapply(2:6, function(k) {
  km <- kmeans(scale(cate_hat), centers = k, nstart = 25)
  ss <- cluster::silhouette(km$cluster, dist(scale(cate_hat)))
  mean(ss[, 3])
})

best_k <- which.max(sil_scores) + 1
cat(sprintf("\nOptimal Cluster Count selected (K) = %d\n", best_k))

# Final K-Means & PCA Visualization
km_opt  <- kmeans(scale(cate_hat), centers = best_k, nstart = 50)
pca_res <- prcomp(scale(cate_hat))
pca_var <- pca_res$sdev^2 / sum(pca_res$sdev^2)

pca_df <- data.frame(
  PC1     = pca_res$x[, 1],
  PC2     = pca_res$x[, 2],
  Cluster = factor(km_opt$cluster)
)

pca_plot <- ggplot(pca_df, aes(x = PC1, y = PC2, color = Cluster)) +
  geom_point(alpha = 0.8, size = 2) +
  theme_minimal() +
  labs(
    title    = paste0("PCA of CATE Contrasts (NYC Flights, K = ", best_k, ")"),
    subtitle = sprintf("PC1: %.1f%% var | PC2: %.1f%% var", pca_var[1] * 100, pca_var[2] * 100),
    x        = "PC1", 
    y        = "PC2"
  )

print(pca_plot)


# ==========================================================
# 8. BOOTSTRAPPED SUMMARY TABLES & LATEX OUTPUT
# ==========================================================

df_summary <- data.frame(
  cluster   = km_opt$cluster,
  tau10_hat = cate_hat[, 1],
  tau20_hat = cate_hat[, 2],
  dep_delay = Y_te[, 1],
  arr_delay = Y_te[, 2]
)

boot_mean_ci <- function(x, B = 200, alpha = 0.05) {
  n <- length(x)
  boot_means <- replicate(B, mean(sample(x, n, replace = TRUE)))
  c(
    mean  = mean(x),
    lower = quantile(boot_means, alpha / 2),
    upper = quantile(boot_means, 1 - alpha / 2)
  )
}

cluster_summary <- df_summary %>%
  group_by(cluster) %>%
  summarise(
    n               = n(),
    mean_tau10      = round(boot_mean_ci(tau10_hat)[1], 3),
    ci_tau10_lower  = round(boot_mean_ci(tau10_hat)[2], 3),
    ci_tau10_upper  = round(boot_mean_ci(tau10_hat)[3], 3),
    mean_tau20      = round(boot_mean_ci(tau20_hat)[1], 3),
    ci_tau20_lower  = round(boot_mean_ci(tau20_hat)[2], 3),
    ci_tau20_upper  = round(boot_mean_ci(tau20_hat)[3], 3),
    mean_dep_delay  = round(mean(dep_delay), 2),
    mean_arr_delay  = round(mean(arr_delay), 2)
  ) %>%
  ungroup()

policy_table <- as.data.frame(table(ope_res$pi_hat))
colnames(policy_table) <- c("Action", "Count")
policy_table$Proportion <- round(policy_table$Count / sum(policy_table$Count), 3)

cat("\n%--- LEARNED POLICY DISTRIBUTION ---\n")
print(kable(policy_table, format = "latex", booktabs = TRUE,
            caption = "Learned Policy Distribution (FQE Target Policy)"))

cat("\n%--- CLUSTER CATE SUMMARY WITH BOOTSTRAP CIs ---\n")
print(kable(cluster_summary, format = "latex", booktabs = TRUE,
            caption = "Cluster-wise Estimated CATEs with Bootstrap 95% CIs") %>%
        kable_styling(latex_options = "hold_position"))

cat("\nREAL DATA PIPELINE EXECUTION COMPLETED SUCCESSFULLY.\n")