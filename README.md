# spatiotempDQN

Small toolkit to simulate spatio-temporal data and train CNN-LSTM DQN-style models using Keras in R.

## Quick start

```r
# install devtools if needed
if (!requireNamespace("devtools", quietly = TRUE)) install.packages("devtools")
devtools::load_all()
# run example
data <- simulate_spatiotemp(n = 400)
mod <- build_model(T_steps = dim(data$X_train)[2], p = dim(data$X_train)[3])
res <- train_dqn(data$X_train, data$W_train, data$Y_obs_train, epochs = 5, model_init = mod)
