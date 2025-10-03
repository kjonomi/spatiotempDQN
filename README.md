# spatiotempDQN

Spatio-temporal Deep Q-Learning package in R.

## Installation

```r
# install.packages("devtools")
devtools::install_github("kjonomi/spatiotempDQN")
```

## Example

```r
library(spatiotempDQN)
data <- simulate_spatiotemp(n = 400)
mod <- build_model(T_steps = dim(data$X_train)[2], p = dim(data$X_train)[3])
res <- train_dqn(data$X_train, data$W_train, data$Y_obs_train, epochs = 5, model_init = mod)
```
