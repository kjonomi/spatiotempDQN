#' Simulate spatio-temporal data
#' @export
simulate_spatiotemp <- function(n = 100, T_steps = 5, p = 3, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  X <- array(rnorm(n * T_steps * p), dim = c(n, T_steps, p))
  W <- sample(0:2, n, replace = TRUE)
  Y <- matrix(rnorm(n * 2), ncol = 2)
  list(X_train = X, W_train = W, Y_obs_train = Y)
}
