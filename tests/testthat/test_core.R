context("core functionality")

test_that("simulate_spatiotemp returns expected shapes", {
  dat <- simulate_spatiotemp(n = 200, T_steps = 8, p = 4, seed = 42)
  expect_is(dat$X_combined, "array")
  expect_equal(dim(dat$X_train)[2], 8)
  expect_true(ncol(dat$Y_obs) == 2)
})

test_that("build_model compiles and predicts", {
  mod <- build_model(T_steps = 8, p = 4, n_actions = 3, n_rewards = 2)
  # small fake batch
  X <- array(rnorm(5 * 8 * 4), dim = c(5, 8, 4))
  preds <- predict(mod, X)
  expect_equal(dim(preds)[1], 5)
  expect_equal(dim(preds)[2], 3 * 2)
})
