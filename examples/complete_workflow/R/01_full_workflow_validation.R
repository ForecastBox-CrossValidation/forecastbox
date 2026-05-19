# ============================================================
# Validacao R: Full Forecasting Workflow
# Pacotes: forecast, ForecastComb, MCS, vars
# ============================================================

library(forecast)
library(vars)

data_dir <- file.path(dirname(sys.frame(1)$ofile), "..", "data")
macro_br <- read.csv(file.path(data_dir, "macro_brazil.csv"))

# === STEP 1: Prepare Data ===
inflation <- ts(macro_br$inflation, start = c(2010, 1), frequency = 12)
n <- length(inflation)
train_end <- n - 24
h <- 12

train <- window(inflation, end = c(2010 + (train_end - 1) %/% 12, ((train_end - 1) %% 12) + 1))
test <- tail(inflation, h)
actual <- as.numeric(test)

cat("=== Step 1: Data ===\n")
cat(sprintf("Training: %d obs, Test: %d obs\n", length(train), length(test)))

# === STEP 2: Auto-Forecast ===
fit_arima <- auto.arima(train)
fit_ets <- ets(train)

fc_arima <- forecast(fit_arima, h = h)
fc_ets <- forecast(fit_ets, h = h)
fc_naive <- naive(train, h = h)
fc_snaive <- snaive(train, h = h)

cat("\n=== Step 2: Models ===\n")
cat(sprintf("ARIMA: %s\n", capture.output(fit_arima)[2]))
cat(sprintf("ETS: %s\n", fit_ets$method))

# === STEP 3: Combination ===
fc_matrix <- cbind(
    arima = as.numeric(fc_arima$mean),
    ets = as.numeric(fc_ets$mean),
    naive = as.numeric(fc_naive$mean),
    snaive = as.numeric(fc_snaive$mean)
)
fc_simple_avg <- rowMeans(fc_matrix)

# Inverse MSE weights (from training CV)
mse_train <- c(
    mean((as.numeric(fitted(fit_arima)) - as.numeric(train))^2, na.rm = TRUE),
    mean((as.numeric(fitted(fit_ets)) - as.numeric(train))^2, na.rm = TRUE),
    mean(diff(as.numeric(train))^2),  # proxy for naive
    mean(diff(as.numeric(train), lag = 12)^2, na.rm = TRUE)  # proxy for snaive
)
inv_mse_weights <- (1 / mse_train) / sum(1 / mse_train)
fc_inv_mse <- fc_matrix %*% inv_mse_weights

cat("\n=== Step 3: Combination ===\n")
cat(sprintf("Inverse MSE weights: %s\n", paste(round(inv_mse_weights, 3), collapse=", ")))

# === STEP 4: Evaluation ===
mae_results <- c(
    arima = mean(abs(actual - as.numeric(fc_arima$mean))),
    ets = mean(abs(actual - as.numeric(fc_ets$mean))),
    naive = mean(abs(actual - as.numeric(fc_naive$mean))),
    snaive = mean(abs(actual - as.numeric(fc_snaive$mean))),
    simple_avg = mean(abs(actual - fc_simple_avg)),
    inv_mse = mean(abs(actual - fc_inv_mse))
)

cat("\n=== Step 4: Evaluation ===\n")
print(sort(mae_results))

# DM tests
e_arima <- actual - as.numeric(fc_arima$mean)
e_ets <- actual - as.numeric(fc_ets$mean)
dm <- dm.test(e_arima, e_ets, h = 1)
cat(sprintf("\nDM(ARIMA vs ETS) p-value: %.4f\n", dm$p.value))

# === STEP 5: VAR Scenarios ===
y <- ts(macro_br[1:train_end, c("gdp_growth", "inflation", "interest_rate")],
        start = c(2010, 1), frequency = 12)
var_fit <- VAR(y, p = 4, type = "const")
var_fc <- predict(var_fit, n.ahead = 12)

cat("\n=== Step 5: Scenarios (IRF) ===\n")
irf_rate <- irf(var_fit, impulse = "interest_rate", response = "inflation", n.ahead = 12)
cat("IRF interest_rate -> inflation:\n")
print(round(irf_rate$irf$interest_rate, 4))

# === SAVE RESULTS ===
results <- data.frame(
    model = names(mae_results),
    mae = as.numeric(mae_results)
)
write.csv(results, file.path(data_dir, "R_full_workflow_validation.csv"), row.names = FALSE)
cat("\n=== Results saved ===\n")
print(results[order(results$mae), ])
