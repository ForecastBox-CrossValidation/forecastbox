# ============================================================
# Referencia R: Pipeline de Forecasting
# Pacotes: forecast
# ============================================================

library(forecast)

data_dir <- file.path(dirname(sys.frame(1)$ofile), "..", "data")
macro_br <- read.csv(file.path(data_dir, "macro_brazil.csv"))

# --- Pipeline manual em R ---

# 1. Preparar dados
gdp <- ts(macro_br$gdp_growth, start = c(2010, 1), frequency = 12)
inflation <- ts(macro_br$inflation, start = c(2010, 1), frequency = 12)
n <- length(gdp)
train_end <- n - 24
h <- 12

train_gdp <- window(gdp, end = c(2010 + (train_end - 1) %/% 12, ((train_end - 1) %% 12) + 1))

# 2. Fit multiple models
fit_arima <- auto.arima(train_gdp)
fit_ets <- ets(train_gdp)
fit_naive <- naive(train_gdp, h = h)

# 3. Generate forecasts
fc_arima <- forecast(fit_arima, h = h)
fc_ets <- forecast(fit_ets, h = h)
fc_naive <- naive(train_gdp, h = h)

# 4. Combine (simple average)
fc_combined <- (as.numeric(fc_arima$mean) + as.numeric(fc_ets$mean) + as.numeric(fc_naive$mean)) / 3

# 5. Evaluate
test <- tail(gdp, h)
actual <- as.numeric(test)

results <- data.frame(
    model = c("auto_arima", "auto_ets", "naive", "combined"),
    mae = c(
        mean(abs(actual - as.numeric(fc_arima$mean))),
        mean(abs(actual - as.numeric(fc_ets$mean))),
        mean(abs(actual - as.numeric(fc_naive$mean))),
        mean(abs(actual - fc_combined))
    ),
    rmse = c(
        sqrt(mean((actual - as.numeric(fc_arima$mean))^2)),
        sqrt(mean((actual - as.numeric(fc_ets$mean))^2)),
        sqrt(mean((actual - as.numeric(fc_naive$mean))^2)),
        sqrt(mean((actual - fc_combined)^2))
    )
)

cat("=== Pipeline Results ===\n")
print(results)

write.csv(results, file.path(data_dir, "R_pipeline_reference.csv"), row.names = FALSE)
cat("Results saved\n")

# 6. Recurring forecast simulation (rolling)
rolling_mae <- c()
for (end_t in (train_end):(n - h)) {
    train <- window(gdp, end = c(2010 + (end_t - 1) %/% 12, ((end_t - 1) %% 12) + 1))
    fit <- auto.arima(train)
    fc <- forecast(fit, h = 1)
    actual_val <- gdp[end_t + 1]
    rolling_mae <- c(rolling_mae, abs(actual_val - as.numeric(fc$mean)))
}

cat(sprintf("\nRolling 1-step MAE (auto.arima): %.4f (sd: %.4f)\n",
    mean(rolling_mae), sd(rolling_mae)))
