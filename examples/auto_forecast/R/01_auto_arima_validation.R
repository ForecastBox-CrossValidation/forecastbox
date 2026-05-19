# ============================================================
# Validacao R: AutoARIMA
# Pacotes: forecast
# ============================================================

library(forecast)

data_dir <- file.path(dirname(sys.frame(1)$ofile), "..", "data")
airline <- read.csv(file.path(data_dir, "airline.csv"))

passengers <- ts(airline$passengers, start = c(1949, 1), frequency = 12)

# --- auto.arima com diferentes criterios ---
fit_aic <- auto.arima(passengers, ic = "aic", stepwise = TRUE, trace = TRUE)
fit_bic <- auto.arima(passengers, ic = "bic", stepwise = TRUE, trace = TRUE)

cat("\n=== Modelo AIC ===\n")
print(summary(fit_aic))

cat("\n=== Modelo BIC ===\n")
print(summary(fit_bic))

# --- Previsao ---
fc_aic <- forecast(fit_aic, h = 24)

# --- Salvar resultados ---
results <- data.frame(
    criterion = c("AIC", "BIC"),
    order_p = c(fit_aic$arma[1], fit_bic$arma[1]),
    order_d = c(fit_aic$arma[6], fit_bic$arma[6]),
    order_q = c(fit_aic$arma[2], fit_bic$arma[2]),
    seasonal_P = c(fit_aic$arma[3], fit_bic$arma[3]),
    seasonal_D = c(fit_aic$arma[7], fit_bic$arma[7]),
    seasonal_Q = c(fit_aic$arma[4], fit_bic$arma[4]),
    aic = c(fit_aic$aic, fit_bic$aic),
    bic = c(fit_aic$bic, fit_bic$bic)
)
write.csv(results, file.path(data_dir, "R_auto_arima_validation.csv"), row.names = FALSE)

# --- Previsoes ---
fc_df <- data.frame(
    horizon = 1:24,
    point_forecast = as.numeric(fc_aic$mean),
    lower_80 = as.numeric(fc_aic$lower[, 1]),
    upper_80 = as.numeric(fc_aic$upper[, 1]),
    lower_95 = as.numeric(fc_aic$lower[, 2]),
    upper_95 = as.numeric(fc_aic$upper[, 2])
)
write.csv(fc_df, file.path(data_dir, "R_auto_arima_forecasts.csv"), row.names = FALSE)
cat("Results saved\n")
