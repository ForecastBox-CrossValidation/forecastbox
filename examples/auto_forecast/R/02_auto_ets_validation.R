# ============================================================
# Validacao R: AutoETS
# Pacotes: forecast
# ============================================================

library(forecast)

data_dir <- file.path(dirname(sys.frame(1)$ofile), "..", "data")
airline <- read.csv(file.path(data_dir, "airline.csv"))

passengers <- ts(airline$passengers, start = c(1949, 1), frequency = 12)

# --- ets automatico ---
fit_auto <- ets(passengers)
cat("=== AutoETS Model ===\n")
print(summary(fit_auto))

# --- Forcar aditivo vs multiplicativo ---
fit_add <- ets(passengers, model = "AAA")
fit_mul <- ets(passengers, model = "MAM")

cat(sprintf("\nAditivo AIC: %.2f\n", fit_add$aic))
cat(sprintf("Multiplicativo AIC: %.2f\n", fit_mul$aic))
cat(sprintf("Auto AIC: %.2f\n", fit_auto$aic))

# --- Previsao ---
fc <- forecast(fit_auto, h = 24)

results <- data.frame(
    model = c("auto", "AAA", "MAM"),
    aic = c(fit_auto$aic, fit_add$aic, fit_mul$aic),
    bic = c(fit_auto$bic, fit_add$bic, fit_mul$bic)
)
write.csv(results, file.path(data_dir, "R_auto_ets_validation.csv"), row.names = FALSE)

fc_df <- data.frame(
    horizon = 1:24,
    point_forecast = as.numeric(fc$mean)
)
write.csv(fc_df, file.path(data_dir, "R_auto_ets_forecasts.csv"), row.names = FALSE)
cat("Results saved\n")
