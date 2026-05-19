# ============================================================
# Validacao R: Online Learning / Time-Varying Weights
# Pacotes: opera
# ============================================================

library(opera)

data_dir <- file.path(dirname(sys.frame(1)$ofile), "..", "data")
df <- read.csv(file.path(data_dir, "inflation_forecasts.csv"))

actual <- df$actual
forecasts <- as.matrix(df[, c("fc_arima", "fc_ets", "fc_var", "fc_naive", "fc_drift")])

# --- EWA (Exponentially Weighted Average) ---
ewa <- mixture(Y = actual, experts = forecasts, model = "EWA")
cat("=== EWA ===\n")
cat(sprintf("MAE: %.6f\n", mean(abs(ewa$prediction - actual))))

# --- BOA (Bernstein Online Aggregation) ---
boa <- mixture(Y = actual, experts = forecasts, model = "BOA")
cat("\n=== BOA ===\n")
cat(sprintf("MAE: %.6f\n", mean(abs(boa$prediction - actual))))

# --- MLpol (ML-Poly) ---
mlpol <- mixture(Y = actual, experts = forecasts, model = "MLpol")
cat("\n=== MLpol ===\n")
cat(sprintf("MAE: %.6f\n", mean(abs(mlpol$prediction - actual))))

# --- Salvar pesos ao longo do tempo ---
weights_ewa <- as.data.frame(ewa$weights)
colnames(weights_ewa) <- c("w_arima", "w_ets", "w_var", "w_naive", "w_drift")
write.csv(weights_ewa, file.path(data_dir, "R_online_weights.csv"), row.names = FALSE)

results <- data.frame(
    method = c("EWA", "BOA", "MLpol"),
    mae = c(
        mean(abs(ewa$prediction - actual)),
        mean(abs(boa$prediction - actual)),
        mean(abs(mlpol$prediction - actual))
    )
)
write.csv(results, file.path(data_dir, "R_online_validation.csv"), row.names = FALSE)
cat("\nResults saved\n")
