# ============================================================
# Validacao R: Forecast Combination
# Pacotes: ForecastComb
# ============================================================

library(ForecastComb)

data_dir <- file.path(dirname(sys.frame(1)$ofile), "..", "data")
df <- read.csv(file.path(data_dir, "inflation_forecasts.csv"))

# --- Preparar dados para ForecastComb ---
actual <- df$actual
forecasts <- as.matrix(df[, c("fc_arima", "fc_ets", "fc_var", "fc_naive", "fc_drift")])

# Split treino/teste (primeiros 60 treino, ultimos 60 teste)
train_idx <- 1:60
test_idx <- 61:120

fc_data <- foreccomb(actual[train_idx], forecasts[train_idx, ],
                     actual[test_idx], forecasts[test_idx, ])

# --- Metodos de combinacao ---
sa <- comb_SA(fc_data)          # Simple Average
inv_w <- comb_InvW(fc_data)     # Inverse MSE weights
ols <- comb_OLS(fc_data)        # OLS (Granger-Ramanathan)
bg <- comb_BG(fc_data)          # Bates-Granger

cat("=== Simple Average ===\n")
cat(sprintf("MAE: %.6f\n", mean(abs(sa$Forecasts_Test - actual[test_idx]))))

cat("\n=== Inverse MSE ===\n")
cat(sprintf("MAE: %.6f\n", mean(abs(inv_w$Forecasts_Test - actual[test_idx]))))
cat(sprintf("Weights: %s\n", paste(round(inv_w$Weights, 4), collapse=", ")))

cat("\n=== OLS ===\n")
cat(sprintf("MAE: %.6f\n", mean(abs(ols$Forecasts_Test - actual[test_idx]))))
cat(sprintf("Weights: %s\n", paste(round(ols$Weights, 4), collapse=", ")))

cat("\n=== Bates-Granger ===\n")
cat(sprintf("MAE: %.6f\n", mean(abs(bg$Forecasts_Test - actual[test_idx]))))
cat(sprintf("Weights: %s\n", paste(round(bg$Weights, 4), collapse=", ")))

# --- Salvar resultados ---
results <- data.frame(
    method = c("simple_average", "inverse_mse", "ols", "bates_granger"),
    mae = c(
        mean(abs(sa$Forecasts_Test - actual[test_idx])),
        mean(abs(inv_w$Forecasts_Test - actual[test_idx])),
        mean(abs(ols$Forecasts_Test - actual[test_idx])),
        mean(abs(bg$Forecasts_Test - actual[test_idx]))
    )
)
write.csv(results, file.path(data_dir, "R_combination_validation.csv"), row.names = FALSE)
cat("\nResults saved\n")
