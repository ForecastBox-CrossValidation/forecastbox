# ============================================================
# Validacao R: Model Confidence Set
# Pacotes: MCS
# ============================================================

library(MCS)

data_dir <- file.path(dirname(sys.frame(1)$ofile), "..", "data")
df <- read.csv(file.path(data_dir, "inflation_forecasts.csv"))

actual <- df$actual[61:120]
fc_arima <- df$fc_arima[61:120]
fc_ets <- df$fc_ets[61:120]
fc_var <- df$fc_var[61:120]
fc_naive <- df$fc_naive[61:120]
fc_drift <- df$fc_drift[61:120]

# --- Loss matrix (squared errors) ---
losses <- cbind(
    arima = (actual - fc_arima)^2,
    ets = (actual - fc_ets)^2,
    var = (actual - fc_var)^2,
    naive = (actual - fc_naive)^2,
    drift = (actual - fc_drift)^2
)

# --- MCS ---
mcs_result <- MCSprocedure(losses, alpha = 0.10, B = 5000, statistic = "Tmax")

cat("=== MCS Results (alpha=0.10) ===\n")
print(mcs_result)

# --- Salvar ---
mcs_df <- data.frame(
    model = colnames(losses),
    in_mcs_10 = colnames(losses) %in% names(mcs_result@show),
    loss_mean = colMeans(losses)
)
write.csv(mcs_df, file.path(data_dir, "R_mcs_validation.csv"), row.names = FALSE)
cat("Results saved\n")
