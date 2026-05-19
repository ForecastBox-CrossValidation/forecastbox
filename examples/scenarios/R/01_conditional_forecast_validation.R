# ============================================================
# Validacao R: Conditional Forecast via VAR
# Pacotes: vars
# ============================================================

library(vars)

data_dir <- file.path(dirname(sys.frame(1)$ofile), "..", "data")
df <- read.csv(file.path(data_dir, "us_macro_quarterly.csv"))

y <- ts(df[, c("gdp_growth", "inflation", "fed_funds", "unemployment")],
        start = c(1990, 1), frequency = 4)

# --- Estimar VAR ---
lag_sel <- VARselect(y, lag.max = 8, type = "const")
optimal_lag <- lag_sel$selection["AIC(n)"]
cat(sprintf("Optimal lag (AIC): %d\n", optimal_lag))

fit <- VAR(y, p = optimal_lag, type = "const")

# --- Previsao incondicional ---
fc_uncond <- predict(fit, n.ahead = 8)
cat("\n=== Unconditional Forecast ===\n")
print(fc_uncond$fcst$gdp_growth)

# --- Impulse Response Functions (proxy for conditional) ---
irf_ff <- irf(fit, impulse = "fed_funds", response = c("gdp_growth", "inflation", "unemployment"),
              n.ahead = 8, boot = TRUE, runs = 500)
cat("\n=== IRF: Fed Funds -> GDP ===\n")
print(irf_ff$irf$fed_funds)

# --- Salvar resultados ---
uncond_df <- data.frame(
    horizon = 1:8,
    gdp_growth = fc_uncond$fcst$gdp_growth[, "fcst"],
    inflation = fc_uncond$fcst$inflation[, "fcst"],
    fed_funds = fc_uncond$fcst$fed_funds[, "fcst"],
    unemployment = fc_uncond$fcst$unemployment[, "fcst"]
)
write.csv(uncond_df, file.path(data_dir, "R_unconditional_forecast.csv"), row.names = FALSE)

irf_df <- data.frame(
    horizon = 0:8,
    irf_gdp = irf_ff$irf$fed_funds[, "gdp_growth"],
    irf_inflation = irf_ff$irf$fed_funds[, "inflation"],
    irf_unemployment = irf_ff$irf$fed_funds[, "unemployment"]
)
write.csv(irf_df, file.path(data_dir, "R_irf_fed_funds.csv"), row.names = FALSE)
cat("Results saved\n")
