# ============================================================
# Validacao R: VAR com selecao automatica de lag
# Pacotes: vars
# ============================================================

library(vars)

data_dir <- file.path(dirname(sys.frame(1)$ofile), "..", "..", "basic_forecasting", "data")
macro_br <- read.csv(file.path(data_dir, "macro_brazil.csv"))

# --- Preparar dados multivariados ---
y <- ts(macro_br[, c("gdp_growth", "inflation")], start = c(2010, 1), frequency = 12)

# --- Selecao de lag ---
lag_select <- VARselect(y, lag.max = 12, type = "const")
cat("=== Selecao de Lag ===\n")
print(lag_select$selection)

# --- Estimar VAR ---
optimal_lag <- lag_select$selection["AIC(n)"]
fit <- VAR(y, p = optimal_lag, type = "const")
cat("\n=== VAR Summary ===\n")
print(summary(fit))

# --- Granger Causality ---
granger_gdp <- causality(fit, cause = "gdp_growth")
granger_inf <- causality(fit, cause = "inflation")

cat("\n=== Granger: GDP -> Inflation ===\n")
print(granger_gdp$Granger)
cat("\n=== Granger: Inflation -> GDP ===\n")
print(granger_inf$Granger)

# --- Previsao ---
fc <- predict(fit, n.ahead = 12)

results <- data.frame(
    selected_lag = optimal_lag,
    granger_gdp_pvalue = granger_gdp$Granger$p.value,
    granger_inf_pvalue = granger_inf$Granger$p.value
)
write.csv(results, file.path(dirname(sys.frame(1)$ofile), "..", "data", "R_auto_var_validation.csv"), row.names = FALSE)
cat("Results saved\n")
