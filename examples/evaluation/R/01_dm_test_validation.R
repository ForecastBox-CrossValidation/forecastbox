# ============================================================
# Validacao R: Diebold-Mariano Test
# Pacotes: forecast
# ============================================================

library(forecast)

data_dir <- file.path(dirname(sys.frame(1)$ofile), "..", "data")
df <- read.csv(file.path(data_dir, "inflation_forecasts.csv"))

actual <- df$actual
fc_arima <- df$fc_arima
fc_ets <- df$fc_ets
fc_var <- df$fc_var
fc_naive <- df$fc_naive
fc_drift <- df$fc_drift

# --- Test set (second half) ---
test_idx <- 61:120
a <- actual[test_idx]

models <- list(arima = fc_arima[test_idx], ets = fc_ets[test_idx],
               var = fc_var[test_idx], naive = fc_naive[test_idx],
               drift = fc_drift[test_idx])

model_names <- names(models)
n_models <- length(models)

# --- Pairwise DM tests ---
dm_pvalues <- matrix(NA, n_models, n_models, dimnames = list(model_names, model_names))

for (i in 1:(n_models - 1)) {
    for (j in (i + 1):n_models) {
        e1 <- a - models[[i]]
        e2 <- a - models[[j]]
        dm <- dm.test(e1, e2, alternative = "two.sided", h = 1)
        dm_pvalues[i, j] <- dm$p.value
        dm_pvalues[j, i] <- dm$p.value
    }
}

cat("=== DM Pairwise P-values ===\n")
print(round(dm_pvalues, 4))

# --- Salvar ---
write.csv(dm_pvalues, file.path(data_dir, "R_dm_pvalues.csv"))
cat("Results saved\n")
