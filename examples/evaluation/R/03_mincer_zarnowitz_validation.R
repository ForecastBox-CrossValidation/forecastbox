# ============================================================
# Validacao R: Mincer-Zarnowitz Regression
# Pacotes: lmtest, car
# ============================================================

library(lmtest)
library(car)

data_dir <- file.path(dirname(sys.frame(1)$ofile), "..", "data")
df <- read.csv(file.path(data_dir, "inflation_forecasts.csv"))

actual <- df$actual[61:120]

# --- Mincer-Zarnowitz: y = alpha + beta * yhat ---
models <- c("fc_arima", "fc_ets", "fc_var", "fc_naive", "fc_drift")
results <- data.frame()

for (m in models) {
    yhat <- df[[m]][61:120]
    fit <- lm(actual ~ yhat)

    # Test H0: alpha=0, beta=1 (joint F-test)
    ftest <- linearHypothesis(fit, c("(Intercept) = 0", "yhat = 1"))

    results <- rbind(results, data.frame(
        model = m,
        alpha = coef(fit)[1],
        beta = coef(fit)[2],
        r_squared = summary(fit)$r.squared,
        f_statistic = ftest$F[2],
        f_pvalue = ftest$`Pr(>F)`[2]
    ))
}

cat("=== Mincer-Zarnowitz Results ===\n")
print(results)

write.csv(results, file.path(data_dir, "R_mincer_zarnowitz_validation.csv"), row.names = FALSE)
cat("Results saved\n")
