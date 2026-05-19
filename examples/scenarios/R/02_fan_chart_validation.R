# ============================================================
# Validacao R: Fan Chart via Bootstrap
# Pacotes: vars
# ============================================================

library(vars)

data_dir <- file.path(dirname(sys.frame(1)$ofile), "..", "data")
df <- read.csv(file.path(data_dir, "us_macro_quarterly.csv"))

y <- ts(df[, c("gdp_growth", "inflation", "fed_funds", "unemployment")],
        start = c(1990, 1), frequency = 4)

fit <- VAR(y, p = 4, type = "const")

# --- Bootstrap forecast (for fan chart) ---
fc <- predict(fit, n.ahead = 8, ci = 0.95)

# Extract GDP growth quantiles
gdp_fc <- fc$fcst$gdp_growth
fan_data <- data.frame(
    horizon = 1:8,
    mean = gdp_fc[, "fcst"],
    lower_95 = gdp_fc[, "lower"],
    upper_95 = gdp_fc[, "upper"]
)

write.csv(fan_data, file.path(data_dir, "R_fan_chart_validation.csv"), row.names = FALSE)
cat("Fan chart data saved\n")
print(fan_data)
