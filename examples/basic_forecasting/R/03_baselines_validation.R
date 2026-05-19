# ============================================================
# Validacao R: Modelos Baseline
# Pacotes: forecast
# ============================================================

library(forecast)

this_script <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", this_script[grep("--file=", this_script)])
if (length(script_path) == 0) script_path <- "."
data_dir <- file.path(dirname(script_path), "..", "data")
macro_br <- read.csv(file.path(data_dir, "macro_brazil.csv"))

gdp_br <- ts(macro_br$gdp_growth, start = c(2010, 1), frequency = 12)
n <- length(gdp_br)
h <- 24

train <- window(gdp_br, end = c(2010 + (n - h - 1) %/% 12, ((n - h - 1) %% 12) + 1))
test <- tail(gdp_br, h)

# --- Modelos baseline ---
fc_naive <- naive(train, h = h)
fc_snaive <- snaive(train, h = h)
fc_drift <- rwf(train, h = h, drift = TRUE)
fc_mean <- meanf(train, h = h)

# --- Metricas ---
calc_metrics <- function(actual, predicted, train_data) {
    mae_val <- mean(abs(actual - predicted))
    rmse_val <- sqrt(mean((actual - predicted)^2))
    scale_factor <- mean(abs(diff(as.numeric(train_data))))
    mase_val <- mae_val / scale_factor
    c(MAE = mae_val, RMSE = rmse_val, MASE = mase_val)
}

actual <- as.numeric(test)
results <- rbind(
    data.frame(model = "naive", t(calc_metrics(actual, as.numeric(fc_naive$mean), train))),
    data.frame(model = "snaive", t(calc_metrics(actual, as.numeric(fc_snaive$mean), train))),
    data.frame(model = "drift", t(calc_metrics(actual, as.numeric(fc_drift$mean), train))),
    data.frame(model = "mean", t(calc_metrics(actual, as.numeric(fc_mean$mean), train)))
)

write.csv(results, file.path(data_dir, "R_baselines_validation.csv"), row.names = FALSE)
cat("Baseline results saved\n")
print(results)
