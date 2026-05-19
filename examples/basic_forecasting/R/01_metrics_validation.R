# ============================================================
# Validacao R: Metricas de Forecast (MAE, RMSE, MAPE, MASE)
# Pacotes: forecast, Metrics
# ============================================================

library(forecast)
library(Metrics)

# --- Carregar dados ---
this_script <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", this_script[grep("--file=", this_script)])
if (length(script_path) == 0) script_path <- "."
data_dir <- file.path(dirname(script_path), "..", "data")
macro_br <- read.csv(file.path(data_dir, "macro_brazil.csv"))
macro_us <- read.csv(file.path(data_dir, "macro_us.csv"))

# --- Criar serie temporal ---
gdp_br <- ts(macro_br$gdp_growth, start = c(2010, 1), frequency = 12)
inflation_br <- ts(macro_br$inflation, start = c(2010, 1), frequency = 12)

# --- Naive forecast (ultimos 24 meses como teste) ---
train_end <- length(gdp_br) - 24
train <- window(gdp_br, end = c(2010 + (train_end - 1) %/% 12, ((train_end - 1) %% 12) + 1))
test <- window(gdp_br, start = c(2010 + train_end %/% 12, (train_end %% 12) + 1))

naive_fc <- naive(train, h = 24)

# --- Calcular metricas ---
actual <- as.numeric(test)
predicted <- as.numeric(naive_fc$mean)

mae_val <- mae(actual, predicted)
rmse_val <- rmse(actual, predicted)
mape_val <- mape(actual, predicted)

# MASE manual (como no forecast package)
scale_factor <- mean(abs(diff(as.numeric(train))))
mase_val <- mean(abs(actual - predicted)) / scale_factor

cat(sprintf("MAE:  %.6f\n", mae_val))
cat(sprintf("RMSE: %.6f\n", rmse_val))
cat(sprintf("MAPE: %.6f\n", mape_val))
cat(sprintf("MASE: %.6f\n", mase_val))

# --- Salvar resultados ---
results <- data.frame(
    metric = c("MAE", "RMSE", "MAPE", "MASE"),
    value = c(mae_val, rmse_val, mape_val, mase_val),
    model = "naive",
    variable = "gdp_growth",
    dataset = "macro_brazil"
)
write.csv(results, file.path(data_dir, "R_metrics_validation.csv"), row.names = FALSE)
cat("Results saved to R_metrics_validation.csv\n")
