# ============================================================
# Validacao R: Cross-Validation Temporal
# Pacotes: forecast
# ============================================================

library(forecast)

this_script <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", this_script[grep("--file=", this_script)])
if (length(script_path) == 0) script_path <- "."
data_dir <- file.path(dirname(script_path), "..", "data")
macro_br <- read.csv(file.path(data_dir, "macro_brazil.csv"))

inflation_br <- ts(macro_br$inflation, start = c(2010, 1), frequency = 12)

# --- Expanding window CV ---
min_train <- 60  # 5 anos minimo
h <- 12  # horizonte de 12 meses
n <- length(inflation_br)

expanding_errors <- c()
for (end_train in min_train:(n - h)) {
    train <- window(inflation_br, end = c(2010 + (end_train - 1) %/% 12, ((end_train - 1) %% 12) + 1))
    test_start <- end_train + 1
    test_vals <- inflation_br[test_start:(test_start + h - 1)]

    fc <- naive(train, h = h)
    expanding_errors <- c(expanding_errors, mean(abs(as.numeric(test_vals) - as.numeric(fc$mean))))
}

# --- Rolling window CV (window=60) ---
window_size <- 60
rolling_errors <- c()
for (start_idx in 1:(n - window_size - h + 1)) {
    end_idx <- start_idx + window_size - 1
    train <- inflation_br[start_idx:end_idx]
    train_ts <- ts(train, frequency = 12)
    test_vals <- inflation_br[(end_idx + 1):(end_idx + h)]

    fc <- naive(train_ts, h = h)
    rolling_errors <- c(rolling_errors, mean(abs(as.numeric(test_vals) - as.numeric(fc$mean))))
}

cv_results <- data.frame(
    strategy = c("expanding", "rolling_60"),
    mean_mae = c(mean(expanding_errors), mean(rolling_errors)),
    sd_mae = c(sd(expanding_errors), sd(rolling_errors)),
    n_folds = c(length(expanding_errors), length(rolling_errors))
)
write.csv(cv_results, file.path(data_dir, "R_cv_validation.csv"), row.names = FALSE)
cat("CV results saved\n")
print(cv_results)
