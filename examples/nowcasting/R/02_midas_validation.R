# ============================================================
# Validacao R: MIDAS Regression
# Pacotes: midasr
# ============================================================

library(midasr)

# Resolve script directory robustly
args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args[grep("--file=", args)])
if (length(script_path) > 0) {
    script_dir <- dirname(script_path)
} else {
    script_dir <- "."
}
data_dir <- normalizePath(file.path(script_dir, "..", "data"))
df <- read.csv(file.path(data_dir, "mixed_freq.csv"))

# --- Preparar dados ---
gdp <- df$gdp_growth
ip <- df$industrial_production

# Quarterly GDP (non-NA values)
gdp_q <- gdp[!is.na(gdp)]
n_q <- length(gdp_q)

# Monthly IP reshaped to (n_quarters, 3)
ip_matrix <- matrix(ip, ncol = 3, byrow = TRUE)

# --- MIDAS with Exponential Almon ---
# y_q ~ midas(ip_monthly, nealmon, 3 lags per quarter)
fit_almon <- midas_r(gdp_q ~ mls(ip_matrix, 1:3, 3, nealmon),
                     start = list(ip_matrix = c(0.1, -0.01)))

cat("=== MIDAS (Exp Almon) ===\n")
print(summary(fit_almon))

# --- U-MIDAS (unrestricted) ---
fit_umidas <- lm(gdp_q ~ ip_matrix)
cat("\n=== U-MIDAS ===\n")
print(summary(fit_umidas))

# --- Salvar ---
results <- data.frame(
    method = c("exp_almon", "u_midas"),
    r_squared = c(summary(fit_almon)$r.squared, summary(fit_umidas)$r.squared),
    aic = c(AIC(fit_almon), AIC(fit_umidas))
)
write.csv(results, file.path(data_dir, "R_midas_validation.csv"), row.names = FALSE)
cat("Results saved\n")
