# ============================================================
# Validacao R: DFM Nowcasting
# Pacotes: nowcasting (ou statsmodels via reticulate)
# ============================================================

# Nota: O pacote 'nowcasting' pode nao estar no CRAN em todas as versoes.
# Alternativa: usar DFM via estimacao manual com EM.

library(stats)

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
monthly_vars <- c("industrial_production", "retail_sales", "confidence_index")
X <- as.matrix(df[, monthly_vars])

# Standardize
X_std <- scale(X)

# --- PCA como proxy para DFM ---
pca <- prcomp(X_std, center = FALSE, scale. = FALSE)
factors <- pca$x[, 1:2]  # 2 fatores

cat("=== PCA Loadings ===\n")
print(pca$rotation[, 1:2])
cat(sprintf("\nVariance explained: %.2f%%, %.2f%%\n",
    summary(pca)$importance[2, 1] * 100,
    summary(pca)$importance[2, 2] * 100))

# --- Nowcast GDP via regressao nos fatores ---
gdp <- df$gdp_growth
quarterly_idx <- which(!is.na(gdp))
gdp_q <- gdp[quarterly_idx]
factors_q <- factors[quarterly_idx, ]

fit <- lm(gdp_q ~ factors_q)
cat("\n=== GDP ~ Factors Regression ===\n")
print(summary(fit))

# Nowcast (ultimo trimestre)
nowcast_val <- fitted(fit)[length(gdp_q)]
cat(sprintf("\nNowcast GDP: %.4f\n", nowcast_val))

# --- Salvar ---
results <- data.frame(
    factor1_var_explained = summary(pca)$importance[2, 1],
    factor2_var_explained = summary(pca)$importance[2, 2],
    r_squared = summary(fit)$r.squared,
    nowcast_gdp = nowcast_val
)
write.csv(results, file.path(data_dir, "R_dfm_validation.csv"), row.names = FALSE)
cat("Results saved\n")
