* ============================================================
* Validacao Stata: VAR com selecao de lag
* ============================================================

clear all
set more off

import delimited using "../../basic_forecasting/data/macro_brazil.csv", clear
gen date_stata = monthly(substr(date, 1, 7), "YM")
format date_stata %tm
tsset date_stata

* --- Selecao de lag ---
varsoc gdp_growth inflation, maxlag(12)

* --- Estimar VAR com lag selecionado ---
* (usar lag sugerido pelo AIC)
var gdp_growth inflation, lags(1/4)

* --- Granger causality ---
vargranger

* --- Previsao ---
fcast compute fc_, step(12)

* --- Exportar ---
keep date_stata gdp_growth inflation fc_gdp_growth fc_inflation
export delimited using "../data/stata_var_validation.csv", replace
