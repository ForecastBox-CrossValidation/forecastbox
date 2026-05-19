* ============================================================
* Validacao Stata: VAR Forecast e IRF
* ============================================================

clear all
set more off

import delimited using "../data/us_macro_quarterly.csv", clear
gen date_stata = quarterly(substr(date, 1, 7), "YQ")
format date_stata %tq
tsset date_stata

* --- Selecao de lag ---
varsoc gdp_growth inflation fed_funds unemployment, maxlag(8)

* --- Estimar VAR ---
var gdp_growth inflation fed_funds unemployment, lags(1/4)

* --- Previsao incondicional ---
fcast compute fc_, step(8)

* --- IRF ---
irf create var_irf, step(8) set(irf_results)
irf table oirf, impulse(fed_funds) response(gdp_growth inflation unemployment)

* --- Exportar previsao ---
keep date_stata gdp_growth inflation fed_funds unemployment fc_*
export delimited using "../data/stata_var_forecast.csv", replace

* --- Exportar IRF ---
irf table oirf, impulse(fed_funds) response(gdp_growth) saving("../data/stata_irf_ff_gdp.csv")
