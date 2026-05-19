* ============================================================
* Validacao Stata: Forecast Combination
* ============================================================

clear all
set more off

import delimited using "../data/inflation_forecasts.csv", clear
gen date_stata = monthly(substr(date, 1, 7), "YM")
format date_stata %tm
tsset date_stata

local n = _N
local train_end = 60

* --- 1. Simple Average ---
gen fc_simple_avg = (fc_arima + fc_ets + fc_var + fc_naive + fc_drift) / 5

* --- 2. Inverse MSE Weights (calculados no treino) ---
* Calcular MSE de cada modelo no treino
foreach model in arima ets var naive drift {
    gen sq_err_`model' = (actual - fc_`model')^2 if _n <= `train_end'
}

foreach model in arima ets var naive drift {
    quietly summarize sq_err_`model' if _n <= `train_end'
    local mse_`model' = r(mean)
    local inv_mse_`model' = 1 / `mse_`model''
}

local sum_inv = `inv_mse_arima' + `inv_mse_ets' + `inv_mse_var' + `inv_mse_naive' + `inv_mse_drift'

local w_arima = `inv_mse_arima' / `sum_inv'
local w_ets = `inv_mse_ets' / `sum_inv'
local w_var = `inv_mse_var' / `sum_inv'
local w_naive = `inv_mse_naive' / `sum_inv'
local w_drift = `inv_mse_drift' / `sum_inv'

gen fc_inv_mse = `w_arima' * fc_arima + `w_ets' * fc_ets + `w_var' * fc_var + `w_naive' * fc_naive + `w_drift' * fc_drift

display "Inverse MSE weights:"
display "  ARIMA: `w_arima'"
display "  ETS: `w_ets'"
display "  VAR: `w_var'"
display "  Naive: `w_naive'"
display "  Drift: `w_drift'"

* --- 3. OLS Combination (Granger-Ramanathan) ---
regress actual fc_arima fc_ets fc_var fc_naive fc_drift if _n <= `train_end'
predict fc_ols, xb

* --- Metricas no teste ---
foreach method in simple_avg inv_mse ols {
    gen err_`method' = abs(actual - fc_`method') if _n > `train_end'
    quietly summarize err_`method' if _n > `train_end'
    display "MAE `method': " r(mean)
}

* --- Exportar ---
preserve
collapse (mean) err_simple_avg err_inv_mse err_ols if _n > `train_end'
rename err_simple_avg mae_simple_avg
rename err_inv_mse mae_inv_mse
rename err_ols mae_ols
export delimited using "../data/stata_combination_validation.csv", replace
restore
