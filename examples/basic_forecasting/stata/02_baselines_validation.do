* ============================================================
* Validacao Stata: Modelos Baseline
* ============================================================

clear all
set more off

import delimited using "../data/macro_brazil.csv", clear
gen date_stata = monthly(substr(date, 1, 7), "YM")
format date_stata %tm
tsset date_stata

local n = _N
local train_end = `n' - 24
local h = 24

* --- Naive: ultimo valor ---
local naive_val = gdp_growth[`train_end']
gen fc_naive = `naive_val' if _n > `train_end'

* --- Mean: media do treino ---
quietly summarize gdp_growth if _n <= `train_end'
gen fc_mean = r(mean) if _n > `train_end'

* --- Drift: regressao linear no tempo ---
gen time_idx = _n
quietly regress gdp_growth time_idx if _n <= `train_end'
predict fc_drift_all, xb
gen fc_drift = fc_drift_all if _n > `train_end'

* --- Seasonal Naive: valor de 12 meses atras ---
gen fc_snaive = gdp_growth[_n - 12] if _n > `train_end'

* --- Calcular MAE para cada modelo ---
foreach model in naive mean drift snaive {
    gen err_`model' = abs(gdp_growth - fc_`model') if _n > `train_end'
    quietly summarize err_`model' if _n > `train_end'
    display "MAE `model': " r(mean)
}

* --- Exportar ---
preserve
collapse (mean) err_naive err_mean err_drift err_snaive if _n > `train_end'
rename err_naive mae_naive
rename err_mean mae_mean
rename err_drift mae_drift
rename err_snaive mae_snaive
export delimited using "../data/stata_baselines_validation.csv", replace
restore
