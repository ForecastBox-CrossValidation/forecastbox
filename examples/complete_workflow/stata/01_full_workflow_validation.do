* ============================================================
* Validacao Stata: Full Forecasting Workflow
* ============================================================

clear all
set more off

import delimited using "../data/macro_brazil.csv", clear
gen date_stata = monthly(substr(date, 1, 7), "YM")
format date_stata %tm
tsset date_stata

local n = _N
local train_end = `n' - 24
local h = 12

display "=== Step 1: Data ==="
display "Total obs: `n', Training: `train_end', Test: `h'"

* === STEP 2: Fit Models ===

* ARIMA(1,0,1) - approximation of auto.arima
arima inflation if _n <= `train_end', arima(1,0,1)
predict fc_arima, y dynamic(`=`train_end'+1')

* Naive
local last_val = inflation[`train_end']
gen fc_naive = `last_val' if _n > `train_end'

* Seasonal Naive
gen fc_snaive = inflation[_n - 12] if _n > `train_end'

* Mean
quietly summarize inflation if _n <= `train_end'
gen fc_mean = r(mean) if _n > `train_end'

display _newline "=== Step 2: Models Fitted ==="

* === STEP 3: Combination ===
gen fc_simple_avg = (fc_arima + fc_naive + fc_snaive + fc_mean) / 4 if _n > `train_end'

display _newline "=== Step 3: Combination ==="

* === STEP 4: Evaluation ===
foreach model in arima naive snaive mean simple_avg {
    gen err_`model' = abs(inflation - fc_`model') if _n > `train_end' & _n <= `=`train_end'+`h''
    quietly summarize err_`model' if _n > `train_end' & _n <= `=`train_end'+`h''
    display "MAE `model': " r(mean)
}

* DM test: ARIMA vs Naive
gen dm_loss = (inflation - fc_arima)^2 - (inflation - fc_naive)^2 if _n > `train_end' & _n <= `=`train_end'+`h''
newey dm_loss if _n > `train_end' & _n <= `=`train_end'+`h'', lag(1)
display _newline "DM(ARIMA vs Naive) t-stat: " _b[_cons] / _se[_cons]

* === STEP 5: VAR Scenarios ===
display _newline "=== Step 5: VAR Scenarios ==="
var gdp_growth inflation interest_rate if _n <= `train_end', lags(1/4)

* IRF
irf create workflow_irf, step(12) set(workflow_irf_results)
irf table oirf, impulse(interest_rate) response(inflation)

* Forecast
fcast compute fc_var_, step(12)

* === EXPORT ===
preserve
keep if _n > `train_end' & _n <= `=`train_end'+`h''
collapse (mean) err_arima err_naive err_snaive err_mean err_simple_avg
foreach v of varlist err_* {
    rename `v' mae`=substr("`v'", 4, .)'
}
export delimited using "../data/stata_full_workflow_validation.csv", replace
restore

display _newline "=== Results Saved ==="
