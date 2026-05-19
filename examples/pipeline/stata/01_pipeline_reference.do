* ============================================================
* Referencia Stata: Pipeline de Forecasting
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

* === STEP 1: Prepare Data ===
display "Total observations: `n'"
display "Training set: 1 to `train_end'"
display "Test set: `=`train_end'+1' to `n'"

* === STEP 2: Fit Models (on training data) ===

* ARIMA
arima gdp_growth if _n <= `train_end', arima(1,0,1)
predict fc_arima, y dynamic(`=`train_end'+1')

* Naive (last observation)
local last_val = gdp_growth[`train_end']
gen fc_naive = `last_val' if _n > `train_end'

* Mean
quietly summarize gdp_growth if _n <= `train_end'
gen fc_mean = r(mean) if _n > `train_end'

* === STEP 3: Combine (simple average) ===
gen fc_combined = (fc_arima + fc_naive + fc_mean) / 3 if _n > `train_end'

* === STEP 4: Evaluate ===
foreach model in arima naive mean combined {
    gen err_`model' = abs(gdp_growth - fc_`model') if _n > `train_end' & _n <= `=`train_end'+`h''
    quietly summarize err_`model' if _n > `train_end' & _n <= `=`train_end'+`h''
    display "MAE `model': " r(mean)
}

* === STEP 5: Export Results ===
preserve
keep if _n > `train_end' & _n <= `=`train_end'+`h''
collapse (mean) err_arima err_naive err_mean err_combined
rename err_arima mae_arima
rename err_naive mae_naive
rename err_mean mae_mean
rename err_combined mae_combined
export delimited using "../data/stata_pipeline_reference.csv", replace
restore

* === STEP 6: Rolling Estimation ===
gen rolling_error = .
local start_roll = `train_end'
local end_roll = `n' - 1

forvalues t = `start_roll'/`end_roll' {
    capture quietly arima gdp_growth if _n <= `t', arima(1,0,1)
    if _rc == 0 {
        quietly predict temp_fc, y dynamic(`=`t'+1')
        local fc_val = temp_fc[`=`t'+1']
        local actual_val = gdp_growth[`=`t'+1']
        quietly replace rolling_error = abs(`actual_val' - `fc_val') in `=`t'-`start_roll'+1'
        drop temp_fc
    }
}

quietly summarize rolling_error if rolling_error != .
display "Rolling 1-step MAE (ARIMA): " r(mean) " (sd: " r(sd) ")"
