* ============================================================
* Validacao Stata: Exponential Smoothing
* ============================================================

clear all
set more off

import delimited using "../data/airline.csv", clear
gen date_stata = monthly(substr(date, 1, 7), "YM")
format date_stata %tm
tsset date_stata

* --- Single exponential smoothing ---
tssmooth exponential sm_single = passengers

* --- Double (Holt) ---
tssmooth shinters sm_holt = passengers

* --- Holt-Winters (aditivo e multiplicativo nao disponivel em todas versoes) ---
* Holt-Winters com sazonalidade
tssmooth hwinters sm_hw = passengers, sn0(12)

* --- Comparar erro no periodo de teste ---
local n = _N
local train_end = `n' - 24

gen in_test = (_n > `train_end')
gen err_single = abs(passengers - sm_single) if in_test
gen err_holt = abs(passengers - sm_holt) if in_test
gen err_hw = abs(passengers - sm_hw) if in_test

quietly summarize err_single if in_test
local mae_single = r(mean)
quietly summarize err_holt if in_test
local mae_holt = r(mean)
quietly summarize err_hw if in_test
local mae_hw = r(mean)

display "MAE Single: `mae_single'"
display "MAE Holt: `mae_holt'"
display "MAE HW: `mae_hw'"

* --- Exportar ---
clear
input str20 model float mae
"single_exp" `mae_single'
"holt" `mae_holt'
"holt_winters" `mae_hw'
end
export delimited using "../data/stata_ets_validation.csv", replace
