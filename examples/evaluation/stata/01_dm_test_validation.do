* ============================================================
* Validacao Stata: Diebold-Mariano Test (manual)
* ============================================================

clear all
set more off

import delimited using "../data/inflation_forecasts.csv", clear
gen date_stata = monthly(substr(date, 1, 7), "YM")
format date_stata %tm
tsset date_stata

* --- Usar apenas teste (obs 61-120) ---
keep if _n >= 61

* --- Loss differentials (squared errors) ---
gen e_arima2 = (actual - fc_arima)^2
gen e_ets2 = (actual - fc_ets)^2
gen e_var2 = (actual - fc_var)^2
gen e_naive2 = (actual - fc_naive)^2
gen e_drift2 = (actual - fc_drift)^2

* --- DM test: ARIMA vs ETS ---
gen d_arima_ets = e_arima2 - e_ets2
newey d_arima_ets, lag(1)
* t-statistic on constant = DM statistic
display "DM(ARIMA vs ETS) t-stat: " _b[_cons] / _se[_cons]
display "DM(ARIMA vs ETS) p-value: " 2 * ttail(e(N) - 1, abs(_b[_cons] / _se[_cons]))

* --- DM test: ARIMA vs Naive ---
gen d_arima_naive = e_arima2 - e_naive2
newey d_arima_naive, lag(1)
display "DM(ARIMA vs Naive) t-stat: " _b[_cons] / _se[_cons]
display "DM(ARIMA vs Naive) p-value: " 2 * ttail(e(N) - 1, abs(_b[_cons] / _se[_cons]))

* --- DM test: ETS vs Naive ---
gen d_ets_naive = e_ets2 - e_naive2
newey d_ets_naive, lag(1)
display "DM(ETS vs Naive) t-stat: " _b[_cons] / _se[_cons]
display "DM(ETS vs Naive) p-value: " 2 * ttail(e(N) - 1, abs(_b[_cons] / _se[_cons]))

* --- Exportar ---
clear
input str30 comparison float dm_tstat float dm_pvalue
end
* (valores preenchidos pelo output acima - formato de referencia)
export delimited using "../data/stata_dm_validation.csv", replace
