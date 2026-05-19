* ============================================================
* Validacao Stata: Mincer-Zarnowitz Regression
* ============================================================

clear all
set more off

import delimited using "../data/inflation_forecasts.csv", clear
gen date_stata = monthly(substr(date, 1, 7), "YM")
format date_stata %tm
tsset date_stata

keep if _n >= 61

* --- Mincer-Zarnowitz para cada modelo ---
foreach model in fc_arima fc_ets fc_var fc_naive fc_drift {
    display _newline "=== MZ Regression: `model' ==="
    regress actual `model'

    * Teste H0: constante=0 e coef=1
    test (_cons = 0) (`model' = 1)
    display "F-stat: " r(F)
    display "p-value: " r(p)
}

* --- Exportar resultados do ARIMA como exemplo ---
regress actual fc_arima
local alpha = _b[_cons]
local beta = _b[fc_arima]
local r2 = e(r2)
test (_cons = 0) (fc_arima = 1)
local f_stat = r(F)
local f_pval = r(p)

clear
input str15 model float alpha float beta float r2 float f_stat float f_pvalue
"fc_arima" `alpha' `beta' `r2' `f_stat' `f_pval'
end
export delimited using "../data/stata_mz_validation.csv", replace
