* ============================================================
* Validacao Stata: ARIMA com selecao de ordem
* ============================================================

clear all
set more off

import delimited using "../data/airline.csv", clear
gen date_stata = monthly(substr(date, 1, 7), "YM")
format date_stata %tm
tsset date_stata

* --- Selecao de modelo via AIC ---
* Testar combinacoes (p,d,q) x (P,D,Q)12
local best_aic = .
local best_p = 0
local best_q = 0

* Primeira diferenca + diferenca sazonal
gen d_passengers = D.passengers
gen ds_passengers = D12.passengers
gen dds_passengers = D.D12.passengers

forvalues p = 0/3 {
    forvalues q = 0/3 {
        capture quietly arima passengers, arima(`p',1,`q') sarima(0,1,1,12)
        if _rc == 0 {
            quietly estat ic
            matrix ic = r(S)
            local this_aic = ic[1,5]
            if `this_aic' < `best_aic' {
                local best_aic = `this_aic'
                local best_p = `p'
                local best_q = `q'
            }
        }
    }
}

display "Best model: ARIMA(`best_p',1,`best_q')(0,1,1)[12]"
display "AIC: `best_aic'"

* --- Estimar melhor modelo ---
arima passengers, arima(`best_p',1,`best_q') sarima(0,1,1,12)
estat ic

* --- Previsao ---
predict fc_passengers, y dynamic(tm(1959m1))

* --- Exportar ---
preserve
keep if date_stata >= tm(1959m1)
keep date_stata passengers fc_passengers
export delimited using "../data/stata_arima_validation.csv", replace
restore
