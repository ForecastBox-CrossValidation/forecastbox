* ============================================================
* Validacao Stata: Bridge Equations
* ============================================================
* Este script replica a estimacao de bridge equations feita
* pelo forecastbox para fins de validacao cruzada.
*
* Dados: ../data/mixed_freq.csv (mesmo dataset dos notebooks Python)
*
* Limitacoes do Stata para nowcasting com bridge equations:
* - Nao possui funcionalidade nativa de bridge equations;
*   requer montagem manual via collapse + merge + regress.
* - Temporal aggregation deve ser feita manualmente (collapse),
*   enquanto forecastbox trata isso automaticamente.
* - Nao suporta ragged-edge nativamente: observacoes faltantes
*   nos indicadores mensais requerem tratamento ad hoc.
* - Nao possui framework integrado de avaliacao out-of-sample
*   (rolling window, expanding window) para nowcasting.
* - Combinacao de forecasts de multiplas bridges nao e nativa.
* ============================================================

clear all
set more off

import delimited using "../data/mixed_freq.csv", clear
gen date_stata = monthly(substr(date, 1, 7), "YM")
format date_stata %tm
tsset date_stata

* --- Temporal aggregation: monthly -> quarterly ---
gen quarter = qofd(dofm(date_stata))
format quarter %tq

* Aggregate monthly indicators to quarterly
preserve
collapse (mean) industrial_production retail_sales confidence_index, by(quarter)
tempfile quarterly_indicators
save `quarterly_indicators'
restore

* Get quarterly GDP
keep if gdp_growth != .
gen quarter2 = qofd(dofm(date_stata))
keep quarter2 gdp_growth
rename quarter2 quarter

merge 1:1 quarter using `quarterly_indicators'
keep if _merge == 3
drop _merge

tsset quarter

* --- Single indicator bridge ---
regress gdp_growth industrial_production
display "R2 (IP only): " e(r2)
predict fc_bridge_ip, xb

* --- Multi-indicator bridge ---
regress gdp_growth industrial_production retail_sales confidence_index
display "R2 (all indicators): " e(r2)
predict fc_bridge_all, xb

* --- Exportar ---
gen err_ip = abs(gdp_growth - fc_bridge_ip)
gen err_all = abs(gdp_growth - fc_bridge_all)
quietly summarize err_ip
local mae_ip = r(mean)
quietly summarize err_all
local mae_all = r(mean)

clear
input str20 model float mae float r2
"bridge_ip" `mae_ip' .
"bridge_all" `mae_all' .
end
export delimited using "../data/stata_bridge_validation.csv", replace
