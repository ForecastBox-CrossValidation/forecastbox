* ============================================================
* Validacao Stata: Stress Test via Scenario Modification
* ============================================================

clear all
set more off

import delimited using "../data/us_macro_quarterly.csv", clear
gen date_stata = quarterly(substr(date, 1, 7), "YQ")
format date_stata %tq
tsset date_stata

* --- Estimar VAR ---
var gdp_growth inflation fed_funds unemployment, lags(1/4)

* --- Baseline forecast ---
fcast compute fc_base_, step(8)

* --- Stress scenario: Fed funds +200bps ---
* Modify forecast by adding shock via manual adjustment
gen fc_stress_ff = fc_base_fed_funds + 2.0 if fc_base_fed_funds != .

* Approximate GDP impact via IRF coefficient
* (simplified - real conditional forecast would use Waggoner-Zha)
irf create stress_irf, step(8) set(stress_results)

display "Baseline and stress forecasts computed"
display "Compare fc_base_* vs stressed values"

* --- Exportar ---
keep if fc_base_gdp_growth != .
keep date_stata fc_base_gdp_growth fc_base_inflation fc_base_fed_funds fc_base_unemployment fc_stress_ff
export delimited using "../data/stata_stress_test.csv", replace
