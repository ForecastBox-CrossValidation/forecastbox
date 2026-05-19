* ============================================================
* Validacao Stata: Metricas de Forecast (MAE, RMSE, MAPE)
* ============================================================

clear all
set more off

* --- Carregar dados ---
import delimited using "../data/macro_brazil.csv", clear
gen date_stata = monthly(substr(date, 1, 7), "YM")
format date_stata %tm
tsset date_stata

* --- Separar treino/teste (ultimos 24 meses) ---
gen obs_n = _n
local n = _N
local train_end = `n' - 24

* --- Naive forecast: ultimo valor do treino ---
local last_train_val = gdp_growth[`train_end']
gen naive_forecast = `last_train_val' if obs_n > `train_end'

* --- Calcular erros (apenas no teste) ---
gen error = gdp_growth - naive_forecast if obs_n > `train_end'
gen abs_error = abs(error)
gen sq_error = error^2
gen pct_error = abs(error / gdp_growth) * 100 if gdp_growth != 0 & obs_n > `train_end'

* --- Metricas ---
quietly summarize abs_error if obs_n > `train_end'
local mae = r(mean)
display "MAE: " `mae'

quietly summarize sq_error if obs_n > `train_end'
local rmse = sqrt(r(mean))
display "RMSE: " `rmse'

quietly summarize pct_error if obs_n > `train_end'
local mape = r(mean)
display "MAPE: " `mape'

* --- Exportar resultados ---
clear
input str10 metric float value
"MAE" `mae'
"RMSE" `rmse'
"MAPE" `mape'
end
export delimited using "../data/stata_metrics_validation.csv", replace
