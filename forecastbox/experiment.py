"""ForecastExperiment - High-level forecasting workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from forecastbox._logging import get_logger
from forecastbox.core.forecast import Forecast

logger = get_logger("experiment")


@dataclass
class ExperimentResults:
    """Container for ForecastExperiment results.

    Attributes
    ----------
    forecasts : dict[str, Forecast]
        Individual forecasts by model name.
    combination : Forecast | None
        Combined forecast.
    evaluation : pd.DataFrame | None
        Metrics and test results.
    scenarios : dict[str, Any] | None
        Scenario analysis results.
    cv_results : dict[str, Any] | None
        Cross-validation results per model.
    ranking : pd.DataFrame | None
        Model ranking by metrics.
    mcs : Any | None
        Model Confidence Set result.
    metadata : dict[str, Any]
        Additional metadata.
    """

    forecasts: dict[str, Forecast] = field(default_factory=dict)
    combination: Forecast | None = None
    evaluation: pd.DataFrame | None = None
    scenarios: dict[str, Any] | None = None
    cv_results: dict[str, Any] | None = None
    ranking: pd.DataFrame | None = None
    mcs: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, directory: str | Path) -> None:
        """Save all results to a directory.

        Parameters
        ----------
        directory : str or Path
            Output directory. Created if not exists.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Save individual forecasts
        forecasts_dir = directory / "forecasts"
        forecasts_dir.mkdir(exist_ok=True)
        for name, fc in self.forecasts.items():
            safe_name = name.replace("/", "_").replace(" ", "_")
            fc.save(forecasts_dir / f"{safe_name}.json")

        # Save combination
        if self.combination is not None:
            self.combination.save(directory / "combination.json")

        # Save evaluation
        if self.evaluation is not None:
            self.evaluation.to_csv(directory / "evaluation.csv")

        # Save ranking
        if self.ranking is not None:
            self.ranking.to_csv(directory / "ranking.csv")

        # Save scenarios
        if self.scenarios is not None:
            with open(directory / "scenarios.json", "w") as f:
                json.dump(
                    {k: str(v) for k, v in self.scenarios.items()},
                    f,
                    indent=2,
                    default=str,
                )

        # Save CV results
        if self.cv_results is not None:
            with open(directory / "cv_results.json", "w") as f:
                json.dump(
                    {k: str(v) for k, v in self.cv_results.items()},
                    f,
                    indent=2,
                    default=str,
                )

        # Save metadata
        with open(directory / "metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2, default=str)

        logger.info("Results saved to %s", directory)

    @classmethod
    def load(cls, directory: str | Path) -> ExperimentResults:
        """Load results from a directory.

        Parameters
        ----------
        directory : str or Path
            Directory with saved results.

        Returns
        -------
        ExperimentResults
            Loaded results.
        """
        directory = Path(directory)

        # Load forecasts
        forecasts: dict[str, Forecast] = {}
        forecasts_dir = directory / "forecasts"
        if forecasts_dir.exists():
            for fc_path in forecasts_dir.glob("*.json"):
                fc = Forecast.load(fc_path)
                forecasts[fc.model_name or fc_path.stem] = fc

        # Load combination
        combination = None
        combo_path = directory / "combination.json"
        if combo_path.exists():
            combination = Forecast.load(combo_path)

        # Load evaluation
        evaluation = None
        eval_path = directory / "evaluation.csv"
        if eval_path.exists():
            evaluation = pd.read_csv(eval_path, index_col=0)

        # Load ranking
        ranking = None
        rank_path = directory / "ranking.csv"
        if rank_path.exists():
            ranking = pd.read_csv(rank_path, index_col=0)

        # Load metadata
        metadata: dict[str, Any] = {}
        meta_path = directory / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)

        return cls(
            forecasts=forecasts,
            combination=combination,
            evaluation=evaluation,
            ranking=ranking,
            metadata=metadata,
        )

    def report(self, output: str | Path, format: str | None = None) -> None:
        """Generate a full report.

        Parameters
        ----------
        output : str or Path
            Output file path.
        format : str or None
            Report format ('html', 'md', 'json'). Inferred from extension if None.
        """
        output = Path(output)
        if format is None:
            format = output.suffix.lstrip(".")
            if not format:
                format = "html"

        report_content = self._build_report(format)

        with open(output, "w") as f:
            f.write(report_content)

        logger.info("Report saved to %s", output)

    def _build_report(self, format: str) -> str:
        """Build report content in the specified format."""
        if format == "html":
            return self._build_html_report()
        elif format == "md":
            return self._build_md_report()
        elif format == "json":
            return self._build_json_report()
        else:
            msg = f"Unknown report format: {format}. Use 'html', 'md', or 'json'."
            raise ValueError(msg)

    def _build_html_report(self) -> str:
        """Build an HTML report."""
        lines = [
            "<!DOCTYPE html>",
            "<html><head><title>ForecastBox Experiment Report</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 40px; }",
            "table { border-collapse: collapse; width: 100%; margin: 20px 0; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "th { background-color: #4CAF50; color: white; }",
            "tr:nth-child(even) { background-color: #f2f2f2; }",
            "h1 { color: #333; } h2 { color: #555; }",
            "</style></head><body>",
            "<h1>ForecastBox Experiment Report</h1>",
        ]

        # Summary
        lines.append("<h2>Summary</h2>")
        lines.append(f"<p>Models: {len(self.forecasts)}</p>")
        lines.append(
            f"<p>Combination: {'Yes' if self.combination else 'No'}</p>"
        )

        # Forecasts
        lines.append("<h2>Forecasts</h2>")
        lines.append(
            "<table><tr><th>Model</th><th>Horizon</th>"
            "<th>Point (first 5)</th></tr>"
        )
        for name, fc in self.forecasts.items():
            point_str = ", ".join(f"{x:.2f}" for x in fc.point[:5])
            lines.append(
                f"<tr><td>{name}</td><td>{fc.horizon}</td>"
                f"<td>{point_str}</td></tr>"
            )
        lines.append("</table>")

        # Evaluation
        if self.evaluation is not None:
            lines.append("<h2>Evaluation</h2>")
            lines.append(self.evaluation.to_html())

        # Ranking
        if self.ranking is not None:
            lines.append("<h2>Model Ranking</h2>")
            lines.append(self.ranking.to_html())

        lines.append("</body></html>")
        return "\n".join(lines)

    def _build_md_report(self) -> str:
        """Build a Markdown report."""
        lines = ["# ForecastBox Experiment Report", ""]
        lines.append("## Summary")
        lines.append(f"- Models: {len(self.forecasts)}")
        lines.append(
            f"- Combination: {'Yes' if self.combination else 'No'}"
        )
        lines.append("")

        lines.append("## Forecasts")
        lines.append("| Model | Horizon | Point (first 5) |")
        lines.append("|-------|---------|------------------|")
        for name, fc in self.forecasts.items():
            point_str = ", ".join(f"{x:.2f}" for x in fc.point[:5])
            lines.append(f"| {name} | {fc.horizon} | {point_str} |")
        lines.append("")

        if self.evaluation is not None:
            lines.append("## Evaluation")
            lines.append(self.evaluation.to_markdown())
            lines.append("")

        if self.ranking is not None:
            lines.append("## Model Ranking")
            lines.append(self.ranking.to_markdown())
            lines.append("")

        return "\n".join(lines)

    def _build_json_report(self) -> str:
        """Build a JSON report."""
        report: dict[str, Any] = {
            "n_models": len(self.forecasts),
            "models": list(self.forecasts.keys()),
            "has_combination": self.combination is not None,
            "metadata": self.metadata,
        }
        if self.evaluation is not None:
            report["evaluation"] = self.evaluation.to_dict()
        if self.ranking is not None:
            report["ranking"] = self.ranking.to_dict()
        return json.dumps(report, indent=2, default=str)

    def summary(self) -> str:
        """Return a human-readable summary.

        Returns
        -------
        str
            Text summary of the experiment results.
        """
        lines = [
            "=" * 60,
            "ForecastBox Experiment Summary",
            "=" * 60,
            "",
            f"Models: {len(self.forecasts)}",
        ]

        for name, fc in self.forecasts.items():
            lines.append(f"  - {name} (horizon={fc.horizon})")

        if self.combination is not None:
            lines.append(f"\nCombination: {self.combination.model_name}")
            lines.append(f"  Point[0:3]: {self.combination.point[:3]}")

        if self.evaluation is not None:
            lines.append("\nEvaluation:")
            lines.append(str(self.evaluation))

        if self.ranking is not None:
            lines.append("\nRanking:")
            lines.append(str(self.ranking))

        if self.mcs is not None:
            lines.append(f"\nMCS: {self.mcs}")

        if self.cv_results is not None:
            lines.append(f"\nCross-Validation: {len(self.cv_results)} models")

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)


class ForecastExperiment:
    """High-level forecasting workflow.

    Combines auto-forecasting, combination, evaluation, scenarios,
    cross-validation, and reporting in a single unified API.

    Parameters
    ----------
    data : pd.DataFrame or pd.Series
        Input data.
    target : str
        Target variable name.
    models : list[str]
        Models to fit. Options: 'auto_arima', 'auto_ets', 'theta', 'var'.
    combination : str
        Combination method: 'mean', 'median', 'bma', 'inverse_mse', 'ols', etc.
    scenarios : dict[str, dict[str, float]] or None
        Conditional scenarios. Keys are scenario names, values are variable-value dicts.
    horizon : int
        Forecast horizon.
    cv_type : str
        Cross-validation type: 'expanding' or 'sliding'.
    cv_initial : int or None
        Initial window for CV. Defaults to half the data length.
    evaluation : list[str]
        Statistical tests to run: 'dm', 'mcs', 'gw', 'mz', 'encompassing'.
    report_format : str
        Report output format: 'html', 'md', 'json'.

    Example
    -------
    >>> exp = ForecastExperiment(
    ...     data=data, target='ipca',
    ...     models=['auto_arima', 'auto_ets', 'theta'],
    ...     combination='bma',
    ...     horizon=12,
    ... )
    >>> results = exp.run()
    >>> results.report('report.html')
    """

    def __init__(
        self,
        data: pd.DataFrame | pd.Series,
        target: str,
        models: list[str] | None = None,
        combination: str = "bma",
        scenarios: dict[str, dict[str, float]] | None = None,
        horizon: int = 12,
        cv_type: str = "expanding",
        cv_initial: int | None = None,
        evaluation: list[str] | None = None,
        report_format: str = "html",
    ) -> None:
        if isinstance(data, pd.Series):
            self.data = data.to_frame(name=target)
        else:
            self.data = data

        self.target = target
        self.models = models or ["auto_arima", "auto_ets"]
        self.combination = combination
        self.scenarios = scenarios
        self.horizon = horizon
        self.cv_type = cv_type
        self.cv_initial = cv_initial
        self.evaluation_tests = evaluation or ["dm", "mcs"]
        self.report_format = report_format

        self._results: ExperimentResults | None = None

    def run(self) -> ExperimentResults:
        """Execute the full experiment pipeline.

        Returns
        -------
        ExperimentResults
            Results containing forecasts, combination, evaluation, etc.
        """
        logger.info("Starting ForecastExperiment with models=%s", self.models)

        results = ExperimentResults(
            metadata={
                "target": self.target,
                "models": self.models,
                "combination_method": self.combination,
                "horizon": self.horizon,
                "cv_type": self.cv_type,
                "n_observations": len(self.data),
            }
        )

        # Step 1: Auto-forecast all models
        logger.info("Step 1: Auto-forecasting...")
        results.forecasts = self._auto_forecast()

        # Step 2: Combine forecasts
        if len(results.forecasts) > 1:
            logger.info("Step 2: Combining forecasts...")
            results.combination = self._combine(results.forecasts)

        # Step 3: Cross-validation
        logger.info("Step 3: Cross-validation...")
        results.cv_results = self._cross_validate()

        # Step 4: Evaluation
        if len(results.forecasts) > 1:
            logger.info("Step 4: Evaluating models...")
            eval_result = self._evaluate(results.forecasts, results.cv_results)
            results.evaluation = eval_result.get("metrics_df")
            results.ranking = eval_result.get("ranking")
            results.mcs = eval_result.get("mcs")

        # Step 5: Scenarios
        if self.scenarios:
            logger.info("Step 5: Running scenarios...")
            results.scenarios = self._scenarios()

        self._results = results
        logger.info("Experiment complete.")
        return results

    def _auto_forecast(self) -> dict[str, Forecast]:
        """Fit all models and produce forecasts.

        Returns
        -------
        dict[str, Forecast]
            Forecasts keyed by model name.
        """
        series = self.data[self.target].dropna()
        forecasts: dict[str, Forecast] = {}

        model_map: dict[str, type] = {}

        # Try to import model classes
        try:
            from forecastbox.auto.arima import AutoARIMA

            model_map["auto_arima"] = AutoARIMA
        except ImportError:
            logger.warning("AutoARIMA not available")

        try:
            from forecastbox.auto.ets import AutoETS

            model_map["auto_ets"] = AutoETS
        except ImportError:
            logger.warning("AutoETS not available")

        try:
            from forecastbox.auto._adapters import ThetaAdapter

            model_map["theta"] = ThetaAdapter
        except ImportError:
            logger.warning("Theta not available")

        try:
            from forecastbox.auto.var import AutoVAR

            model_map["var"] = AutoVAR
        except ImportError:
            logger.warning("AutoVAR not available")

        for model_name in self.models:
            if model_name not in model_map:
                logger.warning("Model '%s' not available, skipping", model_name)
                continue

            try:
                model_cls = model_map[model_name]
                estimator = model_cls()

                if model_name == "var":
                    fit_result = estimator.fit(self.data)
                else:
                    fit_result = estimator.fit(series)

                fc = fit_result.forecast(h=self.horizon)
                forecasts[fc.model_name or model_name] = fc
                logger.info("  %s: OK", model_name)
            except Exception as e:
                logger.warning("  %s: FAILED (%s)", model_name, e)

        return forecasts

    def _combine(self, forecasts: dict[str, Forecast]) -> Forecast:
        """Combine forecasts using the configured method.

        Parameters
        ----------
        forecasts : dict[str, Forecast]
            Individual forecasts.

        Returns
        -------
        Forecast
            Combined forecast.
        """
        fc_list = list(forecasts.values())

        if self.combination in ("mean", "median"):
            return Forecast.combine(fc_list, method=self.combination)

        try:
            from forecastbox.combination import SimpleCombiner

            combiner = SimpleCombiner(method=self.combination)
            return combiner.combine(fc_list)
        except (ImportError, Exception):
            logger.warning(
                "Combination method '%s' not available, falling back to mean",
                self.combination,
            )
            return Forecast.combine(fc_list, method="mean")

    def _evaluate(
        self,
        forecasts: dict[str, Forecast],
        cv_results: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Evaluate models with metrics and statistical tests.

        Parameters
        ----------
        forecasts : dict[str, Forecast]
            Individual forecasts.
        cv_results : dict[str, Any] or None
            Cross-validation results for computing error series.

        Returns
        -------
        dict[str, Any]
            Dictionary with 'metrics_df', 'ranking', and optionally 'mcs'.
        """
        result: dict[str, Any] = {}

        # Compute metrics from CV results if available
        if cv_results:
            metrics_data: dict[str, dict[str, float]] = {}
            for model_name, cv_res in cv_results.items():
                if isinstance(cv_res, dict) and "metrics" in cv_res:
                    metrics_data[model_name] = cv_res["metrics"]
                elif hasattr(cv_res, "summary"):
                    try:
                        summary = cv_res.summary()
                        if isinstance(summary, dict):
                            metrics_data[model_name] = summary
                    except Exception:
                        pass

            if metrics_data:
                result["metrics_df"] = pd.DataFrame(metrics_data).T
                # Ranking by RMSE if available
                if "rmse" in result["metrics_df"].columns:
                    result["ranking"] = result["metrics_df"].sort_values("rmse")
                else:
                    result["ranking"] = result["metrics_df"]

        # Run MCS if requested
        if "mcs" in self.evaluation_tests and cv_results:
            try:
                from forecastbox.evaluation.mcs import model_confidence_set

                # Build error series from CV results
                error_series: dict[str, np.ndarray] = {}
                for model_name, cv_res in cv_results.items():
                    if isinstance(cv_res, dict) and "errors" in cv_res:
                        error_series[model_name] = np.array(cv_res["errors"])
                    elif hasattr(cv_res, "errors"):
                        error_series[model_name] = np.array(cv_res.errors)

                if len(error_series) >= 2:
                    # model_confidence_set expects actual + forecasts dict
                    # We have errors, so we construct dummy actual/forecasts
                    min_len = min(len(e) for e in error_series.values())
                    actual = np.zeros(min_len)
                    fc_dict = {
                        name: errs[:min_len]
                        for name, errs in error_series.items()
                    }
                    result["mcs"] = model_confidence_set(
                        actual=actual,
                        forecasts=fc_dict,
                    )
            except ImportError:
                logger.warning("MCS module not available")
            except Exception as e:
                logger.warning("MCS test failed: %s", e)

        return result

    def _scenarios(self) -> dict[str, Any]:
        """Run conditional scenario analysis.

        Returns
        -------
        dict[str, Any]
            Scenario results.
        """
        if not self.scenarios:
            return {}

        scenario_results: dict[str, Any] = {}

        try:
            from forecastbox.auto.var import AutoVAR
            from forecastbox.scenarios.builder import ScenarioBuilder

            var = AutoVAR()
            var_result = var.fit(self.data)
            builder = ScenarioBuilder(var_result.model)

            for name, conditions in self.scenarios.items():
                # ScenarioBuilder expects dict[str, list[float]]
                list_conditions = {
                    k: [v] * self.horizon if isinstance(v, (int, float)) else list(v)
                    for k, v in conditions.items()
                }
                builder.add_scenario(name, list_conditions)

            results = builder.run(steps=self.horizon)
            for name in self.scenarios:
                try:
                    scenario_data = results.scenarios.get(name, {})
                    scenario_results[name] = {
                        var_name: {
                            "point": fc.point.tolist(),
                        }
                        for var_name, fc in scenario_data.items()
                    }
                except Exception:
                    scenario_results[name] = str(results)

        except ImportError:
            logger.warning("Scenario module not available")
        except Exception as e:
            logger.warning("Scenario analysis failed: %s", e)

        return scenario_results

    def _cross_validate(self) -> dict[str, Any]:
        """Run cross-validation for all models.

        Returns
        -------
        dict[str, Any]
            CV results per model.
        """
        series = self.data[self.target].dropna()
        initial = self.cv_initial or max(60, len(series) // 2)
        cv_results: dict[str, Any] = {}

        try:
            from forecastbox.cv import expanding_window_cv, rolling_window_cv

            cv_fn = (
                expanding_window_cv
                if self.cv_type == "expanding"
                else rolling_window_cv
            )
        except ImportError:
            logger.warning("CV module not available")
            return cv_results

        model_map: dict[str, type] = {}
        try:
            from forecastbox.auto.arima import AutoARIMA

            model_map["auto_arima"] = AutoARIMA
        except ImportError:
            pass
        try:
            from forecastbox.auto.ets import AutoETS

            model_map["auto_ets"] = AutoETS
        except ImportError:
            pass
        try:
            from forecastbox.auto._adapters import ThetaAdapter

            model_map["theta"] = ThetaAdapter
        except ImportError:
            pass

        for model_name in self.models:
            if model_name not in model_map:
                continue

            try:
                model_cls = model_map[model_name]
                h = self.horizon

                def model_fn(
                    s: pd.Series,
                    cls: type = model_cls,
                    horizon: int = h,
                ) -> Forecast:
                    est = cls()
                    fit_result = est.fit(s)
                    return fit_result.forecast(h=horizon)

                cv_kwargs: dict[str, Any] = {
                    "data": series,
                    "model_fn": model_fn,
                    "horizon": self.horizon,
                    "step": 1,
                }
                if self.cv_type == "expanding":
                    cv_kwargs["initial_window"] = initial
                else:
                    cv_kwargs["window"] = initial

                result = cv_fn(**cv_kwargs)
                cv_results[model_name] = result
                logger.info("  CV %s: OK", model_name)
            except Exception as e:
                logger.warning("  CV %s: FAILED (%s)", model_name, e)

        return cv_results

    def _generate_report(self, output: str | Path) -> None:
        """Generate experiment report.

        Parameters
        ----------
        output : str or Path
            Output file path.
        """
        if self._results is None:
            msg = "No results to report. Run the experiment first."
            raise RuntimeError(msg)

        self._results.report(output, format=self.report_format)
