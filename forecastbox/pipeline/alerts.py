"""AlertSystem - Automated forecast degradation alerts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from forecastbox.pipeline.monitor import ForecastMonitor


@dataclass
class AlertRule:
    """Definition of an alert rule.

    Attributes
    ----------
    name : str
        Name of the rule.
    metric : str
        Metric to monitor: 'rmse', 'mae', 'bias', 'hit_rate'.
    condition : str
        Condition type: 'above', 'below', 'change'.
    threshold : float
        Threshold value.
    window : int
        Evaluation window in periods.
    severity : str
        Alert severity: 'info', 'warning', 'critical'.
    """

    name: str
    metric: str
    condition: str
    threshold: float
    window: int
    severity: str = "warning"


@dataclass
class Alert:
    """A triggered alert instance.

    Attributes
    ----------
    rule : str
        Name of the rule that triggered.
    timestamp : datetime
        When the alert was triggered.
    metric_value : float
        Current value of the metric.
    threshold : float
        Threshold that was exceeded.
    severity : str
        Alert severity.
    message : str
        Human-readable alert message.
    """

    rule: str
    timestamp: datetime
    metric_value: float
    threshold: float
    severity: str
    message: str


# Preset configurations
_PRESETS: dict[str, dict[str, Any]] = {
    "rmse_spike": {
        "metric": "rmse",
        "condition": "above",
        "threshold": 1.5,
        "window": 6,
        "severity": "warning",
    },
    "bias_drift": {
        "metric": "bias",
        "condition": "above",
        "threshold": 0.5,
        "window": 12,
        "severity": "warning",
    },
    "coverage_drop": {
        "metric": "hit_rate",
        "condition": "below",
        "threshold": 0.8,
        "window": 12,
        "severity": "critical",
    },
    "model_change": {
        "metric": "rmse",
        "condition": "change",
        "threshold": 0.3,
        "window": 6,
        "severity": "info",
    },
}


class AlertSystem:
    """System for monitoring forecast quality and triggering alerts.

    Parameters
    ----------
    monitor : ForecastMonitor
        Monitor to evaluate rules against.
    """

    def __init__(self, monitor: ForecastMonitor) -> None:
        self.monitor = monitor
        self.rules: list[AlertRule] = []
        self._triggered: list[Alert] = []

    def add_rule(
        self,
        name: str,
        metric: str = "rmse",
        condition: str = "above",
        threshold: float = 1.5,
        window: int = 6,
        severity: str = "warning",
    ) -> None:
        """Add an alert rule.

        Parameters
        ----------
        name : str
            Rule name.
        metric : str
            Metric to monitor: 'rmse', 'mae', 'bias', 'hit_rate'.
        condition : str
            Condition: 'above', 'below', 'change'.
        threshold : float
            Threshold value.
        window : int
            Evaluation window.
        severity : str
            Severity: 'info', 'warning', 'critical'.
        """
        rule = AlertRule(
            name=name,
            metric=metric,
            condition=condition,
            threshold=threshold,
            window=window,
            severity=severity,
        )
        self.rules.append(rule)

    def add_preset(self, preset_name: str) -> None:
        """Add a preset alert rule.

        Parameters
        ----------
        preset_name : str
            Preset name: 'rmse_spike', 'bias_drift', 'coverage_drop', 'model_change'.
        """
        if preset_name not in _PRESETS:
            available = ", ".join(sorted(_PRESETS.keys()))
            msg = f"Unknown preset '{preset_name}'. Available: {available}"
            raise ValueError(msg)

        config = _PRESETS[preset_name]
        self.add_rule(name=preset_name, **config)

    def remove_rule(self, name: str) -> None:
        """Remove a rule by name.

        Parameters
        ----------
        name : str
            Rule name to remove.
        """
        self.rules = [r for r in self.rules if r.name != name]

    def _evaluate_rule(self, rule: AlertRule) -> Alert | None:
        """Evaluate a single rule against current monitor state."""
        pairs = self.monitor._get_matched_pairs()
        if pairs.empty or len(pairs) < rule.window:
            return None

        errors = pairs["actual"].values - pairs["forecast"].values
        recent_errors = errors[-rule.window :]
        historical_errors = errors[: -rule.window] if len(errors) > rule.window else errors

        current_value: float = 0.0
        triggered = False

        if rule.metric == "rmse":
            recent_rmse = float(np.sqrt(np.mean(recent_errors**2)))
            historical_rmse = float(np.sqrt(np.mean(historical_errors**2)))
            current_value = recent_rmse

            if rule.condition == "above":
                if historical_rmse > 1e-10:
                    triggered = recent_rmse > rule.threshold * historical_rmse
                else:
                    triggered = recent_rmse > rule.threshold
            elif rule.condition == "change" and historical_rmse > 1e-10:
                change = abs(recent_rmse - historical_rmse) / historical_rmse
                triggered = change > rule.threshold
                current_value = change

        elif rule.metric == "mae":
            recent_mae = float(np.mean(np.abs(recent_errors)))
            historical_mae = float(np.mean(np.abs(historical_errors)))
            current_value = recent_mae

            if rule.condition == "above":
                if historical_mae > 1e-10:
                    triggered = recent_mae > rule.threshold * historical_mae
                else:
                    triggered = recent_mae > rule.threshold

        elif rule.metric == "bias":
            recent_bias = float(np.abs(np.mean(recent_errors)))
            current_value = recent_bias

            if rule.condition == "above":
                triggered = recent_bias > rule.threshold

        elif rule.metric == "hit_rate":
            if "lower_95" in pairs.columns and "upper_95" in pairs.columns:
                valid = pairs.dropna(subset=["lower_95", "upper_95"]).tail(rule.window)
                if len(valid) > 0:
                    inside = (valid["actual"] >= valid["lower_95"]) & (
                        valid["actual"] <= valid["upper_95"]
                    )
                    current_value = float(inside.mean())
                    if rule.condition == "below":
                        triggered = current_value < rule.threshold

        if triggered:
            return Alert(
                rule=rule.name,
                timestamp=datetime.now(),
                metric_value=current_value,
                threshold=rule.threshold,
                severity=rule.severity,
                message=(
                    f"Alert '{rule.name}': {rule.metric}={current_value:.4f} "
                    f"{rule.condition} threshold={rule.threshold}"
                ),
            )

        return None

    def check(self) -> list[Alert]:
        """Check all rules and return triggered alerts.

        Returns
        -------
        list[Alert]
            List of triggered alerts.
        """
        new_alerts: list[Alert] = []
        for rule in self.rules:
            alert = self._evaluate_rule(rule)
            if alert is not None:
                new_alerts.append(alert)
                self._triggered.append(alert)
        return new_alerts

    def history(self) -> list[Alert]:
        """Return history of all triggered alerts.

        Returns
        -------
        list[Alert]
            All alerts triggered since creation.
        """
        return list(self._triggered)

    def summary(self) -> str:
        """Generate summary of alert system state.

        Returns
        -------
        str
            Formatted summary.
        """
        lines: list[str] = []
        lines.append("=" * 50)
        lines.append("ALERT SYSTEM SUMMARY")
        lines.append("=" * 50)
        lines.append(f"\nActive rules: {len(self.rules)}")
        for rule in self.rules:
            lines.append(
                f"  - {rule.name}: {rule.metric} {rule.condition} "
                f"{rule.threshold} (window={rule.window}, severity={rule.severity})"
            )
        lines.append(f"\nTotal alerts triggered: {len(self._triggered)}")
        if self._triggered:
            lines.append("\nRecent alerts:")
            for alert in self._triggered[-5:]:
                lines.append(f"  [{alert.severity.upper()}] {alert.message}")
        lines.append("\n" + "=" * 50)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"AlertSystem(rules={len(self.rules)}, "
            f"triggered={len(self._triggered)})"
        )
