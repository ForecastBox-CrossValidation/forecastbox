"""Tests for dataset loading."""

from __future__ import annotations

import pandas as pd
import pytest

from forecastbox.datasets import describe_dataset, list_datasets, load_dataset

ALL_DATASETS = [
    "macro_brazil",
    "macro_us",
    "macro_europe",
    "airline",
    "sunspot",
    "m3_monthly",
    "m3_quarterly",
    "m4_monthly",
    "m4_quarterly",
    "tourism",
    "electricity",
    "gas",
    "retail",
    "exchange_rates",
    "interest_rates",
    "commodity_prices",
    "us_gdp_vintages",
    "brazil_gdp_vintages",
    "simulated_var",
    "simulated_dfm",
]


class TestDatasets:
    """Tests for dataset loading."""

    def test_list_datasets(self) -> None:
        """list_datasets() returns all ~20 datasets."""
        names = list_datasets()
        assert len(names) >= 18
        assert "macro_brazil" in names
        assert "macro_us" in names
        assert "airline" in names
        assert "simulated_var" in names

    @pytest.mark.parametrize("name", ALL_DATASETS)
    def test_load_dataset(self, name: str) -> None:
        """Each dataset loads without error and returns dict of Series."""
        data = load_dataset(name)
        assert isinstance(data, dict)
        assert len(data) > 0
        for key, series in data.items():
            assert isinstance(series, pd.Series), f"{name}/{key} is not a Series"
            assert len(series) > 0, f"{name}/{key} is empty"
            assert isinstance(series.index, pd.DatetimeIndex), f"{name}/{key} has no DatetimeIndex"

    @pytest.mark.parametrize("name", ALL_DATASETS)
    def test_describe_dataset(self, name: str) -> None:
        """Each dataset has a description."""
        desc = describe_dataset(name)
        assert isinstance(desc, str)
        assert len(desc) > 10

    def test_unknown_dataset_raises(self) -> None:
        """Loading unknown dataset raises ValueError."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_dataset("nonexistent_dataset")

    def test_macro_brazil_columns(self) -> None:
        """macro_brazil has expected columns."""
        data = load_dataset("macro_brazil")
        expected = {"ipca", "selic", "cambio", "pib_mensal", "producao_industrial"}
        assert expected.issubset(set(data.keys()))

    def test_airline_length(self) -> None:
        """airline has 144 observations."""
        data = load_dataset("airline")
        assert len(data["passengers"]) == 144

    def test_simulated_var_length(self) -> None:
        """simulated_var has 500 observations."""
        data = load_dataset("simulated_var")
        assert len(data["var_1"]) == 500

    def test_simulated_dfm_series_count(self) -> None:
        """simulated_dfm has 10 series."""
        data = load_dataset("simulated_dfm")
        assert len(data) == 10
