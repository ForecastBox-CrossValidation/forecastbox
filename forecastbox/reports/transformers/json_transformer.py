"""JSON report transformer."""

from __future__ import annotations

import json
from typing import Any, cast

import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy types."""

    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)  # type: ignore[reportUnknownArgumentType]
        elif isinstance(o, np.floating):
            return float(o)  # type: ignore[reportUnknownArgumentType]
        elif isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


class JSONTransformer:
    """Transform report sections into JSON format."""

    def __init__(self, template: str = "default") -> None:
        self.template = template

    def render(
        self,
        sections: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> str:
        """Render sections to JSON string.

        Parameters
        ----------
        sections : list[dict[str, Any]]
            List of section content dictionaries.
        metadata : dict[str, Any]
            Report metadata.

        Returns
        -------
        str
            JSON string.
        """
        # Remove non-serializable fields
        clean_sections: list[dict[str, Any]] = []
        for section in sections:
            clean = self._clean_section(section)
            clean_sections.append(clean)

        report: dict[str, Any] = {
            "metadata": metadata,
            "sections": clean_sections,
        }

        return json.dumps(report, indent=2, cls=_NumpyEncoder)

    def _clean_section(self, section: dict[str, Any]) -> dict[str, Any]:
        """Remove non-serializable fields from section."""
        clean: dict[str, Any] = {}
        for key, value in section.items():
            if key == "plot_base64":
                # Keep but truncate for readability in JSON
                clean[key] = value[:100] + "..." if value and len(value) > 100 else value
            elif isinstance(value, dict):
                clean[key] = self._clean_dict(cast("dict[str, Any]", value))
            elif isinstance(value, list):
                value_list = cast("list[Any]", value)
                clean[key] = [
                    self._clean_dict(cast("dict[str, Any]", item))
                    if isinstance(item, dict)
                    else item
                    for item in value_list
                ]
            elif isinstance(value, (str, int, float, bool, type(None))):
                clean[key] = value
            elif isinstance(value, np.ndarray):
                clean[key] = value.tolist()
            else:
                clean[key] = str(value)
        return clean

    def _clean_dict(self, d: dict[Any, Any]) -> dict[str, Any]:
        """Recursively clean dict for JSON serialization."""
        clean: dict[Any, Any] = {}
        for key, value in d.items():
            # Ensure key is JSON-serializable
            str_key = key if isinstance(key, (str, int, float, bool)) else str(key)
            if isinstance(value, dict):
                clean[str_key] = self._clean_dict(cast("dict[str, Any]", value))
            elif isinstance(value, np.ndarray):
                clean[str_key] = value.tolist()
            elif isinstance(value, (str, int, float, bool, type(None))):
                clean[str_key] = value
            else:
                try:
                    json.dumps(value)
                    clean[str_key] = value
                except (TypeError, ValueError):
                    clean[str_key] = str(value)
        return clean
