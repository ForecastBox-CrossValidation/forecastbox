"""Markdown report transformer."""

from __future__ import annotations

from typing import Any


class MarkdownTransformer:
    """Transform report sections into Markdown format."""

    def __init__(self, template: str = "default") -> None:
        self.template = template

    def render(
        self,
        sections: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> str:
        """Render sections to Markdown string.

        Parameters
        ----------
        sections : list[dict[str, Any]]
            List of section content dictionaries.
        metadata : dict[str, Any]
            Report metadata.

        Returns
        -------
        str
            Complete Markdown document.
        """
        title = metadata.get("title", "Forecast Report")
        author = metadata.get("author", "")
        date = metadata.get("date", "")

        parts: list[str] = []
        parts.append(f"# {title}")
        parts.append("")
        if author:
            parts.append(f"**Author:** {author}")
        if date:
            parts.append(f"**Date:** {date}")
        if author or date:
            parts.append("")
        parts.append("---")
        parts.append("")

        for section in sections:
            parts.append(self._render_section(section))
            parts.append("")

        return "\n".join(parts)

    def _render_section(self, section: dict[str, Any]) -> str:
        """Render a single section to Markdown."""
        parts: list[str] = []
        title = section.get("title", "Section")
        parts.append(f"## {title}")
        parts.append("")

        text = section.get("text", "")
        if text:
            parts.append(text)
            parts.append("")

        # Key metrics
        if "key_metrics" in section and section["key_metrics"]:
            parts.append("| Metric | Value |")
            parts.append("|:---|---:|")
            for name, value in section["key_metrics"].items():
                parts.append(f"| {name} | {value:.4f} |")
            parts.append("")

        # Evaluation table
        if "evaluation" in section and section["evaluation"]:
            parts.append(self._render_dict_table(section["evaluation"]))

        # Models list
        if "models" in section and isinstance(section["models"], list):
            for model in section["models"]:
                if isinstance(model, dict):
                    name = model.get("name", "Unknown")
                    parts.append(f"- **{name}** (horizon={model.get('horizon', '?')})")
            parts.append("")

        # Weights
        if "weights" in section and section["weights"]:
            parts.append("| Model | Weight |")
            parts.append("|:---|---:|")
            for name, weight in section["weights"].items():
                parts.append(f"| {name} | {weight:.4f} |")
            parts.append("")

        parts.append("---")
        return "\n".join(parts)

    def _render_dict_table(self, data: dict[str, Any]) -> str:
        """Render dict-of-dicts as Markdown table."""
        if not data:
            return ""

        parts: list[str] = []
        first_key = next(iter(data.keys()))

        if isinstance(data[first_key], dict):
            cols = list(data.keys())
            rows = sorted(set().union(*(d.keys() for d in data.values() if isinstance(d, dict))))

            header = "| Model | " + " | ".join(cols) + " |"
            separator = "|:---|" + "|".join(["---:" for _ in cols]) + "|"
            parts.append(header)
            parts.append(separator)

            for row in rows:
                vals: list[str] = []
                for col in cols:
                    val = data[col].get(row, "N/A") if isinstance(data[col], dict) else "N/A"
                    if isinstance(val, float):
                        vals.append(f"{val:.4f}")
                    else:
                        vals.append(str(val))
                parts.append(f"| {row} | " + " | ".join(vals) + " |")
        else:
            header = "| " + " | ".join(str(k) for k in data) + " |"
            separator = "|" + "|".join(["---:" for _ in data]) + "|"
            vals_line = "| " + " | ".join(
                f"{v:.4f}" if isinstance(v, float) else str(v) for v in data.values()
            ) + " |"
            parts.extend([header, separator, vals_line])

        parts.append("")
        return "\n".join(parts)
