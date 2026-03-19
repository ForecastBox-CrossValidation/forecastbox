"""HTML report transformer."""

from __future__ import annotations

from typing import Any


class HTMLTransformer:
    """Transform report sections into HTML format.

    Parameters
    ----------
    template : str
        Template name: 'default', 'executive', 'technical'.
    """

    def __init__(self, template: str = "default") -> None:
        self.template = template

    def render(
        self,
        sections: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> str:
        """Render sections to HTML string.

        Parameters
        ----------
        sections : list[dict[str, Any]]
            List of section content dictionaries.
        metadata : dict[str, Any]
            Report metadata (title, author, date).

        Returns
        -------
        str
            Complete HTML document.
        """
        title = metadata.get("title", "Forecast Report")
        author = metadata.get("author", "")
        date = metadata.get("date", "")

        html_parts: list[str] = []
        html_parts.append("<!DOCTYPE html>")
        html_parts.append("<html lang='en'>")
        html_parts.append("<head>")
        html_parts.append(f"<title>{title}</title>")
        html_parts.append("<meta charset='utf-8'>")
        html_parts.append("<style>")
        html_parts.append(self._get_css())
        html_parts.append("</style>")
        html_parts.append("</head>")
        html_parts.append("<body>")
        html_parts.append(f"<h1>{title}</h1>")
        if author:
            html_parts.append(f"<p class='author'>Author: {author}</p>")
        if date:
            html_parts.append(f"<p class='date'>Date: {date}</p>")

        for section in sections:
            html_parts.append(self._render_section(section))

        html_parts.append("</body>")
        html_parts.append("</html>")

        return "\n".join(html_parts)

    def _render_section(self, section: dict[str, Any]) -> str:
        """Render a single section to HTML."""
        parts: list[str] = []
        parts.append(f"<section class='section-{section.get('type', 'unknown')}'>")
        parts.append(f"<h2>{section.get('title', 'Section')}</h2>")

        text = section.get("text", "")
        if text:
            # Convert markdown bold to HTML
            text = text.replace("**", "<strong>", 1).replace("**", "</strong>", 1)
            parts.append(f"<p>{text}</p>")

        # Key metrics
        if "key_metrics" in section and section["key_metrics"]:
            parts.append("<table class='metrics-table'>")
            parts.append("<tr><th>Metric</th><th>Value</th></tr>")
            for name, value in section["key_metrics"].items():
                parts.append(f"<tr><td>{name}</td><td>{value:.4f}</td></tr>")
            parts.append("</table>")

        # Evaluation table
        if "evaluation" in section and section["evaluation"]:
            parts.append(self._render_dict_table(section["evaluation"]))

        # Models list
        if "models" in section and isinstance(section["models"], list):
            parts.append("<ul>")
            for model in section["models"]:
                if isinstance(model, dict):
                    name = model.get("name", "Unknown")
                    horizon = model.get('horizon', '?')
                    parts.append(
                        f"<li><strong>{name}</strong>"
                        f" (horizon={horizon})</li>"
                    )
            parts.append("</ul>")

        # Weights
        if "weights" in section and section["weights"]:
            parts.append("<table class='weights-table'>")
            parts.append("<tr><th>Model</th><th>Weight</th></tr>")
            for name, weight in section["weights"].items():
                parts.append(f"<tr><td>{name}</td><td>{weight:.4f}</td></tr>")
            parts.append("</table>")

        # Inline plot
        if "plot_base64" in section and section["plot_base64"]:
            parts.append(
                f"<img src='data:image/png;base64,{section['plot_base64']}' "
                f"alt='Plot' class='plot-image' />"
            )

        # Diagnostics
        if "diagnostics" in section and section["diagnostics"]:
            for model_name, diag in section["diagnostics"].items():
                parts.append(f"<h3>{model_name}</h3>")
                if isinstance(diag, dict):
                    parts.append("<table>")
                    for k, v in diag.items():
                        parts.append(f"<tr><td>{k}</td><td>{v:.4f}</td></tr>")
                    parts.append("</table>")

        parts.append("</section>")
        return "\n".join(parts)

    def _render_dict_table(self, data: dict[str, Any]) -> str:
        """Render a dict-of-dicts as HTML table."""
        if not data:
            return ""

        parts: list[str] = ["<table class='data-table'>"]

        # Get columns and rows
        first_key = next(iter(data.keys()))
        if isinstance(data[first_key], dict):
            rows = sorted(set().union(*(d.keys() for d in data.values() if isinstance(d, dict))))
            parts.append("<tr><th>Model</th>")
            for col in data:
                parts.append(f"<th>{col}</th>")
            parts.append("</tr>")

            for row in rows:
                parts.append(f"<tr><td>{row}</td>")
                for col in data:
                    val = data[col].get(row, "N/A") if isinstance(data[col], dict) else "N/A"
                    if isinstance(val, float):
                        parts.append(f"<td>{val:.4f}</td>")
                    else:
                        parts.append(f"<td>{val}</td>")
                parts.append("</tr>")
        else:
            parts.append("<tr>")
            for k, _v in data.items():
                parts.append(f"<th>{k}</th>")
            parts.append("</tr><tr>")
            for v in data.values():
                if isinstance(v, float):
                    parts.append(f"<td>{v:.4f}</td>")
                else:
                    parts.append(f"<td>{v}</td>")
            parts.append("</tr>")

        parts.append("</table>")
        return "\n".join(parts)

    def _get_css(self) -> str:
        """Return CSS styles for the report."""
        return """
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1000px;
            margin: 0 auto;
            padding: 40px;
            background: #fafafa;
            color: #2d2d2d;
            line-height: 1.6;
        }
        h1 { color: #1B3A5C; border-bottom: 3px solid #2E86AB; padding-bottom: 10px; }
        h2 { color: #1B3A5C; margin-top: 30px; border-bottom: 1px solid #ddd; padding-bottom: 5px; }
        h3 { color: #2E86AB; }
        .author, .date { color: #666; font-size: 0.9em; }
        table { border-collapse: collapse; width: 100%; margin: 15px 0; }
        th, td { padding: 10px 15px; text-align: left; border: 1px solid #ddd; }
        th { background-color: #1B3A5C; color: white; }
        tr:nth-child(even) { background-color: #f5f5f5; }
        .plot-image { max-width: 100%; height: auto; margin: 15px 0; border: 1px solid #ddd; }
        section { margin-bottom: 30px; }
        ul { padding-left: 20px; }
        li { margin-bottom: 5px; }
        """
