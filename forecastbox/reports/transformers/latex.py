"""LaTeX report transformer."""

from __future__ import annotations

from typing import Any


class LaTeXTransformer:
    """Transform report sections into LaTeX format."""

    def __init__(self, template: str = "default") -> None:
        self.template = template

    def render(
        self,
        sections: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> str:
        """Render sections to LaTeX string.

        Parameters
        ----------
        sections : list[dict[str, Any]]
            List of section content dictionaries.
        metadata : dict[str, Any]
            Report metadata.

        Returns
        -------
        str
            Complete LaTeX document.
        """
        title = metadata.get("title", "Forecast Report")
        author = metadata.get("author", "")
        date = metadata.get("date", r"\today")

        parts: list[str] = []
        parts.append(r"\documentclass[11pt,a4paper]{article}")
        parts.append(r"\usepackage[utf8]{inputenc}")
        parts.append(r"\usepackage[T1]{fontenc}")
        parts.append(r"\usepackage{booktabs}")
        parts.append(r"\usepackage{graphicx}")
        parts.append(r"\usepackage{hyperref}")
        parts.append(r"\usepackage{geometry}")
        parts.append(r"\geometry{margin=2.5cm}")
        parts.append("")
        parts.append(f"\\title{{{self._escape(title)}}}")
        parts.append(f"\\author{{{self._escape(author)}}}")
        parts.append(f"\\date{{{date}}}")
        parts.append("")
        parts.append(r"\begin{document}")
        parts.append(r"\maketitle")
        parts.append(r"\tableofcontents")
        parts.append(r"\newpage")
        parts.append("")

        for section in sections:
            parts.append(self._render_section(section))
            parts.append("")

        parts.append(r"\end{document}")

        return "\n".join(parts)

    def _render_section(self, section: dict[str, Any]) -> str:
        """Render a single section to LaTeX."""
        parts: list[str] = []
        title = self._escape(section.get("title", "Section"))
        parts.append(f"\\section{{{title}}}")

        text = section.get("text", "")
        if text:
            text = text.replace("**", r"\textbf{", 1).replace("**", "}", 1)
            parts.append(self._escape_text(text))
            parts.append("")

        # Key metrics
        if "key_metrics" in section and section["key_metrics"]:
            parts.append(r"\begin{table}[h]")
            parts.append(r"\centering")
            parts.append(r"\begin{tabular}{lr}")
            parts.append(r"\toprule")
            parts.append(r"Metric & Value \\")
            parts.append(r"\midrule")
            for name, value in section["key_metrics"].items():
                parts.append(f"{self._escape(name)} & {value:.4f} \\\\")
            parts.append(r"\bottomrule")
            parts.append(r"\end{tabular}")
            parts.append(r"\end{table}")
            parts.append("")

        # Models list
        if "models" in section and isinstance(section["models"], list):
            parts.append(r"\begin{itemize}")
            for model in section["models"]:
                if isinstance(model, dict):
                    name = model.get("name", "Unknown")
                    parts.append(f"  \\item \\textbf{{{self._escape(name)}}}")
            parts.append(r"\end{itemize}")
            parts.append("")

        return "\n".join(parts)

    @staticmethod
    def _escape(text: str) -> str:
        """Escape special LaTeX characters."""
        replacements = {
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    @staticmethod
    def _escape_text(text: str) -> str:
        """Escape text but preserve LaTeX commands."""
        # Don't escape backslashes or braces for LaTeX commands
        replacements = {
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text
