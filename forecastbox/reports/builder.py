"""ReportBuilder - Compose and render forecast reports."""

from __future__ import annotations

import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any

from forecastbox.pipeline.pipeline import PipelineResults
from forecastbox.reports.sections import SECTION_REGISTRY
from forecastbox.reports.transformers.html import HTMLTransformer
from forecastbox.reports.transformers.json_transformer import JSONTransformer
from forecastbox.reports.transformers.latex import LaTeXTransformer
from forecastbox.reports.transformers.markdown import MarkdownTransformer
from forecastbox.reports.transformers.pdf import PDFTransformer

_TRANSFORMERS: dict[str, type] = {
    "html": HTMLTransformer,
    "latex": LaTeXTransformer,
    "markdown": MarkdownTransformer,
    "md": MarkdownTransformer,
    "json": JSONTransformer,
    "pdf": PDFTransformer,
}


class ReportBuilder:
    """Build and render forecast reports.

    Parameters
    ----------
    results : PipelineResults
        Pipeline results to report on.
    title : str
        Report title.
    author : str
        Report author.
    template : str
        Template preset: 'default', 'executive', 'technical'.
    """

    def __init__(
        self,
        results: PipelineResults,
        title: str = "Forecast Report",
        author: str = "",
        template: str = "default",
    ) -> None:
        self.results = results
        self.title = title
        self.author = author
        self.template = template
        self._sections: list[dict[str, Any]] = []
        self._section_order: list[str] = []

    def add_section(self, section_type: str, title: str | None = None, **kwargs: Any) -> None:
        """Add a section to the report.

        Parameters
        ----------
        section_type : str
            Section type: 'summary', 'data', 'models', 'forecasts',
            'evaluation', 'combination', 'scenarios', 'nowcast',
            'diagnostics', 'appendix'.
        title : str or None
            Custom section title.
        **kwargs
            Additional section parameters.
        """
        if section_type not in SECTION_REGISTRY:
            available = ", ".join(sorted(SECTION_REGISTRY.keys()))
            msg = f"Unknown section type '{section_type}'. Available: {available}"
            raise ValueError(msg)

        if title:
            kwargs["title"] = title

        generator = SECTION_REGISTRY[section_type]
        section_content = generator(self.results, **kwargs)
        self._sections.append(section_content)
        self._section_order.append(section_type)

    def remove_section(self, section_type: str) -> None:
        """Remove a section by type.

        Parameters
        ----------
        section_type : str
            Section type to remove.
        """
        indices_to_remove = [
            i for i, s in enumerate(self._sections)
            if s.get("type") == section_type
        ]
        for i in reversed(indices_to_remove):
            self._sections.pop(i)
            self._section_order.pop(i)

    def reorder_sections(self, order: list[str]) -> None:
        """Reorder sections by type.

        Parameters
        ----------
        order : list[str]
            Desired order of section types.
        """
        # Build mapping
        section_map: dict[str, list[dict[str, Any]]] = {}
        for section in self._sections:
            stype = section.get("type", "unknown")
            if stype not in section_map:
                section_map[stype] = []
            section_map[stype].append(section)

        # Rebuild in order
        new_sections: list[dict[str, Any]] = []
        new_order: list[str] = []
        for stype in order:
            if stype in section_map:
                for s in section_map[stype]:
                    new_sections.append(s)
                    new_order.append(stype)

        # Append any remaining sections not in order
        ordered_types = set(order)
        for section in self._sections:
            stype = section.get("type", "unknown")
            if stype not in ordered_types:
                new_sections.append(section)
                new_order.append(stype)

        self._sections = new_sections
        self._section_order = new_order

    def render(self, format: str = "html", output: str | None = None) -> str:
        """Render the report in the specified format.

        Parameters
        ----------
        format : str
            Output format: 'html', 'latex', 'markdown', 'json', 'pdf'.
        output : str or None
            File path to save. If None, returns string.

        Returns
        -------
        str
            Report content (or file path for PDF).
        """
        if format not in _TRANSFORMERS:
            available = ", ".join(sorted(_TRANSFORMERS.keys()))
            msg = f"Unknown format '{format}'. Available: {available}"
            raise ValueError(msg)

        metadata: dict[str, Any] = {
            "title": self.title,
            "author": self.author,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "template": self.template,
        }

        transformer_cls = _TRANSFORMERS[format]
        transformer = transformer_cls(template=self.template)

        if format == "pdf":
            content = transformer.render(self._sections, metadata, output=output)
        else:
            content = transformer.render(self._sections, metadata)

        if output and format != "pdf":
            path = Path(output)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

        return content

    def preview(self) -> None:
        """Preview the HTML report in the default browser."""
        import tempfile

        content = self.render(format="html")
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
            f.write(content)
            webbrowser.open(f"file://{f.name}")

    def __repr__(self) -> str:
        return (
            f"ReportBuilder(title='{self.title}', "
            f"sections={len(self._sections)}, template='{self.template}')"
        )
