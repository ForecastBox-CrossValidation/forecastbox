"""PDF report transformer (via LaTeX compilation)."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from forecastbox.reports.transformers.latex import LaTeXTransformer


class PDFTransformer:
    """Transform report sections into PDF via LaTeX compilation.

    Requires pdflatex to be installed on the system.
    """

    def __init__(self, template: str = "default") -> None:
        self.template = template
        self._latex = LaTeXTransformer(template=template)

    def render(
        self,
        sections: list[dict[str, Any]],
        metadata: dict[str, Any],
        output: str | None = None,
    ) -> str:
        """Render sections to PDF.

        Parameters
        ----------
        sections : list[dict[str, Any]]
            List of section content dictionaries.
        metadata : dict[str, Any]
            Report metadata.
        output : str or None
            Output PDF file path. If None, creates in temp directory.

        Returns
        -------
        str
            Path to the generated PDF file.
        """
        latex_content = self._latex.render(sections, metadata)

        # Check if pdflatex is available
        if not shutil.which("pdflatex"):
            # Fall back to just returning LaTeX content with .tex extension note
            if output:
                tex_path = output.replace(".pdf", ".tex")
                with open(tex_path, "w") as f:
                    f.write(latex_content)
                return tex_path
            return latex_content

        # Compile LaTeX to PDF
        with tempfile.TemporaryDirectory() as tmpdir:
            tex_path = Path(tmpdir) / "report.tex"
            with open(tex_path, "w") as f:
                f.write(latex_content)

            # Run pdflatex twice (for TOC)
            for _ in range(2):
                subprocess.run(
                    [
                        "pdflatex", "-interaction=nonstopmode",
                        "-output-directory", tmpdir, str(tex_path),
                    ],
                    capture_output=True,
                    timeout=60,
                    check=False,
                )

            pdf_path = Path(tmpdir) / "report.pdf"
            if pdf_path.exists():
                if output:
                    shutil.copy2(pdf_path, output)
                    return output
                else:
                    final_path = str(Path(tmpdir) / "report.pdf")
                    return final_path

        return latex_content

    def is_available(self) -> bool:
        """Check if pdflatex is available on the system.

        Returns
        -------
        bool
            True if pdflatex is found.
        """
        return shutil.which("pdflatex") is not None
