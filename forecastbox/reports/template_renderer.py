"""Jinja2 template renderer for forecastbox reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

_TEMPLATES_DIR = Path(__file__).parent / "templates"

# Template mapping
_TEMPLATE_MAP: dict[str, str] = {
    "default_html": "default_html.jinja2",
    "default_latex": "default_latex.jinja2",
    "executive_html": "executive_html.jinja2",
    "technical_html": "technical_html.jinja2",
}


def _has_jinja2() -> bool:
    """Check if Jinja2 is available."""
    try:
        import jinja2  # noqa: F401

        return True
    except ImportError:
        return False


def render_template(
    template_name: str,
    sections: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> str | None:
    """Render a report using a Jinja2 template.

    Parameters
    ----------
    template_name : str
        Template name (e.g., 'default_html', 'executive_html').
    sections : list[dict[str, Any]]
        Section content.
    metadata : dict[str, Any]
        Report metadata.

    Returns
    -------
    str or None
        Rendered content, or None if Jinja2 is not available.
    """
    if not _has_jinja2():
        return None

    if template_name not in _TEMPLATE_MAP:
        return None

    import jinja2

    template_file = _TEMPLATE_MAP[template_name]
    template_path = _TEMPLATES_DIR / template_file

    if not template_path.exists():
        return None

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=jinja2.select_autoescape(["html"]),
    )

    template = env.get_template(template_file)
    rendered = template.render(
        sections=sections,
        metadata=metadata,
    )

    return rendered


def list_templates() -> list[str]:
    """List available template names.

    Returns
    -------
    list[str]
        Available template names.
    """
    return sorted(_TEMPLATE_MAP.keys())


def get_template_path(template_name: str) -> Path | None:
    """Get the file path for a template.

    Parameters
    ----------
    template_name : str
        Template name.

    Returns
    -------
    Path or None
        Template file path, or None if not found.
    """
    if template_name in _TEMPLATE_MAP:
        path = _TEMPLATES_DIR / _TEMPLATE_MAP[template_name]
        if path.exists():
            return path
    return None
