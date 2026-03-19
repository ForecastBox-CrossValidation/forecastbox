"""Report generation module for forecastbox."""

from forecastbox.reports.builder import ReportBuilder
from forecastbox.reports.sections import SECTION_REGISTRY
from forecastbox.reports.template_renderer import list_templates, render_template

__all__ = [
    "ReportBuilder",
    "SECTION_REGISTRY",
    "list_templates",
    "render_template",
]
