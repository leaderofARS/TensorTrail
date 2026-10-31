"""Report serialization."""

from .html_report import report_to_html, save_html_report
from .json_report import report_to_dict, report_to_json, save_json_report

__all__ = [
    "report_to_dict",
    "report_to_html",
    "report_to_json",
    "save_html_report",
    "save_json_report",
]
