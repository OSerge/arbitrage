"""One formatter for legacy mapping views and typed financial results."""

from rich.console import Console
from rich.table import Table

from core.interfaces import IResultsFormatter


class RichResultsFormatter(IResultsFormatter):
    def __init__(self):
        self.console = Console()

    def format_single_result(self, results, pair_name=None):
        table = Table(
            title=f"Результаты бэктеста{f' {pair_name}' if pair_name else ''}"
        )
        table.add_column("Метрика")
        table.add_column("Значение")
        payload = results.to_dict() if hasattr(results, "to_dict") else results
        for section in ("performance", "costs", "risk_analysis"):
            for key, value in payload.get(section, {}).items():
                if isinstance(value, dict):
                    continue
                table.add_row(
                    key, "недостаточно данных" if value is None else str(value)
                )
        self.console.print(table)

    def format_summary_results(self, results_dict):
        for pair, result in results_dict.items():
            self.format_single_result(result, str(pair))


class HorizontalResultsFormatter(RichResultsFormatter):
    """Retained import name using the common result formatting implementation."""
