"""
Форматирование результатов бэктестинга
"""

from typing import Dict, Optional
from rich.console import Console, Group
from rich.table import Table
from rich.panel import Panel

from .interfaces import IResultsFormatter


class RichResultsFormatter(IResultsFormatter):
    """Форматирование результатов с использованием Rich"""
    
    def __init__(self):
        self.console = Console()
    
    def format_single_result(self, results: Dict, pair_name: Optional[str] = None) -> None:
        """Форматирование результата одной пары"""
        title = "Результаты бэктеста"
        if pair_name:
            title += f" для пары {pair_name}"
            
        results_table = Table(title=title)
        results_table.add_column("Метрика", style="cyan")
        results_table.add_column("Значение", style="green")
        
        # Добавляем разделитель для коинтеграции
        results_table.add_row("[bold blue]Тест коинтеграции[/bold blue]", "")
        for param, value in results['cointegration'].items():
            if param != 'critical_values':
                results_table.add_row(
                    param,
                    f"{value:.4f}" if isinstance(value, float) else str(value)
                )
        
        # Добавляем разделитель для временных интервалов
        results_table.add_row("[bold blue]Временные интервалы[/bold blue]", "")
        time_info = results['performance']['time_info']
        results_table.add_row("Всего часов", f"{time_info['total_hours']:.0f}")
        results_table.add_row("Торговых дней", f"{time_info['trading_days']:.1f}")
        results_table.add_row("Годовой коэффициент", f"{time_info['annual_factor']:.2f}")
        
        # Добавляем разделитель для метрик производительности
        results_table.add_row("[bold blue]Метрики производительности[/bold blue]", "")
        for metric, value in results['performance'].items():
            if metric != 'time_info':
                results_table.add_row(
                    metric,
                    f"{value:.2%}" if metric != 'sharpe_ratio' else f"{value:.2f}"
                )
        
        # Добавляем разделитель для анализа рисков
        results_table.add_row("[bold blue]Анализ рисков[/bold blue]", "")
        results_table.add_row("VaR 95%", f"{results['risk_analysis']['var_95']:.2%}")
        results_table.add_row("CVaR 95%", f"{results['risk_analysis']['cvar_95']:.2%}")
        
        self.console.print(results_table)
    
    def format_summary_results(self, results_dict: Dict[str, Dict]) -> None:
        """Форматирование сводных результатов"""
        summary_table = Table(title="Сводные результаты бэктеста")
        summary_table.add_column("Пара", style="cyan")
        summary_table.add_column("Коэффициент Шарпа", style="green")
        summary_table.add_column("Годовая доходность", style="green")
        summary_table.add_column("Максимальная просадка", style="green")
        summary_table.add_column("VaR 95%", style="green")
        summary_table.add_column("CVaR 95%", style="green")
        
        for pair, results in results_dict.items():
            summary_table.add_row(
                pair,
                f"{results['performance']['sharpe_ratio']:.2f}",
                f"{results['performance']['annual_return']:.2%}",
                f"{results['performance']['max_drawdown']:.2%}",
                f"{results['risk_analysis']['var_95']:.2%}",
                f"{results['risk_analysis']['cvar_95']:.2%}"
            )
        
        self.console.print(summary_table)


class HorizontalResultsFormatter(IResultsFormatter):
    """Форматирование результатов в горизонтальном виде"""
    
    def __init__(self):
        self.console = Console()
    
    def format_single_result(self, results: Dict, pair_name: Optional[str] = None) -> None:
        """Форматирование результата одной пары в горизонтальном виде"""
        tables = []
        
        # Таблица коинтеграции
        cointegration_table = Table(title="Тест коинтеграции")
        cointegration_table.add_column("Параметр", style="cyan")
        cointegration_table.add_column("Значение", style="green")
        for param, value in results['cointegration'].items():
            if param != 'critical_values':
                cointegration_table.add_row(
                    param,
                    f"{value:.4f}" if isinstance(value, float) else str(value)
                )
        tables.append(cointegration_table)
        
        # Таблица временных интервалов
        time_table = Table(title="Временные интервалы")
        time_table.add_column("Метрика", style="cyan")
        time_table.add_column("Значение", style="green")
        time_info = results['performance']['time_info']
        time_table.add_row("Всего часов", f"{time_info['total_hours']:.0f}")
        time_table.add_row("Торговых дней", f"{time_info['trading_days']:.1f}")
        time_table.add_row("Годовой коэффициент", f"{time_info['annual_factor']:.2f}")
        tables.append(time_table)
        
        # Таблица метрик производительности
        performance_table = Table(title="Метрики производительности")
        performance_table.add_column("Метрика", style="cyan")
        performance_table.add_column("Значение", style="green")
        for metric, value in results['performance'].items():
            if metric != 'time_info':
                performance_table.add_row(
                    metric,
                    f"{value:.2%}" if metric != 'sharpe_ratio' else f"{value:.2f}"
                )
        tables.append(performance_table)
        
        # Таблица анализа рисков
        risk_table = Table(title="Анализ рисков")
        risk_table.add_column("Метрика", style="cyan")
        risk_table.add_column("Значение", style="green")
        risk_table.add_row("VaR 95%", f"{results['risk_analysis']['var_95']:.2%}")
        risk_table.add_row("CVaR 95%", f"{results['risk_analysis']['cvar_95']:.2%}")
        tables.append(risk_table)
        
        # Выводим все таблицы в одну строку
        title = "Результаты бэктеста"
        if pair_name:
            title += f" для пары {pair_name}"
            
        self.console.print(Panel.fit(
            Group(*tables),
            title=title,
            border_style="blue"
        ))
    
    def format_summary_results(self, results_dict: Dict[str, Dict]) -> None:
        """Форматирование сводных результатов"""
        # Используем тот же метод, что и в RichResultsFormatter
        formatter = RichResultsFormatter()
        formatter.format_summary_results(results_dict) 