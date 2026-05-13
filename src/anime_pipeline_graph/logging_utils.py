"""Logging utilities with rich."""

from rich.console import Console
from rich.table import Table

console = Console()


def log_step(title: str, detail: str) -> None:
    """Print a concise step log."""
    console.rule(f"[bold cyan]{title}")
    console.print(detail)


def log_kv(title: str, payload: dict) -> None:
    """Print key-value table."""
    table = Table(title=title)
    table.add_column("Key")
    table.add_column("Value")
    for key, value in payload.items():
        table.add_row(str(key), str(value))
    console.print(table)
