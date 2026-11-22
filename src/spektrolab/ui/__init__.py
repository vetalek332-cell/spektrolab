"""Графический интерфейс SpektroLab."""


def run() -> int:
    """Запустить приложение и вернуть код завершения."""
    from .app import main

    return main()


__all__ = ["run"]
