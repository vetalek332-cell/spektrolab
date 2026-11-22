# -*- coding: utf-8 -*-
"""Вспомогательные пути для тестов (без временных каталогов ОС)."""

from pathlib import Path

ARTIFACTS = Path(__file__).resolve().parent / "_artifacts"


def artifact(name: str) -> Path:
    """Путь к файлу в каталоге тестовых артефактов (создаётся при необходимости)."""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS / name
