"""Табличные спектральные линии для калибровки и распознавания."""

from __future__ import annotations

from typing import Iterable, Sequence

__all__ = [
    "REFERENCE_LINES",
    "CALIBRATION_SOURCES",
    "lines_for",
    "identify",
    "nearest_line",
]

# Длины волн в нанометрах (вакуум/воздух — разница для школьных задач несущественна)
REFERENCE_LINES: dict[str, dict[str, float]] = {
    "Hg": {  # ртутная лампа низкого давления
        "404.66 нм": 404.66,
        "435.83 нм": 435.83,
        "546.07 нм": 546.07,
        "576.96 нм": 576.96,
        "579.07 нм": 579.07,
        "611.60 нм": 611.60,
    },
    "Na": {  # натриевая лампа, дублет D
        "589.00 нм": 589.00,
        "589.59 нм": 589.59,
        "568.80 нм": 568.80,
        "498.30 нм": 498.30,
    },
    "Ne": {  # неоновая лампа
        "585.25 нм": 585.25,
        "588.19 нм": 588.19,
        "594.48 нм": 594.48,
        "607.43 нм": 607.43,
        "614.31 нм": 614.31,
        "616.36 нм": 616.36,
        "621.73 нм": 621.73,
        "626.65 нм": 626.65,
        "630.48 нм": 630.48,
        "638.30 нм": 638.30,
        "640.22 нм": 640.22,
        "650.65 нм": 650.65,
        "659.90 нм": 659.90,
        "667.83 нм": 667.83,
        "671.70 нм": 671.70,
        "692.95 нм": 692.95,
        "703.24 нм": 703.24,
        "717.39 нм": 717.39,
    },
    "H": {  # бальмеровская серия водорода
        "410.17 нм": 410.17,
        "434.05 нм": 434.05,
        "486.13 нм": 486.13,
        "656.28 нм": 656.28,
    },
    "He": {  # гелий
        "447.15 нм": 447.15,
        "471.31 нм": 471.31,
        "492.19 нм": 492.19,
        "501.57 нм": 501.57,
        "587.56 нм": 587.56,
        "667.82 нм": 667.82,
        "706.52 нм": 706.52,
    },
    "Cd": {  # кадмий
        "467.82 нм": 467.82,
        "480.00 нм": 480.00,
        "508.58 нм": 508.58,
        "643.85 нм": 643.85,
    },
}

# Лампы, удобные для калибровки: подпись → элементы
CALIBRATION_SOURCES: dict[str, tuple[str, ...]] = {
    "Ртутная лампа": ("Hg",),
    "Натриевая лампа": ("Na",),
    "Неоновая лампа": ("Ne",),
    "Водородная лампа": ("H",),
    "Гелий-неоновый лазер": ("He", "Ne"),
    "Лазерные диоды": ("Lasers",),
}

_LASERS = {
    "405 нм": 405.0,
    "450 нм": 450.0,
    "473 нм": 473.0,
    "520 нм": 520.0,
    "532 нм": 532.0,
    "635 нм": 635.0,
    "650 нм": 650.0,
    "780 нм": 780.0,
}
REFERENCE_LINES["Lasers"] = _LASERS


def lines_for(elements: str | Iterable[str] | None = None) -> list[tuple[str, float]]:
    """Список ``(элемент, длина волны)`` для выбранных источников."""
    if elements is None:
        names: Sequence[str] = tuple(REFERENCE_LINES)
    elif isinstance(elements, str):
        names = (elements,)
    else:
        names = tuple(elements)

    result: list[tuple[str, float]] = []
    for name in names:
        for wavelength in REFERENCE_LINES.get(name, {}).values():
            result.append((name, float(wavelength)))
    return sorted(result, key=lambda item: item[1])


def nearest_line(
    wavelength_nm: float, elements: str | Iterable[str] | None = None
) -> tuple[str, float, float] | None:
    """Ближайшая табличная линия: ``(элемент, λ_табл, ошибка)``."""
    table = lines_for(elements)
    if not table:
        return None
    element, reference = min(table, key=lambda item: abs(item[1] - wavelength_nm))
    return element, reference, float(wavelength_nm - reference)


def identify(
    wavelengths_nm: Sequence[float],
    elements: str | Iterable[str] | None = None,
    tolerance_nm: float = 2.0,
) -> list[dict[str, float | str]]:
    """Сопоставить измеренные линии с табличными в пределах допуска."""
    table = lines_for(elements)
    matches: list[dict[str, float | str]] = []
    if not table:
        return matches
    for measured in wavelengths_nm:
        element, reference = min(table, key=lambda item: abs(item[1] - measured))
        error = float(measured - reference)
        if abs(error) <= tolerance_nm:
            matches.append(
                {
                    "measured_nm": float(measured),
                    "reference_nm": float(reference),
                    "element": element,
                    "error_nm": error,
                }
            )
    return sorted(matches, key=lambda item: float(item["measured_nm"]))
