# -*- coding: utf-8 -*-
"""Минимальный пример: синтетический спектр → температура → протокол.

Запуск из корня репозитория::

    python examples/quickstart.py
"""

import sys
from pathlib import Path

from spektrolab import Measurement, detect_peaks, fit_planck, identify
from spektrolab.core.simulator import blackbody_spectrum


def _configure_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            continue


def main() -> None:
    _configure_streams()
    # 1. «Измеряем» лампу накаливания: 2900 К, 0.4 % шума
    spectrum = blackbody_spectrum(temperature=2900.0, noise=0.004, seed=42)
    print(
        f"Спектр: {spectrum.wavelength_range[0]:.0f}–{spectrum.wavelength_range[1]:.0f} нм, "
        f"{spectrum.size} точек"
    )

    # 2. Подгоняем закон Планка
    fit = fit_planck(spectrum.wavelengths, spectrum.intensity)
    print(f"Температура: {fit.temperature:.0f} ± {fit.temperature_err:.0f} К")
    print(f"R² = {fit.r_squared:.4f}, RMSE = {fit.rmse:.4f}")

    # 3. Ищем линии и пробуем распознать (для чёрного тела их не будет)
    peaks = detect_peaks(spectrum, prominence=0.05)
    matches = identify([peak["wavelength_nm"] for peak in peaks], elements="Hg")
    print(f"Линий найдено: {len(peaks)}, распознано: {len(matches)}")

    # 4. Сохраняем протокол
    measurement = Measurement(
        spectrum=spectrum,
        fit=fit,
        peaks=peaks,
        device="СпектроЛаб USB · пример",
        operator="examples/quickstart.py",
        notes="Синтетический спектр, лампа накаливания 2900 К.",
    )
    output = Path("example_measurement.json")
    measurement.save_json(output)
    print(f"\nПротокол сохранён: {output}")
    print(measurement.report_markdown())


if __name__ == "__main__":
    main()
