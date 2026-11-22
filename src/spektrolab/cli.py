"""Командная строка SpektroLab.

Примеры::

    spektrolab selftest
    spektrolab simulate --temperature 3000 --noise 0.01
    spektrolab analyze data.csv --report protocol.md
    spektrolab calibrate --pixels 100 300 600 900 --wavelengths 435.83 546.07 576.96 611.60
    spektrolab peaks data.csv --element Hg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .core import (
    Measurement,
    Spectrum,
    calibrate_from_lines,
    detect_peaks,
    fit_planck,
    identify,
    linear_calibration,
    planck_spectral_radiance,
    wien_peak_nm,
)
from .core.simulator import blackbody_spectrum, mercury_spectrum


def _print_title(text: str) -> None:
    print()
    print(text)
    print("=" * len(text))


def _print_fit(fit) -> None:
    print(f"  Температура: {fit.temperature:8.1f} ± {fit.temperature_err:.1f} К")
    print(f"  R2:          {fit.r_squared:8.4f}")
    print(f"  RMSE:        {fit.rmse:8.4f}")
    if fit.message:
        print(f"  Примечание:  {fit.message}")


def _print_peaks(peaks, element: str | None = None, tolerance: float = 2.0) -> None:
    if not peaks:
        print("  Линии не найдены — уменьшите порог --prominence")
        return
    print("  №    длина, нм    Интенсивность   FWHM, нм   Линия")
    for number, peak in enumerate(peaks, start=1):
        fwhm = peak.get("fwhm_nm")
        fwhm_text = f"{float(fwhm):.2f}" if fwhm is not None and np.isfinite(float(fwhm)) else "—"
        label = ""
        if element:
            matches = identify([peak["wavelength_nm"]], element, tolerance)
            if matches:
                match = matches[0]
                label = f"{match['element']} {match['reference_nm']:.2f} " f"({match['error_nm']:+.2f})"
        print(
            f"  {number:<4} {peak['wavelength_nm']:8.2f}   {peak['intensity']:10.1f}   "
            f"{fwhm_text:>8}   {label}"
        )


# ----------------------------------------------------------------------
# команды
# ----------------------------------------------------------------------
def cmd_selftest(args: argparse.Namespace) -> int:
    """Быстрая самопроверка математики без железа и тестового фреймворка."""
    _print_title("SpektroLab - самопроверка")
    failures: list[str] = []

    for target in (2000.0, 3200.0, 6500.0):
        spectrum = blackbody_spectrum(temperature=target, noise=0.004, seed=7)
        fit = fit_planck(spectrum.wavelengths, spectrum.intensity)
        relative = abs(fit.temperature - target) / target
        ok = relative < 0.01 and fit.r_squared > 0.99
        status = "OK " if ok else "FAIL"
        print(
            f"[{status}] Планк: цель {target:6.0f} К -> {fit.temperature:7.1f} К "
            f"({relative * 100:+.2f} %), R2={fit.r_squared:.4f}"
        )
        if not ok:
            failures.append(f"фит {target:.0f} К")

    peak = wien_peak_nm(3000.0)
    ok = abs(peak - 965.9) < 0.5
    print(f"[{'OK ' if ok else 'FAIL'}] Вина: T=3000 К -> lambda_max={peak:.1f} нм")
    if not ok:
        failures.append("закон Вина")

    spectrum = mercury_spectrum(noise=0.005, seed=3)
    peaks = detect_peaks(spectrum, prominence=0.05)
    lines = [peak["wavelength_nm"] for peak in peaks]
    ok = any(abs(line - 546.07) < 0.7 for line in lines) and len(lines) >= 4
    print(f"[{'OK ' if ok else 'FAIL'}] Линии: найдено {len(lines)} -> " f"{', '.join(f'{line:.1f}' for line in lines)}")
    if not ok:
        failures.append("поиск линий")

    pixels = np.array([102.0, 304.0, 611.0, 903.0])
    wavelengths = 340.0 + 0.68 * pixels + 1.2e-4 * pixels**2
    calibration = calibrate_from_lines(pixels, wavelengths, degree=2, pixel_count=1024)
    ok = calibration.rms_nm < 1e-6
    print(
        f"[{'OK ' if ok else 'FAIL'}] Калибровка: RMS={calibration.rms_nm:.4f} нм, "
        f"диапазон {calibration.wavelength_range[0]:.0f}–{calibration.wavelength_range[1]:.0f} нм"
    )
    if not ok:
        failures.append("калибровка")

    loaded = type(calibration).from_dict(calibration.to_dict())
    ok = loaded.coefficients == calibration.coefficients
    print(f"[{'OK ' if ok else 'FAIL'}] Калибровка: сохранение/загрузка (JSON)")
    if not ok:
        failures.append("сохранение калибровки")

    source = blackbody_spectrum(3000.0, noise=0.002, seed=1)
    fit = fit_planck(source.wavelengths, source.intensity)
    measurement = Measurement(spectrum=source, fit=fit, device="selftest")
    markdown = measurement.report_markdown()
    ok = "Температура по закону Планка" in markdown and "3000" in markdown.replace(" ", "")
    print(f"[{'OK ' if ok else 'FAIL'}] Протокол: Markdown сформирован ({len(markdown)} символов)")
    if not ok:
        failures.append("протокол")

    print()
    if failures:
        print("Провалено:", ", ".join(failures))
        return 1
    print("Все проверки пройдены.")
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    _print_title("SpektroLab - синтетический спектр")
    spectrum = blackbody_spectrum(
        temperature=args.temperature, noise=args.noise, seed=args.seed
    )
    print(
        f"Источник: чёрное тело {args.temperature:.0f} К, "
        f"{spectrum.size} точек, {spectrum.wavelength_range[0]:.0f}–"
        f"{spectrum.wavelength_range[1]:.0f} нм"
    )
    fit = fit_planck(
        spectrum.wavelengths,
        spectrum.intensity,
        t_range=(args.t_range[0], args.t_range[1]),
    )
    print("\nПодгонка по закону Планка:")
    _print_fit(fit)

    if args.csv:
        spectrum.save_csv(args.csv)
        print(f"\nСпектр сохранён: {args.csv}")
    if args.json:
        Measurement(spectrum=spectrum, fit=fit, device="simulator").save_json(args.json)
        print(f"Измерение сохранено: {args.json}")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    _print_title(f"SpektroLab - анализ {Path(args.csv).name}")
    spectrum = Spectrum.load_csv(args.csv)
    print(
        f"Спектр: {spectrum.wavelength_range[0]:.1f}–{spectrum.wavelength_range[1]:.1f} нм, "
        f"{spectrum.size} точек, шаг {spectrum.step_nm:.2f} нм"
    )

    fit = fit_planck(
        spectrum.wavelengths,
        spectrum.intensity,
        t_range=(args.t_range[0], args.t_range[1]),
        wl_range=(args.wl_range[0], args.wl_range[1]) if args.wl_range else None,
    )
    print("\nПодгонка по закону Планка:")
    _print_fit(fit)

    peaks = []
    if args.peaks:
        prominence = args.prominence * float(np.max(spectrum.intensity))
        peaks = detect_peaks(spectrum, prominence=prominence)
        print("\nСпектральные линии:")
        _print_peaks(peaks, args.element, args.tolerance)

    measurement = Measurement(
        spectrum=spectrum, fit=fit, peaks=peaks, device=args.device or "файл"
    )
    if args.csv_out:
        measurement.save_csv(args.csv_out)
        print(f"\nCSV с кривой Планка: {args.csv_out}")
    if args.json:
        measurement.save_json(args.json)
        print(f"JSON: {args.json}")
    if args.report:
        measurement.save_report(args.report)
        print(f"Протокол: {args.report}")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    _print_title("SpektroLab - калибровка")
    pixels = [float(value) for value in args.pixels]
    wavelengths = [float(value) for value in args.wavelengths]
    if len(pixels) != len(wavelengths):
        raise ValueError("Количество --pixels и --wavelengths должно совпадать")

    calibration = calibrate_from_lines(
        pixels,
        wavelengths,
        degree=args.degree,
        pixel_count=args.pixel_count,
        device=args.device,
    )
    print(f"Степень полинома: {calibration.degree}")
    print(f"RMS: {calibration.rms_nm:.3f} нм, максимум: {calibration.max_error_nm:.3f} нм")
    first, last = calibration.wavelength_range
    print(f"Диапазон: {first:.1f}–{last:.1f} нм на {calibration.pixel_count} пикселях")

    coefficients = ", ".join(f"{value:.6g}" for value in calibration.coefficients)
    print(f"Коэффициенты: {coefficients}")

    if args.out:
        calibration.save(args.out)
        print(f"\nСохранено: {args.out}")
    return 0


def cmd_calibration_info(args: argparse.Namespace) -> int:
    from .core import Calibration

    _print_title("SpektroLab - калибровка")
    calibration = Calibration.load(args.path)
    first, last = calibration.wavelength_range
    print(f"Степень: {calibration.degree}")
    print(f"RMS: {calibration.rms_nm:.3f} нм")
    print(f"Диапазон: {first:.1f}–{last:.1f} нм")
    print(f"Точек привязки: {len(calibration.points)}")
    print(f"Создана: {calibration.created_at}")
    if calibration.note:
        print(f"Примечание: {calibration.note}")
    return 0


def cmd_peaks(args: argparse.Namespace) -> int:
    _print_title(f"SpektroLab - линии {Path(args.csv).name}")
    spectrum = Spectrum.load_csv(args.csv)
    prominence = args.prominence * float(np.max(spectrum.intensity))
    peaks = detect_peaks(spectrum, prominence=prominence)
    _print_peaks(peaks, args.element, args.tolerance)
    return 0


def cmd_linear_calibration(args: argparse.Namespace) -> int:
    calibration = linear_calibration(
        args.pixel_count, args.wl_min, args.wl_max, device=args.device
    )
    if args.out:
        calibration.save(args.out)
        print(f"Сохранено: {args.out}")
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    from .ui import run

    return run()


# ----------------------------------------------------------------------
# парсер
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spektrolab",
        description="SpektroLab — учебный спектрометр: анализ, калибровка, протоколы.",
    )
    parser.add_argument("--version", action="version", version=f"spektrolab {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    selftest = subparsers.add_parser("selftest", help="самопроверка математики")
    selftest.set_defaults(func=cmd_selftest)

    simulate = subparsers.add_parser("simulate", help="синтетический спектр и подгонка")
    simulate.add_argument("--temperature", type=float, default=3000.0, help="температура, К")
    simulate.add_argument("--noise", type=float, default=0.01, help="уровень шума 0–1")
    simulate.add_argument("--seed", type=int, default=7)
    simulate.add_argument("--t-range", type=float, nargs=2, default=(700.0, 12000.0),
                          metavar=("MIN", "MAX"))
    simulate.add_argument("--csv", type=Path, help="куда сохранить спектр")
    simulate.add_argument("--json", type=Path, help="куда сохранить измерение")
    simulate.set_defaults(func=cmd_simulate)

    analyze = subparsers.add_parser("analyze", help="анализ спектра из CSV")
    analyze.add_argument("csv", type=Path)
    analyze.add_argument("--t-range", type=float, nargs=2, default=(700.0, 12000.0),
                         metavar=("MIN", "MAX"))
    analyze.add_argument("--wl-range", type=float, nargs=2, default=None, metavar=("MIN", "MAX"))
    analyze.add_argument("--peaks", action="store_true", help="найти спектральные линии")
    analyze.add_argument("--prominence", type=float, default=0.05,
                         help="порог линии, доля от максимума (по умолчанию 0.05)")
    analyze.add_argument("--element", default=None, help="элемент для подписи линий")
    analyze.add_argument("--tolerance", type=float, default=2.0, help="допуск, нм")
    analyze.add_argument("--device", default="")
    analyze.add_argument("--csv-out", type=Path, help="CSV с кривой Планка")
    analyze.add_argument("--json", type=Path, help="JSON измерения")
    analyze.add_argument("--report", type=Path, help="Markdown-протокол")
    analyze.set_defaults(func=cmd_analyze)

    calibrate = subparsers.add_parser("calibrate", help="калибровка по эталонным линиям")
    calibrate.add_argument("--pixels", type=float, nargs="+", required=True)
    calibrate.add_argument("--wavelengths", type=float, nargs="+", required=True)
    calibrate.add_argument("--degree", type=int, default=2, choices=(1, 2, 3))
    calibrate.add_argument("--pixel-count", type=int, default=1024)
    calibrate.add_argument("--device", default="")
    calibrate.add_argument("--out", type=Path, help="куда сохранить JSON")
    calibrate.set_defaults(func=cmd_calibrate)

    info = subparsers.add_parser("calibration-info", help="показать калибровку")
    info.add_argument("path", type=Path)
    info.set_defaults(func=cmd_calibration_info)

    peaks = subparsers.add_parser("peaks", help="найти линии в CSV")
    peaks.add_argument("csv", type=Path)
    peaks.add_argument("--prominence", type=float, default=0.05)
    peaks.add_argument("--element", default=None)
    peaks.add_argument("--tolerance", type=float, default=2.0)
    peaks.set_defaults(func=cmd_peaks)

    linear = subparsers.add_parser("linear-calibration", help="калибровка по паспорту")
    linear.add_argument("--pixel-count", type=int, default=1024)
    linear.add_argument("--wl-min", type=float, default=340.0)
    linear.add_argument("--wl-max", type=float, default=1050.0)
    linear.add_argument("--device", default="")
    linear.add_argument("--out", type=Path, help="куда сохранить JSON")
    linear.set_defaults(func=cmd_linear_calibration)

    ui = subparsers.add_parser("ui", help="запустить графическое приложение")
    ui.set_defaults(func=cmd_ui)

    return parser


def _configure_streams() -> None:
    """Не падать на консоли Windows с кодировкой CP1251/CP866."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            continue


def main(argv: list[str] | None = None) -> int:
    _configure_streams()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\nПрервано пользователем", file=sys.stderr)
        return 130
    except Exception as error:  # noqa: BLE001 - CLI должен печатать ошибку, а не трейс
        print(f"Ошибка: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


