"""Подгонка измеренного спектра под закон Планка без SciPy.

Идея: при фиксированной температуре модель линейна по амплитуде и смещению,

    I(λ) = a · P(λ, T) + b,

поэтому для каждого T задача решается методом наименьших квадратов, а по T
остаётся одномерный поиск минимума (сетка + золотое сечение). Такой подход
даёт ту же точность, что ``scipy.optimize.curve_fit``, но не тянет SciPy
в зависимости и работает детерминированно.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .planck import planck_normalized
from .spectrum import Spectrum

__all__ = ["PlanckFit", "fit_planck", "fit_planck_spectrum"]


@dataclass(frozen=True)
class PlanckFit:
    """Результат подгонки по закону Планка."""

    temperature: float
    temperature_err: float
    scale: float
    offset: float
    r_squared: float
    rmse: float
    n_points: int
    success: bool = True
    message: str = ""
    t_range: tuple[float, float] = (0.0, 0.0)

    def model(self, wavelengths) -> np.ndarray:
        """Значения подогнанной кривой в произвольных точках."""
        wl = np.asarray(wavelengths, dtype=float)
        return self.scale * planck_normalized(wl, self.temperature) + self.offset

    def as_dict(self) -> dict[str, float | int | bool | str]:
        return {
            "temperature_k": round(self.temperature, 2),
            "temperature_err_k": round(self.temperature_err, 2),
            "scale": self.scale,
            "offset": self.offset,
            "r_squared": round(self.r_squared, 6),
            "rmse": self.rmse,
            "n_points": self.n_points,
            "success": self.success,
            "message": self.message,
        }


def _linear_fit(wl: np.ndarray, values: np.ndarray, temperature: float):
    """МНК для фиксированной T: возвращает (a, b, SSE, модель)."""
    shape = planck_normalized(wl, temperature)
    design = np.column_stack([shape, np.ones_like(shape)])
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    scale, offset = float(coefficients[0]), float(coefficients[1])
    model = design @ coefficients
    residual = values - model
    return scale, offset, float(np.sum(residual**2)), model


def _golden_section_minimum(
    function,
    lower: float,
    upper: float,
    tolerance: float = 1e-4,
    max_iterations: int = 120,
) -> float:
    """Минимум функции на отрезке методом золотого сечения."""
    inv_phi = (np.sqrt(5.0) - 1.0) / 2.0
    left, right = float(lower), float(upper)
    c = right - inv_phi * (right - left)
    d = left + inv_phi * (right - left)
    fc, fd = function(c), function(d)
    for _ in range(max_iterations):
        if abs(right - left) <= tolerance * max(1.0, abs(right) + abs(left)):
            break
        if fc < fd:
            right, d, fd = d, c, fc
            c = right - inv_phi * (right - left)
            fc = function(c)
        else:
            left, c, fc = c, d, fd
            d = left + inv_phi * (right - left)
            fd = function(d)
    return 0.5 * (left + right)


def fit_planck(
    wavelengths,
    intensity,
    t_range: tuple[float, float] = (700.0, 12000.0),
    n_grid: int = 320,
    wl_range: tuple[float, float] | None = None,
) -> PlanckFit:
    """Подогнать спектр законом Планка и вернуть температуру с погрешностью.

    Parameters
    ----------
    wavelengths, intensity:
        Данные спектра (нм и произвольные единицы).
    t_range:
        Диапазон поиска температуры, К.
    n_grid:
        Число точек грубой сетки по температуре.
    wl_range:
        Необязательный рабочий диапазон длин волн; края, где шум или
        обрезание сенсора, обычно только мешают.
    """
    wl = np.asarray(wavelengths, dtype=float).ravel()
    values = np.asarray(intensity, dtype=float).ravel()
    if wl.size != values.size:
        raise ValueError("Длины массивов не совпадают")
    if wl.size < 8:
        raise ValueError("Для подгонки нужно минимум 8 точек")

    mask = np.isfinite(wl) & np.isfinite(values)
    if wl_range is not None:
        low, high = sorted(wl_range)
        mask &= (wl >= low) & (wl <= high)
    wl, values = wl[mask], values[mask]
    if wl.size < 8:
        raise ValueError("После отсечения диапазона осталось меньше 8 точек")

    order = np.argsort(wl)
    wl, values = wl[order], values[order]

    t_low, t_high = float(t_range[0]), float(t_range[1])
    if not (0 < t_low < t_high):
        raise ValueError("t_range должен быть положительным и возрастающим")

    spread = float(np.ptp(values))
    if spread <= 0 or not np.isfinite(spread):
        return PlanckFit(
            temperature=0.0,
            temperature_err=0.0,
            scale=0.0,
            offset=0.0,
            r_squared=0.0,
            rmse=0.0,
            n_points=int(wl.size),
            success=False,
            message="Спектр постоянный — подгонка невозможна",
            t_range=(t_low, t_high),
        )

    grid = np.geomspace(t_low, t_high, int(np.clip(n_grid, 32, 2000)))
    sse = np.empty_like(grid)
    for index, temperature in enumerate(grid):
        sse[index] = _linear_fit(wl, values, float(temperature))[2]
    best = int(np.argmin(sse))

    left = grid[max(best - 1, 0)]
    right = grid[min(best + 1, grid.size - 1)]
    objective = lambda t: _linear_fit(wl, values, float(t))[2]
    temperature = _golden_section_minimum(objective, float(left), float(right))

    scale, offset, sse_best, model = _linear_fit(wl, values, temperature)
    residual = values - model
    total = float(np.sum((values - values.mean()) ** 2))
    r_squared = 1.0 - sse_best / total if total > 0 else 0.0
    rmse = float(np.sqrt(sse_best / wl.size))

    # Погрешность температуры из ковариационной матрицы (3 параметра).
    delta = max(temperature * 1e-4, 0.01)
    shape = planck_normalized(wl, temperature)
    shape_plus = planck_normalized(wl, temperature + delta)
    shape_minus = planck_normalized(wl, temperature - delta)
    derivative = (shape_plus - shape_minus) / (2.0 * delta)
    jacobian = np.column_stack([scale * derivative, shape, np.ones_like(shape)])

    temperature_err = 0.0
    dof = wl.size - 3
    if dof > 0 and sse_best > 0:
        sigma_squared = sse_best / dof
        try:
            covariance = sigma_squared * np.linalg.pinv(jacobian.T @ jacobian)
            temperature_err = float(np.sqrt(max(covariance[0, 0], 0.0)))
        except np.linalg.LinAlgError:
            temperature_err = 0.0

    message = ""
    if best == 0 or best == grid.size - 1:
        message = "Минимум на границе диапазона температур — расширьте t_range"
    if r_squared < 0.5:
        message = (message + "; " if message else "") + "Низкий R² — источник не похож на чёрное тело"

    return PlanckFit(
        temperature=float(temperature),
        temperature_err=temperature_err,
        scale=scale,
        offset=offset,
        r_squared=float(r_squared),
        rmse=rmse,
        n_points=int(wl.size),
        success=True,
        message=message,
        t_range=(t_low, t_high),
    )


def fit_planck_spectrum(
    spectrum: Spectrum,
    t_range: tuple[float, float] = (700.0, 12000.0),
    n_grid: int = 320,
    wl_range: tuple[float, float] | None = None,
) -> PlanckFit:
    """Удобная обёртка :func:`fit_planck` для объекта :class:`Spectrum`."""
    return fit_planck(
        spectrum.wavelengths,
        spectrum.intensity,
        t_range=t_range,
        n_grid=n_grid,
        wl_range=wl_range,
    )
