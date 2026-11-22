"""Протокол измерения: сохранение, экспорт и отчёт."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .calibration import Calibration
from .fit import PlanckFit
from .spectrum import Spectrum

__all__ = ["Measurement"]


@dataclass
class Measurement:
    """Одно измерение: спектр + подгонка + линии + калибровка."""

    spectrum: Spectrum
    fit: PlanckFit | None = None
    peaks: list[dict[str, float]] = field(default_factory=list)
    calibration: Calibration | None = None
    device: str = ""
    operator: str = ""
    notes: str = ""
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now().isoformat(timespec="seconds")

    # ------------------------------------------------------------------
    # сериализация
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        peaks = [
            {
                "wavelength_nm": round(float(peak.get("wavelength_nm", 0.0)), 3),
                "intensity": round(float(peak.get("intensity", 0.0)), 3),
                "fwhm_nm": (
                    round(float(peak["fwhm_nm"]), 3)
                    if peak.get("fwhm_nm") is not None
                    and np.isfinite(float(peak.get("fwhm_nm", np.nan)))
                    else None
                ),
            }
            for peak in self.peaks
        ]
        return {
            "format": "spektrolab.measurement",
            "version": 1,
            "created_at": self.created_at,
            "device": self.device,
            "operator": self.operator,
            "notes": self.notes,
            "wavelength_nm": [float(value) for value in self.spectrum.wavelengths],
            "intensity": [float(value) for value in self.spectrum.intensity],
            "fit": self.fit.as_dict() if self.fit else None,
            "peaks": peaks,
            "calibration": self.calibration.to_dict() if self.calibration else None,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Measurement":
        if payload.get("format") not in (None, "spektrolab.measurement"):
            raise ValueError("Это не файл измерения SpektroLab")
        spectrum = Spectrum(
            np.asarray(payload["wavelength_nm"], dtype=float),
            np.asarray(payload["intensity"], dtype=float),
            label=str(payload.get("device", "")),
        )
        fit_payload = payload.get("fit")
        fit = None
        if fit_payload:
            fit = PlanckFit(
                temperature=float(fit_payload["temperature_k"]),
                temperature_err=float(fit_payload.get("temperature_err_k", 0.0)),
                scale=float(fit_payload.get("scale", 0.0)),
                offset=float(fit_payload.get("offset", 0.0)),
                r_squared=float(fit_payload.get("r_squared", 0.0)),
                rmse=float(fit_payload.get("rmse", 0.0)),
                n_points=int(fit_payload.get("n_points", spectrum.size)),
                success=bool(fit_payload.get("success", True)),
                message=str(fit_payload.get("message", "")),
            )
        calibration = None
        if payload.get("calibration"):
            calibration = Calibration.from_dict(payload["calibration"])
        return cls(
            spectrum=spectrum,
            fit=fit,
            peaks=list(payload.get("peaks", [])),
            calibration=calibration,
            device=str(payload.get("device", "")),
            operator=str(payload.get("operator", "")),
            notes=str(payload.get("notes", "")),
            created_at=str(payload.get("created_at", "")),
        )

    def save_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return target

    @classmethod
    def load_json(cls, path: str | Path) -> "Measurement":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save_csv(self, path: str | Path) -> Path:
        """CSV: λ, интенсивность и, если есть, кривая Планка."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        model = (
            self.fit.model(self.spectrum.wavelengths) if self.fit and self.fit.success else None
        )
        with target.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh, delimiter=";")
            header = ["wavelength_nm", "intensity"]
            if model is not None:
                header.append("planck_fit")
            writer.writerow(header)
            for index, (wavelength, value) in enumerate(self.spectrum.to_rows()):
                row = [wavelength, value]
                if model is not None:
                    row.append(float(model[index]))
                writer.writerow(row)
        return target

    # ------------------------------------------------------------------
    # отчёт
    # ------------------------------------------------------------------
    def report_markdown(self) -> str:
        """Протокол в Markdown — печатается и вставляется в отчёт."""
        low, high = self.spectrum.wavelength_range
        lines = [
            "# Протокол измерения «СпектроЛаб»",
            "",
            f"- **Дата и время:** {self.created_at}",
            f"- **Прибор:** {self.device or 'не указан'}",
            f"- **Оператор:** {self.operator or 'не указан'}",
            f"- **Спектр:** {low:.1f}–{high:.1f} нм, {self.spectrum.size} точек",
        ]
        if self.calibration:
            lines.append(
                f"- **Калибровка:** полином степени {self.calibration.degree}, "
                f"RMS {self.calibration.rms_nm:.2f} нм"
            )
        lines.append("")

        if self.fit and self.fit.success:
            lines += [
                "## Температура по закону Планка",
                "",
                "| Параметр | Значение |",
                "|---|---|",
                f"| Температура | **{self.fit.temperature:.0f} ± "
                f"{self.fit.temperature_err:.0f} К** |",
                f"| R² | {self.fit.r_squared:.4f} |",
                f"| RMSE | {self.fit.rmse:.3f} |",
                f"| Точек в подгонке | {self.fit.n_points} |",
            ]
            if self.fit.message:
                lines.append(f"| Примечание | {self.fit.message} |")
            lines.append("")

        if self.peaks:
            lines += [
                "## Спектральные линии",
                "",
                "| № | λ, нм | Интенсивность | FWHM, нм |",
                "|---|---|---|---|",
            ]
            for number, peak in enumerate(self.peaks, start=1):
                fwhm = peak.get("fwhm_nm")
                fwhm_text = f"{float(fwhm):.2f}" if fwhm and np.isfinite(float(fwhm)) else "—"
                lines.append(
                    f"| {number} | {float(peak['wavelength_nm']):.2f} | "
                    f"{float(peak.get('intensity', 0.0)):.1f} | {fwhm_text} |"
                )
            lines.append("")

        if self.notes:
            lines += ["## Примечания", "", self.notes, ""]

        lines += [
            "---",
            "Подпись оператора: ____________________",
            "",
            "*Отчёт сформирован программой SpektroLab.*",
        ]
        return "\n".join(lines)

    def save_report(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.report_markdown(), encoding="utf-8")
        return target
