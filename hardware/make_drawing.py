#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Генератор габаритного чертежа «СпектроЛаб» (hardware/drawing.svg).

Чертёж строится из тех же расчётных данных, что и optics.py: положения щели,
коллиматора, решётки и камеры, углы дифракции, размеры сенсора. Запуск:

    python hardware/make_drawing.py

Результат — hardware/drawing.svg (вид сверху с ходами лучей и вид сбоку).
"""

from __future__ import annotations

import math
import pathlib
import sys
from xml.sax.saxutils import escape as _xml_escape

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import optics as O  # noqa: E402

CFG = O.DEFAULT
LAY = O.LAYOUT
A = math.radians(LAY.camera_mount_angle_deg)

# --- ключевые точки в миллиметрах (координаты оптической скамьи) ------------
SLIT = (LAY.slit_x_mm, LAY.axis_y_mm)
COLL = (LAY.collimator_x_mm, LAY.axis_y_mm)
GRAT = (LAY.grating_x_mm, LAY.axis_y_mm)
LENS_C = LAY.camera_axis_point(LAY.grating_to_camera_mm)   # передняя плоскость объектива
PLATE_C = LAY.camera_axis_point(38.0)                       # кронштейн и плоскость сенсора
MODULE_END = LAY.camera_axis_point(50.0)                    # задняя плоскость платы
U = (-math.sin(A), math.cos(A))          # направление вдоль сенсора
N = (math.cos(A), math.sin(A))           # ось объектива

SCALE = 3.0
W, H = 1180, 880


def ru(value: float, digits: int = 1) -> str:
    """Число с десятичной запятой (инженерная запись в РФ)."""
    return f"{value:.{digits}f}".replace(".", ",")


class Svg:
    """Холст в миллиметрах с фиксированным масштабом."""

    def __init__(self, ox: float, oy: float, scale: float = SCALE) -> None:
        self.ox, self.oy, self.s = ox, oy, scale
        self.items: list[str] = []

    def X(self, x: float) -> float:
        return self.ox + self.s * x

    def Y(self, y: float) -> float:
        return self.oy - self.s * y

    def line(self, x1, y1, x2, y2, stroke="#111111", width=1.1, dash=None,
             marker=None) -> None:
        d = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="url(#{marker})"' if marker else ""
        self.items.append(
            f'<line x1="{self.X(x1):.2f}" y1="{self.Y(y1):.2f}" '
            f'x2="{self.X(x2):.2f}" y2="{self.Y(y2):.2f}" '
            f'stroke="{stroke}" stroke-width="{width}"{d}{m}/>')

    def rect(self, x, y, w, h, stroke="#111111", width=1.1, fill="none",
             dash=None, rx=0.0) -> None:
        """Прямоугольник: (x, y) — левый нижний угол в мм."""
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.items.append(
            f'<rect x="{self.X(x):.2f}" y="{self.Y(y + h):.2f}" '
            f'width="{self.s * w:.2f}" height="{self.s * h:.2f}" '
            f'rx="{self.s * rx:.2f}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{width}"{d}/>')

    def polygon(self, points, stroke="#111111", width=1.0, fill="none",
                dash=None) -> None:
        d = f' stroke-dasharray="{dash}"' if dash else ""
        pts = " ".join(f"{self.X(x):.2f},{self.Y(y):.2f}" for x, y in points)
        self.items.append(
            f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{width}"{d}/>')

    def circle(self, x, y, r, stroke="#111111", width=1.1, fill="none", dash=None) -> None:
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.items.append(
            f'<circle cx="{self.X(x):.2f}" cy="{self.Y(y):.2f}" r="{self.s * r:.2f}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{width}"{d}/>')

    def text(self, x, y, s, size=11, anchor="start", color="#111111", weight="normal") -> None:
        for i, line in enumerate(s.split("\n")):
            self.items.append(
                f'<text x="{self.X(x):.2f}" y="{self.Y(y) + i * size * 1.25:.2f}" '
                f'font-size="{size}" text-anchor="{anchor}" fill="{color}" '
                f'font-weight="{weight}">{_xml_escape(line)}</text>')

    def px_text(self, x, y, s, size=12, anchor="start", color="#111111",
                weight="normal") -> None:
        for i, line in enumerate(s.split("\n")):
            self.items.append(
                f'<text x="{x:.1f}" y="{y + i * size * 1.3:.1f}" font-size="{size}" '
                f'text-anchor="{anchor}" fill="{color}" font-weight="{weight}">'
                f'{_xml_escape(line)}</text>')

    # --- размеры
    def dim_h(self, x1, x2, y, label, tick=2.6, size=10) -> None:
        self.line(x1, y, x2, y, width=0.9)
        for x in (x1, x2):
            self.line(x, y - tick / 2, x, y + tick / 2, width=0.9)
        self.text((x1 + x2) / 2, y + 1.5, label, size=size, anchor="middle")

    def dim_v(self, x, y1, y2, label, tick=2.6, size=10, left=True) -> None:
        self.line(x, y1, x, y2, width=0.9)
        for y in (y1, y2):
            self.line(x - tick / 2, y, x + tick / 2, y, width=0.9)
        self.text(x + (-1.0 if left else 1.0), (y1 + y2) / 2 - 1.5, label,
                  size=size, anchor="end" if left else "start")

    def dim_along(self, x1, y1, x2, y2, label, offset=3.0, size=10) -> None:
        """Размер вдоль наклонной линии (с выносом по нормали)."""
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        nx, ny = -dy / length, dx / length
        self.line(x1, y1, x2, y2, width=0.7, dash="1 1")
        for t in (0.0, 1.0):
            px, py = x1 + t * dx, y1 + t * dy
            self.line(px, py, px + nx * 1.6, py + ny * 1.6, width=0.9)
        mx, my = (x1 + x2) / 2 + nx * offset, (y1 + y2) / 2 + ny * offset
        self.text(mx, my, label, size=size, anchor="middle")

    def leader(self, x, y, tx, ty, label, size=10, anchor="start") -> None:
        self.line(x, y, tx, ty, width=0.7)
        self.circle(x, y, 0.35, fill="#111111", width=0.6)
        self.text(tx + (0.8 if anchor == "start" else -0.8), ty + 0.5, label,
                  size=size, anchor=anchor)


def build() -> str:
    top = Svg(220.0, 255.0)
    side = Svg(220.0, 700.0)
    bg = Svg(0, 0, 1.0)

    # =====================================================================
    #  ВИД СВЕРХУ
    # =====================================================================
    top.rect(-60, -40, 120, 80, width=1.6, rx=5)
    top.rect(-58, -38, 116, 76, width=0.8, dash="4 3")
    top.text(-56, -36.5, "Вид сверху (крышка снята)", size=9, color="#777777")

    # щель и кассета
    top.rect(LAY.slit_x_mm, LAY.axis_y_mm - 17, 4, 34, width=1.0)
    top.line(LAY.slit_x_mm, LAY.axis_y_mm - 4, LAY.slit_x_mm, LAY.axis_y_mm + 4,
             stroke="#c00", width=2.4)
    top.text(LAY.slit_x_mm - 1.0, LAY.axis_y_mm + 13.0, "щель", size=9, anchor="middle")
    top.text(LAY.slit_x_mm - 1.0, LAY.axis_y_mm + 10.6,
             f"{ru(CFG.slit_width_mm, 2)}×{CFG.slit_height_mm:.0f}", size=8,
             anchor="middle", color="#c00")

    # коллиматор
    top.rect(-22.5, LAY.axis_y_mm - 20.3, 7.8, 40.6, width=1.0)
    top.circle(LAY.collimator_x_mm, LAY.axis_y_mm, 12.7, width=1.0)
    top.circle(LAY.collimator_x_mm, LAY.axis_y_mm, CFG.collimator_stop_mm / 2,
               stroke="#066", width=1.4, dash="2 2")
    top.leader(LAY.collimator_x_mm, LAY.axis_y_mm + 9.0, -45, 34.5,
               f"коллиматор f={CFG.collimator_focal_mm:.0f}\nдиафрагма Ø{ru(CFG.collimator_stop_mm)}",
               size=9)

    # решётка
    top.rect(-12.4, LAY.axis_y_mm - 17, 5.8, 34, width=1.0, fill="url(#hatch)")
    top.leader(GRAT[0], LAY.axis_y_mm + 14.5, -3.0, 34.5,
               "решётка 600 штр/мм\nштрихи вертикально", size=9)

    # камера
    top.circle(LENS_C[0], LENS_C[1], 7.0, width=1.4)
    top.circle(LENS_C[0], LENS_C[1], CFG.camera_pupil_mm / 2, stroke="#066",
               width=1.4, dash="2 2")
    # кронштейн (в плане — две линии толщины 4 мм)
    for off in (-1.8, 1.8):
        p1 = (PLATE_C[0] + 17 * U[0] + off * N[0], PLATE_C[1] + 17 * U[1] + off * N[1])
        p2 = (PLATE_C[0] - 17 * U[0] + off * N[0], PLATE_C[1] - 17 * U[1] + off * N[1])
        top.line(p1[0], p1[1], p2[0], p2[1], width=0.9, stroke="#333333")
    # сенсор
    s1 = (PLATE_C[0] + 2.688 * U[0], PLATE_C[1] + 2.688 * U[1])
    s2 = (PLATE_C[0] - 2.688 * U[0], PLATE_C[1] - 2.688 * U[1])
    top.line(s1[0], s1[1], s2[0], s2[1], width=3.4, stroke="#06c")
    # плата камеры
    b1 = (MODULE_END[0] + 16 * U[0], MODULE_END[1] + 16 * U[1])
    b2 = (MODULE_END[0] - 16 * U[0], MODULE_END[1] - 16 * U[1])
    top.line(b1[0], b1[1], b2[0], b2[1], width=1.0, dash="3 2")
    top.text(22, -30.0, "камера M12 f=6", size=8.5, color="#333333")
    top.text(22, -33.0, "сенсор 1920×1080", size=8.5, color="#333333")
    top.line(19.4, 2.5, 23.0, -27.5, width=0.6, stroke="#777777")
    top.text(58, 44.5, "USB 1920×1080, без ИК-фильтра", size=8,
             anchor="end", color="#333333")

    # осевой и дифрагированные лучи
    top.line(SLIT[0], SLIT[1], GRAT[0], GRAT[1], width=1.2, stroke="#080")
    top.line(GRAT[0], GRAT[1], 50.0, GRAT[1], width=0.9, stroke="#888",
             dash="5 3", marker="arrow")
    top.text(20, -14.5, "0-й порядок", size=8.5, color="#555")
    top.text(20, -17.3, "светоловушка", size=8.5, color="#555")
    for x in (46, 50.5, 55):
        top.line(x, -16, x, 10, width=1.5, stroke="#444")

    ray_colors = {340.0: "#7b2fbe", 546.07: "#0a0", 1050.0: "#b00"}
    for lam, col in ray_colors.items():
        th = math.radians(O.diffraction_angle_deg(lam, CFG))
        x2 = GRAT[0] + 40.0 * math.cos(th)
        y2 = GRAT[1] + 40.0 * math.sin(th)
        top.line(GRAT[0], GRAT[1], x2, y2, width=1.2, stroke=col, marker="arrow")
        dy = -2.8 if lam < 500 else 2.4
        anchor = "start" if lam >= 500 else "start"
        top.text(x2 + 1.2, y2 + dy, f"{lam:.0f} нм\nθ={ru(math.degrees(th))}°",
                 size=8, color=col, anchor=anchor)

    # ось камеры и угол
    top.line(GRAT[0], GRAT[1], LENS_C[0], LENS_C[1], width=0.8, stroke="#555", dash="7 3")
    top.items.append(
        f'<path d="M {top.X(GRAT[0] + 12):.2f} {top.Y(GRAT[1]):.2f} '
        f'A {12 * SCALE:.2f} {12 * SCALE:.2f} 0 0 0 '
        f'{top.X(GRAT[0] + 12 * math.cos(A)):.2f} '
        f'{top.Y(GRAT[1] + 12 * math.sin(A)):.2f}" '
        f'fill="none" stroke="#555" stroke-width="0.9"/>')
    top.text(GRAT[0] + 4.0, GRAT[1] - 4.8, "25,5°", size=9, color="#555")
    top.text(GRAT[0] + 3.0, GRAT[1] - 8.2, "ось камеры", size=8, color="#777777")

    # кронштейн: основание (штриховой контур)
    base_pts = [
        (-9.5 + 18 * N[0] - 20 * U[0], -3 + 18 * N[1] - 20 * U[1]),
        (-9.5 + 50 * N[0] - 20 * U[0], -3 + 50 * N[1] - 20 * U[1]),
        (-9.5 + 50 * N[0] + 20 * U[0], -3 + 50 * N[1] + 20 * U[1]),
        (-9.5 + 18 * N[0] + 20 * U[0], -3 + 18 * N[1] + 20 * U[1]),
    ]
    top.polygon(base_pts, stroke="#999999", width=0.7, dash="2 2")

    # размеры
    top.dim_h(-60, 60, 47, "120")
    top.dim_v(-67, -40, 40, "80", left=False)
    top.dim_h(SLIT[0], COLL[0], -45.5, "40 (f коллиматора)")
    top.dim_h(COLL[0], GRAT[0], -52, "8,5")
    top.dim_along(GRAT[0], GRAT[1], LENS_C[0], LENS_C[1], "32", offset=-9.0)

    # =====================================================================
    #  ВИД СБОКУ
    # =====================================================================
    side.rect(-60, 0, 120, 43, width=1.6, rx=5)
    side.rect(-60, 43, 120, 3, width=1.6)
    side.rect(-58, 2, 116, 38, width=0.8, dash="4 3")
    side.text(-56, -8.5, "Вид сбоку (разрез по оптической оси)", size=9, color="#777777")

    side.line(-60, LAY.axis_z_mm, 60, LAY.axis_z_mm, width=0.8, stroke="#777",
              dash="8 3 1 3")

    # кассета щели
    side.rect(LAY.slit_x_mm, 6, 4, 34, width=1.0)
    side.line(LAY.slit_x_mm, LAY.axis_z_mm - CFG.slit_height_mm / 2,
              LAY.slit_x_mm, LAY.axis_z_mm + CFG.slit_height_mm / 2,
              width=2.4, stroke="#c00")
    side.dim_v(-50, LAY.axis_z_mm - 4, LAY.axis_z_mm + 4, "8", tick=2.0)
    side.text(-52, 36.5, "кассета\nщели", size=8, color="#333333")

    # коллиматор
    side.rect(-22.5, 6, 7.8, 34, width=1.0)
    side.circle(LAY.collimator_x_mm, LAY.axis_z_mm, 12.7, width=1.2)
    side.line(-22.5, LAY.axis_z_mm - 2, -22.5, LAY.axis_z_mm + 2, width=2.0,
              stroke="#066")
    side.leader(-18, LAY.axis_z_mm + 11, -52, 48.5,
                "коллиматор: линза Ø25,4 f=40,\nдиафрагма 4,0 мм", size=8)

    # решётка
    side.rect(-12.4, 6, 5.8, 34, width=1.0, fill="#eeeeee")
    side.leader(-9.5, 40, 2, 48.5, "решётка 600 штр/мм", size=8)

    # камера
    side.circle(LENS_C[0], LAY.axis_z_mm, 7.0, width=1.4)
    side.rect(PLATE_C[0] - 1.8, 6, 3.6, 34, width=1.0, fill="#eeeeee")
    side.line(PLATE_C[0] - 1.2, LAY.axis_z_mm, PLATE_C[0] + 1.2, LAY.axis_z_mm,
              width=4.0, stroke="#06c")
    side.text(1.5, 34.0, "объектив f=6", size=8)
    side.rect(MODULE_END[0] - 2.0, 7, 2.0, 32, width=0.9, dash="3 2")
    side.text(27, 5.0, "плата 32×32", size=8)
    side.text(27, 2.2, "+ USB", size=8)

    # кронштейн и основание
    side.rect(-1.9, 8, 46.1, 3, width=0.8, fill="#f4f4f4")
    side.text(30, 10.2, "кронштейн", size=7, color="#777777")

    # светоловушка
    for x in (46, 50.5, 55):
        side.line(x, 12, x, 32, width=1.8, stroke="#444")
    side.text(48, 36.5, "светоловушка\n0-го порядка", size=8, anchor="middle")

    # штативное гнездо
    side.rect(12 - 6.1, 0, 12.2, 6, width=1.0)
    for k in range(4):
        side.line(12 - 5.2, 1.2 + k * 1.2, 12 + 5.2, 1.2 + k * 1.2,
                  width=0.6, stroke="#555")
    side.text(12, -4.8, "1/4\"-20", size=8, anchor="middle")

    # магниты крышки
    for x in (-49, 49):
        side.rect(x - 3.15, 40.9, 6.3, 2.1, width=0.8, fill="#333333")
        side.rect(x - 3.15, 43.0, 6.3, 2.1, width=0.8, fill="#333333")
    side.text(-38, 44.0, "магниты", size=7, color="#555555")

    # размеры сбоку
    side.dim_v(64, 0, 46, "46", left=False)
    side.dim_v(70, 0, 43, "43", left=False)
    side.dim_v(-64, 0, LAY.axis_z_mm, "23")
    side.text(0, -13, "Крышка 3 мм; габарит 120 × 80 × 46 мм; оптическая ось на высоте 23 мм.",
              size=9, anchor="middle")

    # =====================================================================
    #  РАМКА, ШТАМП, ПРИМЕЧАНИЯ
    # =====================================================================
    bg.items.append(f'<rect x="8" y="8" width="{W - 16}" height="{H - 16}" '
                    f'fill="none" stroke="#111" stroke-width="2.2"/>')
    bg.items.append(f'<rect x="14" y="14" width="{W - 28}" height="{H - 28}" '
                    f'fill="none" stroke="#111" stroke-width="0.8"/>')

    # штамп
    bx, by, bw, bh = 700, 700, 452, 160
    bg.items.append(f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" '
                    f'fill="none" stroke="#111" stroke-width="1.6"/>')
    bg.items.append(f'<line x1="{bx}" y1="{by + 44}" x2="{bx + bw}" y2="{by + 44}" '
                    f'stroke="#111" stroke-width="1.2"/>')
    bg.px_text(bx + 12, by + 20, "СПЕКТРОЛАБ SL-H1", 17, weight="bold")
    bg.px_text(bx + 12, by + 39, "Учебный USB-спектрометр 340–1050 нм", 11)
    bg.px_text(bx + bw - 12, by + 20, "Габаритный чертёж", 12, anchor="end")
    rows = [
        ("Масштаб", "1 : 1,5", "Лист", "1 / 1"),
        ("Размеры", "мм", "Материал", "PETG / стекло BK7"),
        ("Разраб.", "hardware/optics.py", "Дата", "2026-11-22"),
    ]
    for i, (k1, v1, k2, v2) in enumerate(rows):
        y = by + 62 + i * 26
        bg.items.append(f'<line x1="{bx}" y1="{y + 10}" x2="{bx + bw}" y2="{y + 10}" '
                        f'stroke="#111" stroke-width="0.5"/>')
        bg.px_text(bx + 12, y, k1, 10, color="#555")
        bg.px_text(bx + 92, y, v1, 11)
        bg.px_text(bx + 270, y, k2, 10, color="#555")
        bg.px_text(bx + 340, y, v2, 11)

    # примечания
    notes = [
        "Технические требования",
        f"1. Оптика: щель {ru(CFG.slit_width_mm, 2)}×{CFG.slit_height_mm:.0f}; коллиматор f={CFG.collimator_focal_mm:.0f} мм,",
        f"   диафрагма Ø{ru(CFG.collimator_stop_mm)}; решётка 600 штр/мм; объектив M12 f={ru(CFG.camera_focal_mm)},",
        f"   ось {ru(CFG.camera_tilt_deg)}° к нормали решётки (оптимум 25,4°).",
        "2. Сенсор 1920×1080, пиксель 2,8 мкм (1/2,9\"), без ИК-фильтра.",
        f"3. Диапазон 340–1050 нм: пиксели {O.pixel_of_wavelength(340, CFG):.0f}…{O.pixel_of_wavelength(1050, CFG):.0f};",
        f"   шаг {ru(O.nm_per_pixel(1050, CFG), 2)}…{ru(O.nm_per_pixel(340, CFG), 2)} нм/px; запас до краёв 436 px.",
        f"4. 0-й порядок вне сенсора (x={ru(O.zero_order_position_mm(CFG), 2)} мм) — светоловушка.",
        "5. Допуски печати ±0,2 мм; посадки щели, решётки, оправы — 0,2 мм.",
        "6. Калибровка по линиям Hg 404,66 / 435,83 / 546,07 / 576,96 /",
        "   579,07 / 611,60 нм; полином 3-й степени; RMS < 1 нм.",
        "7. Контроль: FWHM 546,07 ≤ 2,2 нм; двойник 577/579 различим.",
    ]
    for i, s in enumerate(notes):
        bg.px_text(700, 110 + i * 19, s, 10.5, weight="bold" if i == 0 else "normal")

    # легенда
    ly = 372
    bg.px_text(700, ly, "Ход лучей", 12, weight="bold")
    for i, (lam, col) in enumerate([(340.0, "#7b2fbe"), (546.07, "#0a0"), (1050.0, "#b00")]):
        y = ly + 24 + i * 22
        bg.items.append(f'<line x1="704" y1="{y - 4}" x2="744" y2="{y - 4}" '
                        f'stroke="{col}" stroke-width="2.4"/>')
        bg.px_text(752, y, f"λ = {ru(lam, 2)} нм, θ = {ru(O.diffraction_angle_deg(lam, CFG), 2)}°", 11)
    bg.px_text(704, ly + 100, "— — ось камеры; штрихпунктир — оптическая ось", 10, color="#555")
    bg.px_text(704, ly + 118, "0-й порядок проходит ниже оправы объектива", 10, color="#555")
    bg.px_text(704, ly + 136, "Сенсор: 1040 рабочих пикселей, 0,68 нм/px", 10, color="#555")

    defs = """
<defs>
  <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7"
          markerHeight="7" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="#111111"/>
  </marker>
  <pattern id="hatch" patternUnits="userSpaceOnUse" width="5" height="5"
           patternTransform="rotate(45)">
    <line x1="0" y1="0" x2="0" y2="5" stroke="#666666" stroke-width="0.8"/>
  </pattern>
</defs>"""

    header = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
              f'viewBox="0 0 {W} {H}" font-family="Arial, Helvetica, sans-serif">')
    title = ('<rect width="100%" height="100%" fill="#ffffff"/>'
             f'<text x="{W/2:.0f}" y="40" font-size="20" text-anchor="middle" '
             'font-weight="bold">СПЕКТРОЛАБ — габаритный чертёж оптического тракта</text>'
             f'<text x="{W/2:.0f}" y="60" font-size="12" text-anchor="middle" '
             'fill="#555">Вид сверху с ходами лучей и вид сбоку; размеры в миллиметрах</text>')

    parts = [header, defs, title,
             "<g>", *top.items, "</g>",
             "<g>", *side.items, "</g>",
             "<g>", *bg.items, "</g>",
             "</svg>"]
    return "\n".join(parts)


def main() -> int:
    out = HERE / "drawing.svg"
    svg = build()
    out.write_text(svg, encoding="utf-8")
    print(f"записано: {out} ({len(svg)} байт)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
