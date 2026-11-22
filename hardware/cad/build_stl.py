# -*- coding: utf-8 -*-
"""Генератор печатных STL-деталей «СпектроЛаб» без OpenSCAD.

Геометрия повторяет параметрическую модель ``spektrolab_usb.scad``:
те же размеры корпуса, оптические оси, посадочные места и крепёж.
Метод: SDF (signed distance field) на сетке + marching tetrahedra.
Зависимости: только NumPy (Pillow — для превью, опционально).

Примеры::

    python hardware/cad/build_stl.py                    # все детали + сборка
    python hardware/cad/build_stl.py --parts body lid   # только две детали
    python hardware/cad/build_stl.py --step 1.2        # грубее и быстрее
"""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

MM = 1.0
OUT_DIR = Path(__file__).resolve().parent.parent / "stl"

# ---------------------------------------------------------------------------
# Параметры изделия (мм) — синхронизированы с spektrolab_usb.scad
# ---------------------------------------------------------------------------
BODY_L, BODY_W, BODY_H = 120.0, 80.0, 43.0
LID_T = 3.0
WALL, FLOOR_T = 2.0, 2.0
CORNER_R, CAVITY_R, TOL = 5.0, 3.0, 0.2
CAVITY_TOP = BODY_H - LID_T  # 40
INNER_HX = BODY_L / 2 - WALL  # 58
INNER_HY = BODY_W / 2 - WALL  # 38

AXIS_Y, AXIS_Z = -3.0, 23.0
SLIT_X, COLL_X, GRAT_X = -58.0, -18.0, -9.5
CAM_DIST, CAM_PLATE_DIST, CAM_ANGLE = 32.0, 38.0, 25.5
SLIT_W, SLIT_H = 0.03, 8.0
LENS_D, LENS_TH, STOP_D = 25.4, 5.0, 4.0
GRAT_SIZE, GRAT_TH = 25.4, 3.4
AP_W, AP_H = 14.0, 12.0
CAM_BOARD, CAM_HOLE_PITCH, CAM_BARREL_D = 32.0, 28.0, 14.3

M3_CLEAR, M3_INSERT_D, M3_INSERT_H = 3.4, 4.2, 5.2
M2_CLEAR, M2_INSERT_D, M2_INSERT_H = 2.4, 3.6, 4.2
MAGNET_D, MAGNET_H = 6.3, 2.1
TRIPOD_D, TRIPOD_H = 12.2, 6.0

MAGNET_POINTS = [(-49.0, 34.0), (-49.0, -34.0), (49.0, 34.0), (49.0, -34.0)]
CAM_BASE_HOLES = [(22.0, -15.0), (22.0, 15.0), (44.0, 0.0)]

# ---------------------------------------------------------------------------
# SDF-примитивы
# ---------------------------------------------------------------------------


def _rounded_box(x, y, z, cx, cy, cz, sx, sy, sz, r=0.0):
    """SDF призмы со скруглёнными вертикальными рёбрами (как OpenSCAD rounded_box).

    Скругление задаётся в плоскости XY, торцы по Z плоские. Радиус r не должен
    превышать половину меньшей стороны в XY (для деталей толщиной 3 мм это
    важно: r=5 относится к контуру корпуса, а не к толщине крышки).
    """
    hx, hy = sx * 0.5 - r, sy * 0.5 - r
    qx = np.abs(x - cx) - hx
    qy = np.abs(y - cy) - hy
    rounded_2d = (
        np.sqrt(np.maximum(qx, 0.0) ** 2 + np.maximum(qy, 0.0) ** 2)
        + np.minimum(np.maximum(qx, qy), 0.0)
        - r
    )
    dz = np.abs(z - cz) - sz * 0.5
    outside = np.hypot(np.maximum(rounded_2d, 0.0), np.maximum(dz, 0.0))
    return np.minimum(np.maximum(rounded_2d, dz), 0.0) + outside


def box(x, y, z, x0, x1, y0, y1, z0, z1, r=0.0):
    return _rounded_box(
        x, y, z,
        (x0 + x1) * 0.5, (y0 + y1) * 0.5, (z0 + z1) * 0.5,
        abs(x1 - x0), abs(y1 - y0), abs(z1 - z0), r,
    )


def cylinder(x, y, z, axis: str, cx, cy, cz, radius, height):
    """SDF цилиндра вдоль оси 'x', 'y' или 'z'."""
    if axis == "x":
        dr = np.hypot(y - cy, z - cz) - radius
        da = np.abs(x - cx) - height * 0.5
    elif axis == "y":
        dr = np.hypot(x - cx, z - cz) - radius
        da = np.abs(y - cy) - height * 0.5
    elif axis == "z":
        dr = np.hypot(x - cx, y - cy) - radius
        da = np.abs(z - cz) - height * 0.5
    else:
        raise ValueError(axis)
    return np.minimum(np.maximum(dr, da), 0.0) + np.hypot(
        np.maximum(dr, 0.0), np.maximum(da, 0.0)
    )


def _rot_z(x, y, angle_deg, cx, cy):
    a = math.radians(angle_deg)
    dx, dy = x - cx, y - cy
    return cx + math.cos(a) * dx + math.sin(a) * dy, cy - math.sin(a) * dx + math.cos(a) * dy


def _rot_y(x, z, angle_deg, cx, cz):
    a = math.radians(angle_deg)
    dx, dz = x - cx, z - cz
    return cx + math.cos(a) * dx + math.sin(a) * dz, cz - math.sin(a) * dx + math.cos(a) * dz


def _rot_x(y, z, angle_deg, cy, cz):
    a = math.radians(angle_deg)
    dy, dz = y - cy, z - cz
    return cy + math.cos(a) * dy + math.sin(a) * dz, cz - math.sin(a) * dy + math.cos(a) * dz


def union(*fields):
    result = fields[0]
    for field in fields[1:]:
        result = np.minimum(result, field)
    return result


def difference(base, *cutters):
    result = base
    for cutter in cutters:
        result = np.maximum(result, -cutter)
    return result


def _camera_to_global(lx: float, ly: float) -> tuple[float, float]:
    """Локальная система кронштейна камеры → координаты сборки."""
    a = math.radians(CAM_ANGLE)
    gx = GRAT_X + math.cos(a) * lx - math.sin(a) * ly
    gy = AXIS_Y + math.sin(a) * lx + math.cos(a) * ly
    return gx, gy


# ---------------------------------------------------------------------------
# Детали
# ---------------------------------------------------------------------------


def field_body(x, y, z):
    shell = _rounded_box(x, y, z, 0, 0, BODY_H / 2, BODY_L, BODY_W, BODY_H, CORNER_R)
    cavity_height = BODY_H - FLOOR_T
    cavity = _rounded_box(
        x, y, z, 0, 0, FLOOR_T + cavity_height / 2,
        2 * INNER_HX, 2 * INNER_HY, cavity_height, CAVITY_R,
    )
    features = [
        # направляющие кассеты щели
        box(x, y, z, SLIT_X - 0.5, SLIT_X + 4, -22.2, -20.2, 1.5, 40),
        box(x, y, z, SLIT_X - 0.5, SLIT_X + 4, 14.2, 16.2, 1.5, 40),
        box(x, y, z, SLIT_X - 0.5, SLIT_X + 4, -22.2, 16.2, 1.5, 4),
        # светозащитная диафрагма
        box(x, y, z, -52.5, -51.0, -INNER_HY - 0.5, INNER_HY + 0.5, 1.5, CAVITY_TOP),
        # направляющие оправы коллиматора
        box(x, y, z, -22.5, -14.5, -25.5, -23.5, 1.5, 40),
        box(x, y, z, -22.5, -14.5, 17.5, 19.5, 1.5, 40),
        box(x, y, z, -22.5, -14.5, -25.5, 19.5, 1.5, 4),
        # рёбра канала решётки
        box(x, y, z, -14.0, -12.5, -INNER_HY - 0.5, INNER_HY + 0.5, 1.5, CAVITY_TOP),
        box(x, y, z, -6.4, -4.9, -INNER_HY - 0.5, INNER_HY + 0.5, 1.5, CAVITY_TOP),
        # прилив штативного гнезда
        cylinder(x, y, z, "z", 12, 0, FLOOR_T + 2.5, 8.0, 7.0),
    ]

    # приливы крепления кронштейна камеры
    for lx, ly in CAM_BASE_HOLES:
        gx, gy = _camera_to_global(lx, ly)
        features.append(cylinder(x, y, z, "z", gx, gy, FLOOR_T + 2.5, 4.0, 7.0))

    # приливы магнитов крышки
    for mx, my in MAGNET_POINTS:
        features.append(box(x, y, z, mx - 5, mx + 5, min(my - 4, -INNER_HY - 0.5) if my < 0 else my - 4,
            max(my + 4, INNER_HY + 0.5) if my > 0 else my + 4, 35.5, BODY_H))

    # светоловушка 0-го порядка (наклонные рёбра)
    for cx, angle in ((46.0, 25.0), (50.5, -25.0), (55.0, 25.0)):
        lx, lz = _rot_y(x, z, angle, cx, 22.0)
        features.append(box(lx, y, lz, -1.0, 1.0, -16.0, 10.0, -21.0, 10.0))

    # сначала выбираем полость, затем ДОБАВЛЯЕМ внутренние элементы (как в SCAD)
    part = union(difference(shell, cavity), *features)

    cutters = [
        # окно щели
        box(x, y, z, -61.0, -57.8, -9.2, 3.2, 17.8, 28.2),
        # апертура диафрагмы
        box(x, y, z, -53.0, -50.5, -6.0, 0.0, 19.0, 27.0),
        # окна в рёбрах канала решётки
        box(x, y, z, -15.0, -5.0, -10.0, 4.0, 17.0, 29.0),
        # штативное гнездо
        cylinder(x, y, z, "z", 12, 0, FLOOR_T + 3 - 1.0, TRIPOD_D / 2, TRIPOD_H + 2),
        # кабельный ввод
        cylinder(x, y, z, "x", 60.0, 17.0, AXIS_Z, 4.5, 6.0),
    ]
    # отверстия под термозайки M3 в приливах камеры
    for lx, ly in CAM_BASE_HOLES:
        gx, gy = _camera_to_global(lx, ly)
        cutters.append(
            cylinder(x, y, z, "z", gx, gy, 8.0 - M3_INSERT_H / 2, M3_INSERT_D / 2, M3_INSERT_H)
        )
    # гнёзда магнитов
    for mx, my in MAGNET_POINTS:
        cutters.append(
            cylinder(x, y, z, "z", mx, my, BODY_H - MAGNET_H / 2, MAGNET_D / 2, MAGNET_H + 0.2)
        )
    # пазы под стяжку у кабельного ввода
    cutters.append(box(x, y, z, 59.0, 63.0, 12.5, 14.5, 20.0, 26.0))
    cutters.append(box(x, y, z, 59.0, 63.0, 21.5, 23.5, 20.0, 26.0))

    return difference(part, *cutters)


def field_lid(x, y, z):
    base = _rounded_box(x, y, z, 0, 0, BODY_H + LID_T / 2, BODY_L, BODY_W, LID_T, CORNER_R)
    ribs = union(
        box(x, y, z, -40, 40, -INNER_HY + 6, -INNER_HY + 8, BODY_H - 1.5, BODY_H + 0.3),
        box(x, y, z, -40, 40, INNER_HY - 8, INNER_HY - 6, BODY_H - 1.5, BODY_H + 0.3),
    )
    part = union(base, ribs)
    cutters = [
        cylinder(x, y, z, "z", mx, my, BODY_H + MAGNET_H / 2, MAGNET_D / 2 + 0.05,
                 MAGNET_H + 0.2)
        for mx, my in MAGNET_POINTS
    ]
    return difference(part, *cutters)


def field_slit(x, y, z):
    body = box(x, y, z, SLIT_X, SLIT_X + 4, -20.0, 14.0, 6.0, 40.0)
    cutters = [
        box(x, y, z, SLIT_X - 1.0, SLIT_X + 5.0, -8.0, 2.0, 18.0, 28.0),
        box(x, y, z, SLIT_X + 1.0, SLIT_X + 3.0, -6.0, 0.0, 38.5, 40.0),
    ]
    return difference(body, *cutters)


def field_grating(x, y, z):
    body = box(x, y, z, GRAT_X - 2.8, GRAT_X + 2.8, -20.0, 14.0, 6.0, 40.0)
    lip = box(x, y, z, GRAT_X - 2.8, GRAT_X + 2.8, -13.0, 7.0, 39.0, 41.0)
    part = union(body, lip)
    cutters = [
        box(x, y, z,
            GRAT_X - GRAT_TH / 2, GRAT_X + GRAT_TH / 2,
            -3 - GRAT_SIZE / 2, -3 + GRAT_SIZE / 2,
            AXIS_Z - GRAT_SIZE / 2, AXIS_Z + GRAT_SIZE / 2),
        box(x, y, z, GRAT_X - 4.0, GRAT_X + 4.0, -3 - AP_W / 2, -3 + AP_W / 2,
            AXIS_Z - AP_H / 2, AXIS_Z + AP_H / 2),
        box(x, y, z, GRAT_X - 4.0, GRAT_X + 4.0, -6.0, 0.0, 37.5, 40.0),
    ]
    return difference(part, *cutters)


def field_lens(x, y, z):
    px, t = -22.5, 7.8
    plate = box(x, y, z, px, px + t, -23.3, 17.3, 6.0, 40.0)
    ring = cylinder(x, y, z, "x", px + t + 0.35, AXIS_Y, AXIS_Z, 14.0, 1.8)
    ring = difference(ring, cylinder(x, y, z, "x", px + t + 0.35, AXIS_Y, AXIS_Z, 12.5, 2.6))
    # четыре радиальных паза-лепестка
    for angle in (45.0, 135.0, 225.0, 315.0):
        dy = y - AXIS_Y
        dz = z - AXIS_Z
        u = math.cos(math.radians(angle)) * dy + math.sin(math.radians(angle)) * dz
        v = -math.sin(math.radians(angle)) * dy + math.cos(math.radians(angle)) * dz
        slot = np.maximum.reduce(
            [
                np.abs(v) - 0.6,
                np.abs(u) - 15.0,
                (px + t - 0.9) - x,
                x - (px + t + 1.9),
            ]
        )
        ring = np.maximum(ring, -slot)
    part = union(plate, ring)
    cutters = [
        cylinder(x, y, z, "x", px + 2.0 + 2.9, AXIS_Y, AXIS_Z, (LENS_D + TOL) / 2, 5.8),
        cylinder(x, y, z, "x", px - 0.25 + 1.25, AXIS_Y, AXIS_Z, STOP_D / 2, 2.5),
        box(x, y, z, px + 2.0, px + 5.0, -19.0, -13.0, 8.0, 38.0),
        box(x, y, z, px + 2.0, px + 5.0, 7.0, 13.0, 8.0, 38.0),
    ]
    return difference(part, *cutters)


def field_camera(x, y, z):
    """Кронштейн камеры в локальной системе (плоскость платы — X=const)."""
    base = box(x, y, z, 18.0, 50.0, -20.0, 20.0, 8.0, 11.0)
    plate = box(x, y, z, 36.0, 40.0, -17.0, 17.0, 6.0, 40.0)
    part = union(base, plate)

    # бобышки крепления платы M2
    for dy in (-CAM_HOLE_PITCH / 2, CAM_HOLE_PITCH / 2):
        for dz in (-CAM_HOLE_PITCH / 2, CAM_HOLE_PITCH / 2):
            part = union(
                part,
                cylinder(x, y, z, "x", 42.0, dy, AXIS_Z + dz, 2.75, 4.0),
            )

    # рёбра-косынки (треугольные призмы)
    for side in (-1, 1):
        yc = side * 12.0
        gx = np.maximum.reduce([22.0 - x, x - 36.0, 8.0 - z, z - 22.0, z - (x - 14.0)])
        gusset = np.maximum(gx, np.abs(y - yc) - 1.0)
        part = union(part, gusset)

    cutters = [
        # окно под оправу объектива
        cylinder(x, y, z, "x", 40.0, 0.0, AXIS_Z, CAM_BARREL_D / 2, 10.0),
        # отверстия M2 в бобышках
    ]
    for dy in (-CAM_HOLE_PITCH / 2, CAM_HOLE_PITCH / 2):
        for dz in (-CAM_HOLE_PITCH / 2, CAM_HOLE_PITCH / 2):
            cutters.append(
                cylinder(x, y, z, "x", 42.1, dy, AXIS_Z + dz, M2_INSERT_D / 2, M2_INSERT_H)
            )
    # крепление основания M3 с зенковкой
    for bx, by in CAM_BASE_HOLES:
        cutters.append(cylinder(x, y, z, "z", bx, by, 8.5, M3_CLEAR / 2, 7.0))
        cutters.append(cylinder(x, y, z, "z", bx, by, 9.8, 3.25, 2.4))
    return difference(part, *cutters)


# ---------------------------------------------------------------------------
# Генерация поля и извлечение меша
# ---------------------------------------------------------------------------

PARTS: dict[str, dict] = {
    "body": {
        "field": field_body,
        "bbox": ((-62, 62), (-42, 42), (-2, 46)),
        "step": 1.4,
    },
    "lid": {
        "field": field_lid,
        "bbox": ((-62, 62), (-42, 42), (40, 48)),
        "step": 1.3,
    },
    "slit": {
        "field": field_slit,
        "bbox": ((-60, -52), (-22, 16), (4, 42)),
        "step": 1.0,
    },
    "grating": {
        "field": field_grating,
        "bbox": ((-14, -5), (-22, 16), (4, 43)),
        "step": 1.0,
    },
    "lens": {
        "field": field_lens,
        "bbox": ((-24.5, -13.5), (-25, 19), (4, 42)),
        "step": 0.8,
    },
    "camera": {
        "field": field_camera,
        "bbox": ((16, 52), (-22, 22), (4, 42)),
        "step": 1.0,
    },
}

# Тетраэдры куба (индексы углов 0..7)
CUBE_TETS = [
    (0, 1, 2, 6),
    (0, 2, 3, 6),
    (0, 3, 7, 6),
    (0, 7, 4, 6),
    (0, 4, 5, 6),
    (0, 5, 1, 6),
]
# Случаи marching tetrahedra: код (биты «внутри») → список треугольников из рёбер
TET_CASES: dict[int, list[tuple[tuple[int, int], ...]]] = {
    1: [((3, 0), (3, 1), (3, 2))],
    2: [((2, 0), (2, 1), (2, 3))],
    4: [((1, 0), (1, 2), (1, 3))],
    8: [((0, 1), (0, 2), (0, 3))],
    3: [((0, 2), (0, 3), (1, 3)), ((0, 2), (1, 3), (1, 2))],
    5: [((0, 1), (0, 3), (2, 3)), ((0, 1), (2, 3), (2, 1))],
    6: [((1, 0), (1, 3), (2, 3)), ((1, 0), (2, 3), (2, 0))],
    9: [((0, 1), (0, 2), (3, 2)), ((0, 1), (3, 2), (3, 1))],
    10: [((1, 0), (1, 2), (3, 2)), ((1, 0), (3, 2), (3, 0))],
    12: [((2, 0), (2, 1), (3, 1)), ((2, 0), (3, 1), (3, 0))],
    7: [((0, 1), (0, 2), (0, 3))],
    11: [((1, 0), (1, 2), (1, 3))],
    13: [((2, 0), (2, 1), (2, 3))],
    14: [((3, 0), (3, 1), (3, 2))],
}


def _sample(field: Callable, bbox, step: float):
    (x0, x1), (y0, y1), (z0, z1) = bbox
    pad = 2 * step
    # небольшой сдвиг сетки: иначе плоские грани, кратные шагу, дают
    # вырожденные пересечения ровно в узлах (трещины и немногообразие)
    jitter = step * 0.0137
    xs = np.arange(x0 - pad + jitter, x1 + pad + step * 0.5, step)
    ys = np.arange(y0 - pad + jitter, y1 + pad + step * 0.5, step)
    zs = np.arange(z0 - pad + jitter, z1 + pad + step * 0.5, step)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    values = field(X, Y, Z).astype(np.float32)
    return xs, ys, zs, values


def _extract_mesh(values: np.ndarray, xs, ys, zs, step: float) -> np.ndarray:
    """Marching tetrahedra: возвращает (T, 3, 3) координаты треугольников."""
    nx, ny, nz = values.shape
    base = np.stack(
        np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), np.arange(nz - 1), indexing="ij"),
        axis=-1,
    )  # (nx-1, ny-1, nz-1, 3)

    origin = np.array([xs[0], ys[0], zs[0]], dtype=float)
    triangles: list[np.ndarray] = []

    for tet in CUBE_TETS:
        corner_offsets = np.array(
            [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
             (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
            dtype=float,
        )
        tet_offsets = corner_offsets[list(tet)]
        vals = np.stack(
            [values[base[..., 0] + int(o[0]), base[..., 1] + int(o[1]), base[..., 2] + int(o[2])]
             for o in tet_offsets],
            axis=-1,
        )  # (M,4)
        pos = (base[..., None, :] + tet_offsets[None, :, :]) * step + origin  # (M,4,3)
        inside = vals < 0.0
        codes = (
            inside[..., 0].astype(np.int8) * 8
            + inside[..., 1].astype(np.int8) * 4
            + inside[..., 2].astype(np.int8) * 2
            + inside[..., 3].astype(np.int8)
        )

        flat_codes = codes.ravel()
        flat_vals = vals.reshape(-1, 4)
        flat_pos = pos.reshape(-1, 4, 3)

        for code, case_triangles in TET_CASES.items():
            cells = np.flatnonzero(flat_codes == code)
            if cells.size == 0:
                continue
            cell_vals = flat_vals[cells]
            cell_pos = flat_pos[cells]
            for triangle in case_triangles:
                points = []
                for a, b in triangle:
                    va = cell_vals[:, a]
                    vb = cell_vals[:, b]
                    denominator = va - vb
                    denominator = np.where(np.abs(denominator) < 1e-12, 1e-12, denominator)
                    t = np.clip(va / denominator, 0.0, 1.0)
                    pa = cell_pos[:, a]
                    pb = cell_pos[:, b]
                    points.append(pa + t[:, None] * (pb - pa))
                triangles.append(np.stack(points, axis=1))

    if not triangles:
        raise RuntimeError("Меш пуст — проверьте поле SDF")

    return np.concatenate(triangles, axis=0)


def _orient_outward(mesh: np.ndarray) -> np.ndarray:
    """Согласовать ориентацию треугольников и направить нормали наружу.

    Соседние треугольники должны обходить общее ребро в противоположных
    направлениях; после согласования знак суммарного объёма показывает,
    смотрят ли нормали наружу.
    """
    vertices = np.round(mesh.reshape(-1, 3), 4)
    unique, inverse = np.unique(vertices, axis=0, return_inverse=True)
    indices = inverse.reshape(-1, 3)
    coordinates = mesh.reshape(-1, 3)

    edge_map: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for triangle_index, triangle in enumerate(indices):
        for position in range(3):
            a = int(triangle[position])
            b = int(triangle[(position + 1) % 3])
            key = (a, b) if a < b else (b, a)
            edge_map.setdefault(key, []).append((triangle_index, 0 if a < b else 1))

    flip = np.zeros(indices.shape[0], dtype=bool)
    visited = np.zeros(indices.shape[0], dtype=bool)
    for seed in range(indices.shape[0]):
        if visited[seed]:
            continue
        visited[seed] = True
        stack = [seed]
        while stack:
            triangle_index = stack.pop()
            triangle = indices[triangle_index]
            for position in range(3):
                a = int(triangle[position])
                b = int(triangle[(position + 1) % 3])
                key = (a, b) if a < b else (b, a)
                direction = 0 if a < b else 1
                for neighbour, neighbour_direction in edge_map[key]:
                    if neighbour == triangle_index or visited[neighbour]:
                        continue
                    # эффективное направление = исходное, инвертированное флагом
                    same = (direction ^ int(flip[triangle_index])) == (
                        neighbour_direction ^ int(flip[neighbour])
                    )
                    if same:
                        flip[neighbour] = not flip[neighbour]
                    visited[neighbour] = True
                    stack.append(neighbour)

    result = mesh.copy()
    result[flip] = result[flip][:, ::-1]

    signed_volume = float(
        np.sum(np.einsum("ij,ij->i", result[:, 0], np.cross(result[:, 1], result[:, 2]))) / 6.0
    )
    if signed_volume < 0:
        result = result[:, ::-1]
    return result


def clean_mesh(mesh: np.ndarray) -> np.ndarray:
    """Сварить вершины, убрать вырожденные и дублирующиеся треугольники."""
    vertices = np.round(mesh.reshape(-1, 3), 4)
    unique, inverse = np.unique(vertices, axis=0, return_inverse=True)
    indices = inverse.reshape(-1, 3)

    keep = (
        (indices[:, 0] != indices[:, 1])
        & (indices[:, 1] != indices[:, 2])
        & (indices[:, 0] != indices[:, 2])
    )
    indices = indices[keep]

    sorted_indices = np.sort(indices, axis=1)
    _, first = np.unique(sorted_indices, axis=0, return_index=True)
    indices = indices[np.sort(first)]

    return unique[indices]


def fill_holes(mesh: np.ndarray, max_loop: int = 64) -> np.ndarray:
    """Закрыть небольшие дыры: граничные рёбра собираются в циклы и триангулируются."""
    vertices = np.round(mesh.reshape(-1, 3), 4)
    unique, inverse = np.unique(vertices, axis=0, return_inverse=True)
    indices = inverse.reshape(-1, 3).tolist()
    coordinates = [tuple(row) for row in unique]

    from collections import defaultdict

    edge_count: dict[tuple[int, int], int] = defaultdict(int)
    for triangle in indices:
        for position in range(3):
            a = triangle[position]
            b = triangle[(position + 1) % 3]
            edge_count[(min(a, b), max(a, b))] += 1

    boundary = [edge for edge, count in edge_count.items() if count == 1]
    if not boundary:
        return mesh

    adjacency: dict[int, list[int]] = defaultdict(list)
    for a, b in boundary:
        adjacency[a].append(b)
        adjacency[b].append(a)

    visited: set[tuple[int, int]] = set()
    patches: list[list[int]] = []
    for start_a, start_b in boundary:
        key = (min(start_a, start_b), max(start_a, start_b))
        if key in visited:
            continue
        loop = [start_a]
        current, previous = start_b, start_a
        visited.add(key)
        length = 1
        while length <= max_loop:
            loop.append(current)
            neighbours = [n for n in adjacency[current] if n != previous]
            if not neighbours:
                break
            following = neighbours[0]
            edge = (min(current, following), max(current, following))
            if following == start_a:
                loop.append(start_a)
                visited.add(edge)
                break
            if edge in visited:
                break
            visited.add(edge)
            previous, current = current, following
            length += 1
        else:
            continue
        if len(loop) >= 3 and loop[-1] == start_a:
            patches.append(loop[:-1])

    if not patches:
        return mesh

    new_indices = [list(triangle) for triangle in indices]
    new_coordinates = list(coordinates)
    for loop in patches:
        centre = np.mean([new_coordinates[index] for index in loop], axis=0)
        centre_index = len(new_coordinates)
        new_coordinates.append(tuple(centre))
        for position in range(len(loop)):
            a = loop[position]
            b = loop[(position + 1) % len(loop)]
            # ориентация патча: ребро должно идти противоположно существующему
            existing = None
            for triangle in indices:
                for k in range(3):
                    if triangle[k] == a and triangle[(k + 1) % 3] == b:
                        existing = 1
                        break
                    if triangle[k] == b and triangle[(k + 1) % 3] == a:
                        existing = -1
                        break
                if existing:
                    break
            if existing == 1:
                new_indices.append([b, a, centre_index])
            else:
                new_indices.append([a, b, centre_index])

    array = np.array(new_coordinates)
    return array[np.array(new_indices)]


def orient_by_connectivity(mesh: np.ndarray) -> np.ndarray:
    """Согласовать обход треугольников по рёбрам и направить нормали наружу.

    Для замкнутой ориентируемой поверхности соседние треугольники должны
    обходить общее ребро в разные стороны; после BFS-согласования знак
    объёма каждой компоненты показывает, куда смотрят нормали.
    """
    vertices = np.round(mesh.reshape(-1, 3), 4)
    unique, inverse = np.unique(vertices, axis=0, return_inverse=True)
    indices = inverse.reshape(-1, 3)

    edge_map: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for triangle_index, triangle in enumerate(indices):
        for position in range(3):
            a = int(triangle[position])
            b = int(triangle[(position + 1) % 3])
            key = (a, b) if a < b else (b, a)
            edge_map.setdefault(key, []).append((triangle_index, 0 if a < b else 1))

    flip = np.zeros(indices.shape[0], dtype=bool)
    visited = np.zeros(indices.shape[0], dtype=bool)
    component = np.full(indices.shape[0], -1, dtype=int)
    component_id = 0

    for seed in range(indices.shape[0]):
        if visited[seed]:
            continue
        visited[seed] = True
        component[seed] = component_id
        stack = [seed]
        while stack:
            triangle_index = stack.pop()
            triangle = indices[triangle_index]
            for position in range(3):
                a = int(triangle[position])
                b = int(triangle[(position + 1) % 3])
                key = (a, b) if a < b else (b, a)
                direction = 0 if a < b else 1
                for neighbour, neighbour_direction in edge_map[key]:
                    if neighbour == triangle_index or visited[neighbour]:
                        continue
                    same = (direction ^ int(flip[triangle_index])) == (
                        neighbour_direction ^ int(flip[neighbour])
                    )
                    if same:
                        flip[neighbour] = not flip[neighbour]
                    visited[neighbour] = True
                    component[neighbour] = component_id
                    stack.append(neighbour)
        component_id += 1

    result = mesh.copy()
    result[flip] = result[flip][:, ::-1]

    for component_index in range(component_id):
        mask = component == component_index
        if not np.any(mask):
            continue
        part = result[mask]
        signed_volume = float(
            np.sum(np.einsum("ij,ij->i", part[:, 0], np.cross(part[:, 1], part[:, 2]))) / 6.0
        )
        if signed_volume < 0:
            result[mask] = part[:, ::-1]
    return result


def _transform_mesh(mesh: np.ndarray, rotation_z: float, translation) -> np.ndarray:
    angle = math.radians(rotation_z)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    result = mesh.copy()
    x = mesh[..., 0]
    y = mesh[..., 1]
    result[..., 0] = cos_a * x - sin_a * y + translation[0]
    result[..., 1] = sin_a * x + cos_a * y + translation[1]
    result[..., 2] = mesh[..., 2] + translation[2]
    return result


# ---------------------------------------------------------------------------
# STL
# ---------------------------------------------------------------------------


def mesh_report(mesh: np.ndarray) -> dict:
    """Замкнутость, объём и габариты меша."""
    verts = np.round(mesh.reshape(-1, 3), 4)
    unique, inverse = np.unique(verts, axis=0, return_inverse=True)
    idx = inverse.reshape(-1, 3)
    edges: dict[tuple[int, int], int] = {}
    for triangle in idx:
        for a, b in ((triangle[0], triangle[1]), (triangle[1], triangle[2]),
                     (triangle[2], triangle[0])):
            key = (int(min(a, b)), int(max(a, b)))
            edges[key] = edges.get(key, 0) + 1
    counts = {}
    for value in edges.values():
        counts[value] = counts.get(value, 0) + 1
    closed = set(counts) == {2}
    volume = float(
        abs(np.sum(np.einsum("ij,ij->i", mesh[:, 0], np.cross(mesh[:, 1], mesh[:, 2])))) / 6.0
    )
    bbox = mesh.reshape(-1, 3)
    size = bbox.max(axis=0) - bbox.min(axis=0)
    return {
        "triangles": int(mesh.shape[0]),
        "vertices": int(unique.shape[0]),
        "closed": closed,
        "edges": counts,
        "volume_cm3": volume / 1000.0,
        "size_mm": tuple(round(float(v), 1) for v in size),
    }


def write_stl(path: Path, mesh: np.ndarray, name: str) -> None:
    normals = np.cross(mesh[:, 1] - mesh[:, 0], mesh[:, 2] - mesh[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    lengths = np.where(lengths < 1e-12, 1.0, lengths)
    normals = normals / lengths[:, None]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        header = (f"SpektroLab {name}").encode("ascii", "replace")[:80]
        fh.write(header.ljust(80, b" "))
        fh.write(struct.pack("<I", mesh.shape[0]))
        for normal, triangle in zip(normals, mesh):
            fh.write(struct.pack("<12fH", *normal, *triangle.ravel(), 0))


def build_part(name: str, step_override: float | None = None):
    config = PARTS[name]
    step = step_override or config["step"]
    xs, ys, zs, values = _sample(config["field"], config["bbox"], step)
    mesh = _extract_mesh(values, xs, ys, zs, step)
    mesh = clean_mesh(mesh)
    mesh = orient_by_connectivity(mesh)
    for _ in range(3):
        mesh = fill_holes(mesh)
        mesh = clean_mesh(mesh)
    return mesh


def main() -> int:
    parser = argparse.ArgumentParser(description="Генератор STL деталей «СпектроЛаб»")
    parser.add_argument("--parts", nargs="+", default=list(PARTS), choices=list(PARTS))
    parser.add_argument("--step", type=float, default=None, help="шаг сетки, мм (0.6–1.2)")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--assembly", action="store_true", help="собрать общий STL")
    args = parser.parse_args()

    meshes: dict[str, np.ndarray] = {}
    for name in args.parts:
        mesh = build_part(name, args.step)
        meshes[name] = mesh
        target = args.out / f"{name}.stl"
        write_stl(target, mesh, name)
        info = mesh_report(mesh)
        size_kb = target.stat().st_size / 1024
        status = "замкнут" if info["closed"] else "ЕСТЬ ТРЕЩИНЫ"
        print(
            f"{name:8s} {info['triangles']:7d} треуг.  {size_kb:8.0f} КБ  {status:14s}  "
            f"{info['volume_cm3']:6.1f} см3  {info['size_mm']} мм"
        )

    if args.assembly:
        pieces = []
        for name, mesh in meshes.items():
            if name == "camera":
                pieces.append(_transform_mesh(mesh, CAM_ANGLE, (GRAT_X, AXIS_Y, 0.0)))
            elif name == "lid":
                pieces.append(mesh)
            else:
                pieces.append(mesh)
        assembly = np.concatenate(pieces, axis=0)
        target = args.out / "assembly.stl"
        write_stl(target, assembly, "assembly")
        print(f"assembly {assembly.shape[0]:7d} треугольников  "
              f"{target.stat().st_size / 1024:8.0f} КБ  -> {target}")

    print("Готово.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
