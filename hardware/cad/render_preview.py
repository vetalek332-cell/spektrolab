# -*- coding: utf-8 -*-
"""Рендер превью 3D-модели «СпектроЛаб» (z-buffer, без внешних 3D-библиотек).

Собирает меши из :mod:`build_stl` в пониженном разрешении, проецирует
ортографически и рисует треугольники с плоским затенением через Pillow.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_stl as B  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "3d"
PREVIEW_STEPS = {
    "body": 1.4,
    "lid": 1.3,
    "slit": 1.0,
    "grating": 1.0,
    "lens": 0.9,
    "camera": 0.9,
}
COLORS = {
    "body": (206, 214, 226),
    "lid": (176, 187, 202),
    "slit": (255, 183, 3),
    "grating": (76, 201, 240),
    "lens": (150, 228, 255),
    "camera": (123, 135, 151),
}
BACKGROUND = (11, 18, 32)
CAPTION = (206, 215, 226)
FONT_PATH = "C:/Windows/Fonts/segoeui.ttf"


def camera_transform(mesh: np.ndarray) -> np.ndarray:
    angle = math.radians(B.CAM_ANGLE)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    result = mesh.copy()
    x, y = mesh[..., 0], mesh[..., 1]
    result[..., 0] = cos_a * x - sin_a * y + B.GRAT_X
    result[..., 1] = sin_a * x + cos_a * y + B.AXIS_Y
    return result


def load_scene(include_lid: bool) -> list[tuple[str, np.ndarray]]:
    scene: list[tuple[str, np.ndarray]] = []
    for name, step in PREVIEW_STEPS.items():
        if name == "lid" and not include_lid:
            continue
        mesh = B.build_part(name, step)
        if name == "camera":
            mesh = camera_transform(mesh)
        if name == "lid":
            mesh = mesh.copy()
            mesh[..., 2] += 0.12  # чтобы крышка не совпадала с верхом стенок
        scene.append((name, mesh))
    return scene


def rotation_matrix(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    rz = np.array(
        [[math.cos(yaw), -math.sin(yaw), 0.0],
         [math.sin(yaw), math.cos(yaw), 0.0],
         [0.0, 0.0, 1.0]]
    )
    rx = np.array(
        [[1.0, 0.0, 0.0],
         [0.0, math.cos(pitch), -math.sin(pitch)],
         [0.0, math.sin(pitch), math.cos(pitch)]]
    )
    return rx @ rz


def render_panel(
    scene: list[tuple[str, np.ndarray]],
    width: int,
    height: int,
    yaw: float,
    pitch: float,
    supersample: int = 2,
) -> Image.Image:
    scale = supersample
    canvas_w, canvas_h = width * scale, height * scale
    color_buffer = np.empty((canvas_h, canvas_w, 3), dtype=np.float32)
    color_buffer[:] = BACKGROUND
    depth_buffer = np.full((canvas_h, canvas_w), -np.inf, dtype=np.float32)

    rotation = rotation_matrix(yaw, pitch)
    light = np.array([-0.45, 0.55, 0.75])
    light /= np.linalg.norm(light)
    fill = np.array([0.6, -0.35, 0.4])
    fill /= np.linalg.norm(fill)

    prepared: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    all_points = []
    for name, mesh in scene:
        points = mesh.reshape(-1, 3) @ rotation.T
        all_points.append(points[:, :2])
        triangles = points.reshape(-1, 3, 3)
        normals = np.cross(triangles[:, 1] - triangles[:, 0],
                           triangles[:, 2] - triangles[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1), 1e-12)[:, None]
        prepared.append((triangles, normals, np.array(COLORS[name], dtype=float)))

    xy = np.concatenate(all_points)
    min_xy = xy.min(axis=0)
    max_xy = xy.max(axis=0)
    span = float(max(max_xy - min_xy))
    margin = 14
    scale_px = (min(canvas_w, canvas_h) - 2 * margin) / span
    offset = np.array([canvas_w, canvas_h]) * 0.5 - (min_xy + max_xy) * 0.5 * scale_px

    for triangles, normals, base_color in prepared:
        visible = normals[:, 2] > 0.02
        key = np.clip(normals @ light, 0.0, 1.0)
        back = np.clip(normals @ fill, 0.0, 1.0)
        shade = 0.30 + 0.55 * key + 0.18 * back
        colors = np.clip(base_color[None, :] * shade[:, None], 0, 255).astype(np.int32)
        for index in np.flatnonzero(visible):
            tri = triangles[index]
            screen = np.empty((3, 2), dtype=np.float64)
            screen[:, 0] = tri[:, 0] * scale_px + offset[0]
            screen[:, 1] = canvas_h - (tri[:, 1] * scale_px + offset[1])

            x0 = max(int(np.floor(screen[:, 0].min())), 0)
            x1 = min(int(np.ceil(screen[:, 0].max())), canvas_w - 1)
            y0 = max(int(np.floor(screen[:, 1].min())), 0)
            y1 = min(int(np.ceil(screen[:, 1].max())), canvas_h - 1)
            if x1 < x0 or y1 < y0:
                continue

            ax, ay = screen[0]
            bx, by = screen[1]
            cx, cy = screen[2]
            denominator = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
            if abs(denominator) < 1e-9:
                continue
            xs = np.arange(x0, x1 + 1) + 0.5
            ys = np.arange(y0, y1 + 1) + 0.5
            gx, gy = np.meshgrid(xs, ys)
            w0 = ((by - cy) * (gx - cx) + (cx - bx) * (gy - cy)) / denominator
            w1 = ((cy - ay) * (gx - cx) + (ax - cx) * (gy - cy)) / denominator
            w2 = 1.0 - w0 - w1
            inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
            if not np.any(inside):
                continue
            depth = w0 * tri[0, 2] + w1 * tri[1, 2] + w2 * tri[2, 2]
            patch = depth_buffer[y0 : y1 + 1, x0 : x1 + 1]
            update = inside & (depth > patch)
            if not np.any(update):
                continue
            patch[update] = depth[update]
            color = colors[index].astype(np.float32)
            region = color_buffer[y0 : y1 + 1, x0 : x1 + 1]
            region[update] = color

    image = Image.fromarray(np.clip(color_buffer, 0, 255).astype(np.uint8))
    return image.resize((width, height), Image.LANCZOS)


def render_parts_grid() -> Image.Image:
    """Сетка 3×2 с отдельными печатными деталями."""
    cell_w, cell_h = 470, 380
    gap = 14
    cols, rows = 3, 2
    names = ["body", "lid", "slit", "grating", "lens", "camera"]
    labels = {
        "body": "Корпус",
        "lid": "Крышка",
        "slit": "Кассета щели",
        "grating": "Рамка решётки",
        "lens": "Оправа коллиматора",
        "camera": "Кронштейн камеры",
    }
    canvas = Image.new(
        "RGB",
        (cols * cell_w + gap * (cols + 1), rows * cell_h + gap * (rows + 1)),
        BACKGROUND,
    )
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype(FONT_PATH, 20)
    except OSError:
        font = ImageFont.load_default()

    for index, name in enumerate(names):
        row, col = divmod(index, cols)
        scene = [(name, B.build_part(name, PREVIEW_STEPS[name]))]
        panel = render_panel(scene, cell_w, cell_h, yaw=-32, pitch=-28)
        x = gap + col * (cell_w + gap)
        y = gap + row * (cell_h + gap)
        canvas.paste(panel, (x, y))
        draw.text((x + 6, y + 6), labels[name], fill=CAPTION, font=font)
    return canvas


def main() -> int:
    width, height = 900, 720
    assembly = render_panel(load_scene(include_lid=True), width, height, yaw=-38, pitch=-30)
    internals = render_panel(load_scene(include_lid=False), width, height, yaw=-38, pitch=-30)

    gap = 20
    canvas = Image.new("RGB", (width * 2 + gap * 3, height + gap * 2 + 46), BACKGROUND)
    canvas.paste(assembly, (gap, gap))
    canvas.paste(internals, (width + gap * 2, gap))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype(FONT_PATH, 24)
        small = ImageFont.truetype(FONT_PATH, 17)
    except OSError:
        font = small = ImageFont.load_default()

    draw.text((gap + 8, height + gap + 6), "Сборка: корпус + крышка", fill=CAPTION, font=font)
    draw.text((width + gap * 2 + 8, height + gap + 6),
              "Без крышки: щель, коллиматор, решётка, камера", fill=CAPTION, font=font)
    draw.text((gap + 8, height + gap + 32),
              "СпектроЛаб · USB-спектрометр 340–1050 нм · STL: hardware/stl",
              fill=(120, 136, 156), font=small)

    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "preview.png"
    canvas.save(target, optimize=True)
    print("Сохранено:", target, canvas.size)

    parts = render_parts_grid()
    parts_target = OUT / "parts.png"
    parts.save(parts_target, optimize=True)
    print("Сохранено:", parts_target, parts.size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
