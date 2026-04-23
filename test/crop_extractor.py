from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TEST_DIR = Path(__file__).resolve().parent
CROP_DIR = TEST_DIR / "crop_polearms"
DEBUG_ROOT = TEST_DIR / "debug_step2self_5"
OUT_DIR = DEBUG_ROOT / "polearms"
OBJECTS_ONLY_DIR = DEBUG_ROOT / "objects_only"
SEARCH_DEPTH = 30
MIN_STRIPE_RUN = 1
MAX_STRIPE_LEAD_IN = 4
MIN_OCCUPANCY_RUN = 2
MIN_BORDER_OCCUPANCY = 0.70
MAX_OCCUPANCY_LEAD_IN = 16
MIN_BORDER_TAIL_OCCUPANCY = 0.30
MAX_OCCUPANCY_EXTENSION = 4
MAX_OCCUPANCY_GAP = 1
FG_GATE_DILATE_ITERATIONS = 1

#СКЕЙЛИМ КРОП
def resize_long_side_if_needed(img_bgr: np.ndarray):
    h, w = img_bgr.shape[:2]
    long_side = max(h, w)
    if long_side <= 0:
        return img_bgr.copy()

    scale = float(256) / float(long_side)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    if scale < 1:
        return cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    else:
        return cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_CUBIC)

#КАРТА КОНТУРОВ
def _edge_map_scharr_max_channel_auto(img_bgr: np.ndarray) -> np.ndarray:
    img = img_bgr.astype(np.float32)
    img = cv2.GaussianBlur(img, (5, 5), 0)

    dx = cv2.Scharr(img, cv2.CV_32F, 1, 0)
    dy = cv2.Scharr(img, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(dx, dy)
    mag = mag[..., 0] + mag[..., 1] + mag[..., 2]

    nz = mag[mag > 0]
    if nz.size == 0:
        return np.zeros(img_bgr.shape[:2], dtype=np.uint8)

    thr = np.percentile(nz, 66)  # калибровать
    edge = (mag >= thr).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel, iterations=1)

    edge = normalize_mask(edge)
    edge = keep_components_by_area(edge, min_area=5)
    return edge

#бинаризация маски
def normalize_mask(mask: np.ndarray) -> np.ndarray:
    return np.where(mask > 0, 255, 0).astype(np.uint8) #делает маску бинарной

#Оставляет объекты больше трешхолда
def keep_components_by_area(mask: np.ndarray, min_area: int = 24) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8),
        connectivity=8,
    )

    if n <= 1:
        return normalize_mask(mask)

    out = np.zeros_like(mask, dtype=np.uint8)

    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            out[labels == label] = 255

    return normalize_mask(out)


def remove_small_border_components(mask: np.ndarray, max_area: int = 800) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8),
        connectivity=8,
    )

    if n <= 1:
        return normalize_mask(mask)

    h, w = mask.shape
    out = np.zeros_like(mask, dtype=np.uint8)

    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        ww = int(stats[label, cv2.CC_STAT_WIDTH])
        hh = int(stats[label, cv2.CC_STAT_HEIGHT])
        touches_border = (x == 0 or y == 0 or x + ww == w or y + hh == h)

        if touches_border and area <= max_area:
            continue
        out[labels == label] = 255

    return normalize_mask(out)


def remove_weak_secondary_components(mask: np.ndarray, min_ratio_to_largest: float = 0.16) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8),
        connectivity=8,
    )

    if n <= 1:
        return normalize_mask(mask)

    areas = [int(stats[label, cv2.CC_STAT_AREA]) for label in range(1, n)]
    largest = max(areas) if areas else 0
    if largest <= 0:
        return normalize_mask(mask)

    min_keep_area = max(150, int(round(largest * min_ratio_to_largest)))
    out = np.zeros_like(mask, dtype=np.uint8)
    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_keep_area:
            out[labels == label] = 255

    return normalize_mask(out)


def remove_confirmed_background_regions(
    mask: np.ndarray,
    fg_gate: np.ndarray,
    min_bg_area: int = 24,
) -> np.ndarray:
    bg_candidate = np.logical_and(mask > 0, fg_gate == 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bg_candidate, connectivity=8)

    if n <= 1:
        return normalize_mask(mask)

    remove = np.zeros_like(mask, dtype=np.uint8)
    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_bg_area:
            remove[labels == label] = 255

    out = mask.copy()
    out[remove > 0] = 0
    return normalize_mask(out)

#СОЗДАНИЕ МАСКИ
def edge_mask_base(precomputed_edges: np.ndarray) -> np.ndarray:
    # Применяем морфологию (дилатация и закрытие)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = precomputed_edges.copy()
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

    mask = cv2.medianBlur(mask, 3)
    mask = normalize_mask(mask)
    mask = cv2.erode(mask, k, iterations=1)

    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]

    mask = keep_components_by_area(mask, min_area=50)
    mask = normalize_mask(mask)
    return mask
#Заполнение МАСКИ
def fill_internal_holes(mask: np.ndarray) -> np.ndarray:
    fg = (mask > 0).astype(np.uint8)
    bg = (fg == 0).astype(np.uint8)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(bg, connectivity=8)
    if n <= 1:
        return normalize_mask(mask)

    holes = np.zeros_like(mask, dtype=np.uint8)
    h, w = mask.shape

    for label in range(1, n):
        ys, xs = np.where(labels == label)
        if xs.size == 0:
            continue

        touches_border = (
            xs.min() == 0 or xs.max() == w - 1 or
            ys.min() == 0 or ys.max() == h - 1
        )
        if not touches_border:
            holes[labels == label] = 255

    filled = cv2.bitwise_or(mask, holes)
    return normalize_mask(filled)

"--------------------------------------ЧИСТКА РАМКИ--------------------------------------"
#5 точек вдоль длины
def _sample_positions(length: int) -> list[int]:
    raw = [0.15, 0.30, 0.50, 0.70, 0.85]
    out = []
    for r in raw:
        p = int(round((length - 1) * r))
        p = max(0, min(length - 1, p))
        out.append(p)
    return sorted(set(out))

#ищет полосу по стороне, пропускает все черные, пока не найдет белое. глубина - ГЛОБАЛЬНАЯ КОНСТАНТА
def _run_white_from_side(mask: np.ndarray, idx: int, side: str) -> int:
    h, w = mask.shape

    if side in ("left", "right"):
        length = w
    elif side in ("top", "bottom"):
        length = h
    else:
        raise ValueError(f"Unsupported side: {side}")

    limit = min(length, SEARCH_DEPTH)

    def get_value(pos: int) -> int:
        if side == "left":
            return int(mask[idx, pos])
        if side == "right":
            return int(mask[idx, w - 1 - pos])
        if side == "top":
            return int(mask[pos, idx])
        if side == "bottom":
            return int(mask[h - 1 - pos, idx])
        raise ValueError(f"Unsupported side: {side}")

    p = 0
    while p < limit:
        while p < limit and get_value(p) == 0:
            p += 1
        if p >= limit:
            return 0

        start = p
        if start > MAX_STRIPE_LEAD_IN:
            return 0
        while p < limit and get_value(p) > 0:
            p += 1
        run = p - start

        if run >= MIN_STRIPE_RUN:
            return start + run

    return 0

#оценка толщины рамки для поиска по всем сторонам
def _robust_border_width(
    runs: list[int],
    min_nonzero_samples: int = 3,
    min_width: int = 2,
) -> int:
    nz = [int(v) for v in runs if int(v) >= min_width]
    if len(nz) < min_nonzero_samples:
        return 0
    nz.sort()
    return int(round(float(np.median(nz))))


def _occupancy_values_from_side(mask: np.ndarray, side: str) -> list[float]:
    h, w = mask.shape

    if side in ("top", "bottom"):
        length = h
        span = float(w)
    elif side in ("left", "right"):
        length = w
        span = float(h)
    else:
        raise ValueError(f"Unsupported side: {side}")

    limit = min(length, SEARCH_DEPTH)
    occ = []
    for pos in range(limit):
        if side == "top":
            line = mask[pos, :]
        elif side == "bottom":
            line = mask[h - 1 - pos, :]
        elif side == "left":
            line = mask[:, pos]
        else:
            line = mask[:, w - 1 - pos]
        occ.append(float(np.count_nonzero(line)) / max(span, 1.0))
    return occ


def _occupancy_run_from_side(mask: np.ndarray, side: str) -> int:
    occ = _occupancy_values_from_side(mask, side)
    limit = len(occ)
    p = 0
    while p < limit and occ[p] < MIN_BORDER_OCCUPANCY:
        p += 1
    if p >= limit or p > MAX_OCCUPANCY_LEAD_IN:
        return 0

    start = p
    while p < limit and occ[p] >= MIN_BORDER_OCCUPANCY:
        p += 1
    run = p - start
    if run < MIN_OCCUPANCY_RUN:
        return 0
    return start + run


def _extend_occupancy_run_from_side(mask: np.ndarray, side: str, run: int) -> int:
    if run <= 0:
        return 0

    occ = _occupancy_values_from_side(mask, side)
    limit = len(occ)
    p = int(run)
    last_keep = p - 1
    gaps = 0
    max_p = min(limit, p + MAX_OCCUPANCY_EXTENSION)

    while p < max_p:
        if occ[p] >= MIN_BORDER_TAIL_OCCUPANCY:
            last_keep = p
            gaps = 0
            p += 1
            continue

        if (
            gaps < MAX_OCCUPANCY_GAP
            and p + 1 < max_p
            and occ[p + 1] >= MIN_BORDER_TAIL_OCCUPANCY
        ):
            last_keep = p
            gaps += 1
            p += 1
            continue

        break

    return last_keep + 1
# ================ИСПОЛЬЗУЕТ 3 СВЕРХУ
#запускает поиск по всем сторонам
def detect_border_stripes(mask: np.ndarray) -> dict:
    h, w = mask.shape
    y_samples = _sample_positions(h)
    x_samples = _sample_positions(w)
    left_runs = [_run_white_from_side(mask, y, "left") for y in y_samples]
    right_runs = [_run_white_from_side(mask, y, "right") for y in y_samples]
    top_runs = [_run_white_from_side(mask, x, "top") for x in x_samples]
    bottom_runs = [_run_white_from_side(mask, x, "bottom") for x in x_samples]
    left_occ = _occupancy_run_from_side(mask, "left")
    right_occ = _occupancy_run_from_side(mask, "right")
    top_occ = _occupancy_run_from_side(mask, "top")
    bottom_occ = _occupancy_run_from_side(mask, "bottom")

    left = max(_robust_border_width(left_runs), _extend_occupancy_run_from_side(mask, "left", left_occ))
    right = max(_robust_border_width(right_runs), _extend_occupancy_run_from_side(mask, "right", right_occ))
    top = max(_robust_border_width(top_runs), _extend_occupancy_run_from_side(mask, "top", top_occ))
    bottom = max(_robust_border_width(bottom_runs), _extend_occupancy_run_from_side(mask, "bottom", bottom_occ))

    return {
        "left": left,
        "right": right,
        "top": top,
        "bottom": bottom,
        "left_runs": left_runs,
        "right_runs": right_runs,
        "top_runs": top_runs,
        "bottom_runs": bottom_runs,
        "y_samples": y_samples,
        "x_samples": x_samples,
    }

#ВЫРЕЗАТЬ РАМКУ. можно указать стороны l r t b, либо отрежет все 4 (mask, "t", "b")
def cleanup_mask_by_detected_border_hard(
    mask: np.ndarray,
    *sides: str
) -> tuple[np.ndarray, dict]:
    """
    Удаляет рамку с выбранных сторон.

    Возвращает:
    - out: маска после среза
    - info: словарь с детекцией и реально применёнными срезами (cut_*)
    """

    if mask.max() > 1:
        mask = normalize_mask(mask)

    info = detect_border_stripes(mask)

    out = mask.copy()
    h, w = out.shape

    l = int(info["left"]) + 1 if int(info["left"]) > 0 else 0
    r = int(info["right"]) + 1 if int(info["right"]) > 0 else 0
    t = int(info["top"]) + 1 if int(info["top"]) > 0 else 0
    b = int(info["bottom"]) + 1 if int(info["bottom"]) > 0 else 0

    if not sides:
        cut_left = cut_right = cut_top = cut_bottom = True
    else:
        selected = {s.lower() for s in sides}
        cut_left = "l" in selected
        cut_right = "r" in selected
        cut_top = "t" in selected
        cut_bottom = "b" in selected

    applied = {
        "cut_left": l if cut_left and l > 0 else 0,
        "cut_right": r if cut_right and r > 0 else 0,
        "cut_top": t if cut_top and t > 0 else 0,
        "cut_bottom": b if cut_bottom and b > 0 else 0,
    }

    if applied["cut_left"] > 0:
        out[:, :min(applied["cut_left"], w)] = 0
    if applied["cut_right"] > 0:
        out[:, max(0, w - applied["cut_right"]):w] = 0
    if applied["cut_top"] > 0:
        out[:min(applied["cut_top"], h), :] = 0
    if applied["cut_bottom"] > 0:
        out[max(0, h - applied["cut_bottom"]):h, :] = 0

    out = normalize_mask(out)
    out = keep_components_by_area(out, min_area=550)
    info["applied"] = applied

    return out, info

#ВЫРЕЗАНИЕ ПО МАСКЕ
def make_object_bgra(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w, 4), dtype=np.uint8)
    out[:, :, :3] = img_bgr
    out[:, :, 3] = mask
    out[mask == 0, :3] = 0
    return out

#Восстановление разрывов вызванных срезом рамки
def restore_bridges_from_cut_lines(
    seed: np.ndarray,
    info: dict,
    offset: int = 2,
    min_run: int = 1,
    min_gap: int = 2,
    max_gap: int = 80,
    bridge_thickness: int = 2,
) -> np.ndarray:
    """
    Восстанавливает короткие мостики сразу для всех 4 сторон,
    но только рядом с реальными линиями среза рамки.

    Логика для каждой стороны:
    - берем cut_* из info["applied"]
    - если по стороне ничего не срезали -> пропуск
    - смотрим только 2-3 линии сразу после среза
    - ищем 2 белых ранна с gap между ними
    - если gap подходит по размеру, дорисовываем короткий мостик
    """

    out = normalize_mask(seed)
    h, w = out.shape
    applied = info.get("applied", {})

    def find_runs(line: np.ndarray) -> list[tuple[int, int]]:
        runs, start = [], None
        for i, v in enumerate(line):
            if v > 0 and start is None:
                start = i
            elif v == 0 and start is not None:
                if i - start >= min_run:
                    runs.append((start, i - 1))
                start = None
        if start is not None and len(line) - start >= min_run:
            runs.append((start, len(line) - 1))
        return runs

    def find_gaps(line: np.ndarray) -> list[tuple[int, int]]:
        runs = find_runs(line)
        gaps = []
        if len(runs) < 2:
            return gaps

        for a, b in zip(runs, runs[1:]):
            g0, g1 = a[1] + 1, b[0] - 1
            gap = g1 - g0 + 1
            if min_gap <= gap <= max_gap:
                gaps.append((g0, g1))

        return gaps

    sides = (
        ("t", int(applied.get("cut_top", 0))),
        ("b", int(applied.get("cut_bottom", 0))),
        ("l", int(applied.get("cut_left", 0))),
        ("r", int(applied.get("cut_right", 0))),
    )

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for side, cut in sides:
        if cut <= 0:
            continue

        bridge = np.zeros_like(out, dtype=np.uint8)

        if side == "t":
            pos_iter = range(min(max(cut, 0), h - 1), min(cut + offset + 1, h))
            for y in pos_iter:
                gaps = find_gaps(out[y, :])
                if not gaps:
                    continue
                for g0, g1 in gaps:
                    cv2.line(bridge, (g0, y), (g1, y), 255, bridge_thickness)
                break

        elif side == "b":
            y0 = max(0, h - cut - offset)
            y1 = min(h, h - cut)
            for y in range(y1 - 1, y0 - 1, -1):
                gaps = find_gaps(out[y, :])
                if not gaps:
                    continue
                for g0, g1 in gaps:
                    cv2.line(bridge, (g0, y), (g1, y), 255, bridge_thickness)
                break

        elif side == "l":
            pos_iter = range(min(max(cut, 0), w - 1), min(cut + offset + 1, w))
            for x in pos_iter:
                gaps = find_gaps(out[:, x])
                if not gaps:
                    continue
                for g0, g1 in gaps:
                    cv2.line(bridge, (x, g0), (x, g1), 255, bridge_thickness)
                break

        elif side == "r":
            x0 = max(0, w - cut - offset)
            x1 = min(w, w - cut)
            for x in range(x1 - 1, x0 - 1, -1):
                gaps = find_gaps(out[:, x])
                if not gaps:
                    continue
                for g0, g1 in gaps:
                    cv2.line(bridge, (x, g0), (x, g1), 255, bridge_thickness)
                break

        if np.count_nonzero(bridge) > 0:
            bridge = cv2.dilate(bridge, k, iterations=1)
            out = normalize_mask(cv2.bitwise_or(out, bridge))

    return out


#ХЭЛПЕРЫ ЕСЛИ НАЧНЕТ ЛОВИТЬСЯ УЗОР
def _mean_patch_bgr(img_bgr: np.ndarray, cx: int, cy: int, radius: int) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    x0 = max(0, cx - radius)
    x1 = min(w, cx + radius + 1)
    y0 = max(0, cy - radius)
    y1 = min(h, cy + radius + 1)
    patch = img_bgr[y0:y1, x0:x1].astype(np.float32)
    return patch.mean(axis=(0, 1))

def _patch_stats_bgr(img_bgr: np.ndarray, cx: int, cy: int, radius: int) -> tuple[np.ndarray, float]:
    h, w = img_bgr.shape[:2]
    x0 = max(0, cx - radius)
    x1 = min(w, cx + radius + 1)
    y0 = max(0, cy - radius)
    y1 = min(h, cy + radius + 1)
    patch = img_bgr[y0:y1, x0:x1].astype(np.float32)
    mean = patch.mean(axis=(0, 1))
    rough = float(np.sqrt(np.mean(np.var(patch.reshape(-1, 3), axis=0))))
    return mean, rough


def _build_bg_from_corner_samples(
    h: int,
    w: int,
    samples: list[tuple[tuple[float, float], np.ndarray]],
) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    bg_acc = np.zeros((h, w, 3), dtype=np.float32)
    w_acc = np.zeros((h, w, 1), dtype=np.float32)

    for (cx, cy), color in samples:
        dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
        weight = 1.0 / np.maximum(dist2, 1.0)
        bg_acc += weight[..., None] * color[None, None, :]
        w_acc += weight[..., None]

    return bg_acc / np.maximum(w_acc, 1e-6)


def _rgb_distance_to_segment(image_f32: np.ndarray, c0: np.ndarray, c1: np.ndarray) -> np.ndarray:
    v = (c1 - c0).astype(np.float32)
    vv = float(np.dot(v, v))
    if vv < 1e-6:
        return np.linalg.norm(image_f32 - c0[None, None, :], axis=2)

    p = image_f32 - c0[None, None, :]
    t = np.clip(np.sum(p * v[None, None, :], axis=2) / vv, 0.0, 1.0)
    proj = c0[None, None, :] + t[..., None] * v[None, None, :]
    return np.linalg.norm(image_f32 - proj, axis=2)


def build_foreground_gate_from_card_bg(
    img_bgr: np.ndarray,
    applied_cuts: dict | None = None,
    *,
    return_debug: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    h, w = img_bgr.shape[:2]

    cuts = applied_cuts or {}
    cut_left = max(int(cuts.get("cut_left", 0)), 0)
    cut_right = max(int(cuts.get("cut_right", 0)), 0)
    cut_top = max(int(cuts.get("cut_top", 0)), 0)
    cut_bottom = max(int(cuts.get("cut_bottom", 0)), 0)

    inner_x0 = float(cut_left)
    inner_x1 = float(max(cut_left, w - 1 - cut_right))
    inner_y0 = float(cut_top)
    inner_y1 = float(max(cut_top, h - 1 - cut_bottom))

    inner_w = max(1.0, inner_x1 - inner_x0 + 1.0)
    inner_h = max(1.0, inner_y1 - inner_y0 + 1.0)

    inset = max(8, int(round(min(inner_h, inner_w) * 0.08)))
    radius = max(2, int(round(min(h, w) * 0.02)))

    left_x = int(round(inner_x0 + inset))
    right_x = int(round(inner_x1 - inset))
    top_y = int(round(inner_y0 + inset))
    bottom_y = int(round(inner_y1 - inset))

    # Build the background from inner corners after the preliminary border cut.
    # This keeps the old successful behavior, but avoids sampling the raw outer frame.
    sample_points = {
        "tl": (left_x, top_y),
        "tr": (right_x, top_y),
        "bl": (left_x, bottom_y),
        "br": (right_x, bottom_y),
    }

    stats = []
    for name, (cx, cy) in sample_points.items():
        cx = max(0, min(w - 1, cx))
        cy = max(0, min(h - 1, cy))
        mean, rough = _patch_stats_bgr(img_bgr, cx, cy, radius)
        stats.append((name, (float(cx), float(cy)), mean, rough))

    stats_by_name = {name: (point, mean, rough) for name, point, mean, rough in stats}

    all_names = ["tl", "tr", "bl", "br"]
    colors = np.stack([stats_by_name[name][1] for name in all_names], axis=0)
    median_color = np.median(colors, axis=0)

    scored = []
    for name in all_names:
        point, mean, rough = stats_by_name[name]
        consistency = float(np.linalg.norm(mean - median_color))
        score = consistency + 1.5 * rough
        scored.append((score, name, point, mean, rough))

    scored.sort(key=lambda item: item[0])
    selected = scored[:3]

    bg = _build_bg_from_corner_samples(
        h,
        w,
        [(point, mean) for _, _, point, mean, _ in selected],
    )

    img_f32 = img_bgr.astype(np.float32)
    diff = np.linalg.norm(img_f32 - bg, axis=2)

    border = max(10, int(round(min(inner_h, inner_w) * 0.06)))
    border_vals_parts = []
    for _, _, point, _, _ in selected:
        cx = int(round(point[0]))
        cy = int(round(point[1]))
        x0 = max(0, cx - border)
        x1 = min(w, cx + border + 1)
        y0 = max(0, cy - border)
        y1 = min(h, cy + border + 1)
        border_vals_parts.append(diff[y0:y1, x0:x1].ravel())
    border_vals = np.concatenate(border_vals_parts)

    border_med = float(np.median(border_vals))
    border_mad = float(np.median(np.abs(border_vals - border_med)))
    thr = float(np.clip(border_med + 3.0 * max(border_mad, 2.0), 18.0, 40.0))

    gate = (diff >= thr).astype(np.uint8) * 255

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    gate = cv2.morphologyEx(gate, cv2.MORPH_OPEN, k3, iterations=1)
    gate = cv2.morphologyEx(gate, cv2.MORPH_CLOSE, k5, iterations=1)

    gate = normalize_mask(gate)

    if not return_debug:
        return gate

    top_means = [mean for _, name, _, mean, _ in selected if name in ("tl", "tr")]
    bottom_means = [mean for _, name, _, mean, _ in selected if name in ("bl", "br")]
    all_means = [mean for _, _, _, mean, _ in selected]
    top_mean = np.mean(top_means if top_means else all_means, axis=0).astype(np.float32)
    bottom_mean = np.mean(bottom_means if bottom_means else all_means, axis=0).astype(np.float32)

    return gate, {
        "diff": diff,
        "thr": float(thr),
        "top_mean": top_mean,
        "bottom_mean": bottom_mean,
    }


def _apply_foreground_gate_from_card_bg(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    applied_cuts: dict | None = None,
    *,
    dilate_iterations: int = 1,
    min_area: int = 150,
) -> tuple[np.ndarray, np.ndarray]:
    fg_gate, gate_debug = build_foreground_gate_from_card_bg(
        img_bgr,
        applied_cuts,
        return_debug=True,
    )
    fg_gate = cv2.dilate(
        fg_gate,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=dilate_iterations,
    )
    gated = remove_confirmed_background_regions(mask, fg_gate, min_bg_area=24)
    gated = keep_components_by_area(gated, min_area=min_area)
    gated = remove_small_border_components(gated, max_area=800)

    gated = remove_weak_secondary_components(gated, min_ratio_to_largest=0.16)

    diff = gate_debug["diff"]
    thr = float(gate_debug["thr"])
    seg = _rgb_distance_to_segment(
        img_bgr.astype(np.float32),
        gate_debug["top_mean"],
        gate_debug["bottom_mean"],
    )
    weak_bg = np.logical_and.reduce([
        gated > 0,
        seg < 4.0,
        diff < thr + 10.0,
    ])
    if np.count_nonzero(weak_bg) >= 500:
        gated[weak_bg] = 0
        gated = keep_components_by_area(gated, min_area=min_area)
        gated = remove_small_border_components(gated, max_area=800)
        gated = remove_weak_secondary_components(gated, min_ratio_to_largest=0.16)

    return gated, fg_gate



#ОБЕРТКИ ДЛЯ ВЫЗОВА
def extract_object_with_info_from_crop(crop_bgr: np.ndarray) -> tuple[np.ndarray, dict]:
    crop_scaled = resize_long_side_if_needed(crop_bgr)
    em_scharr = _edge_map_scharr_max_channel_auto(crop_scaled)
    mask_base = edge_mask_base(em_scharr)

    # Сначала грубо оцениваем срезы рамки, потом строим fg_gate уже
    # внутри очищенного окна карточки.
    _, pre_info = cleanup_mask_by_detected_border_hard(mask_base)
    mask_base, _ = _apply_foreground_gate_from_card_bg(
        crop_scaled,
        mask_base,
        pre_info.get("applied"),
        dilate_iterations=FG_GATE_DILATE_ITERATIONS,
        min_area=150,
    )

    seed, info = cleanup_mask_by_detected_border_hard(mask_base)
    seed_fixed = restore_bridges_from_cut_lines(seed, info)
    filled_fixed = fill_internal_holes(seed_fixed)
    filled_fixed, _ = _apply_foreground_gate_from_card_bg(
        crop_scaled,
        filled_fixed,
        info.get("applied"),
        dilate_iterations=FG_GATE_DILATE_ITERATIONS,
        min_area=150,
    )
    obj = make_object_bgra(crop_scaled, filled_fixed)
    return obj, {
        "scaled_shape": crop_scaled.shape[:2],
        "applied_cuts": dict(info.get("applied", {})),
        "cleanup_info": info,
    }


def extract_object_from_crop(crop_bgr: np.ndarray) -> np.ndarray:
    obj, _ = extract_object_with_info_from_crop(crop_bgr)
    return obj


def extract_object_from_path(path: Path) -> np.ndarray:
    crop_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if crop_bgr is None:
        raise FileNotFoundError(f"Unreadable image: {path}")
    return extract_object_from_crop(crop_bgr)


def extract_object_with_info_from_path(path: Path) -> tuple[np.ndarray, dict]:
    crop_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if crop_bgr is None:
        raise FileNotFoundError(f"Unreadable image: {path}")
    return extract_object_with_info_from_crop(crop_bgr)


#читалка для дебага
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
def save_img(path: Path, img: np.ndarray):
    ensure_dir(path.parent)
    cv2.imwrite(str(path), img)


def save_bg_candidate(path: Path, mask: np.ndarray, fg_gate: np.ndarray):
    bg_candidate = np.logical_and(mask > 0, fg_gate == 0).astype(np.uint8) * 255
    save_img(path, bg_candidate)


def run_debug2():
    ensure_dir(OUT_DIR)

    files = sorted(CROP_DIR.glob("*.png"))
    if not files:
        raise FileNotFoundError(f"Нет png в {CROP_DIR}")

    print(f"[INFO] crops: {len(files)}")
    print(f"[OUT]  {OUT_DIR}")

    for path in files:
        print(f"[CROP] {path.name}")

        crop_src = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if crop_src is None:
            print(f"[WARN] skip unreadable: {path}")
            continue

        crop_out_dir = OUT_DIR / path.stem
        ensure_dir(crop_out_dir)

        save_img(crop_out_dir / "original.png", crop_src)

        crop_scaled = resize_long_side_if_needed(crop_src)
        save_img(crop_out_dir / "scaled_input.png", crop_scaled)

        # карта  — adaptive color scharr
        em_scharr2 = _edge_map_scharr_max_channel_auto(crop_scaled)
        save_img(crop_out_dir / "em_scharr2.png", em_scharr2)


        # ---------- PIPELINE ----------
        mask_base2 = edge_mask_base(em_scharr2)
        save_img(crop_out_dir / "mask_base2.png", mask_base2)
        # Анти-узор после грубой оценки срезов рамки.
        _, pre_info2 = cleanup_mask_by_detected_border_hard(mask_base2)
        mask_base2, fg_gate_pre2 = _apply_foreground_gate_from_card_bg(
            crop_scaled,
            mask_base2,
            pre_info2.get("applied"),
            dilate_iterations=FG_GATE_DILATE_ITERATIONS,
            min_area=150,
        )
        save_img(crop_out_dir / "fg_gate_pre2.png", fg_gate_pre2)

        seed2, info2 = cleanup_mask_by_detected_border_hard(mask_base2)
        save_img(crop_out_dir / "seed_after_cut_all2.png", seed2)

        seed_fixed2 = restore_bridges_from_cut_lines(seed2, info2)
        save_img(crop_out_dir / "seed_fixed2.png", seed_fixed2)

        filled_fixed2 = fill_internal_holes(seed_fixed2)
        filled_before_gate2 = filled_fixed2.copy()
        filled_fixed2, fg_gate_final2 = _apply_foreground_gate_from_card_bg(
            crop_scaled,
            filled_fixed2,
            info2.get("applied"),
            dilate_iterations=FG_GATE_DILATE_ITERATIONS,
            min_area=150,
        )
        save_img(crop_out_dir / "fg_gate_final2.png", fg_gate_final2)
        save_bg_candidate(crop_out_dir / "bg_candidate_final2.png", filled_before_gate2, fg_gate_final2)
        save_img(crop_out_dir / "filled__fixed.png", filled_fixed2)

        obj2 = make_object_bgra(crop_scaled, filled_fixed2)
        save_img(crop_out_dir / "object2.png", obj2)
        save_img(OBJECTS_ONLY_DIR / f"{CROP_DIR.name}__{path.name}", obj2)


    print("[DONE]")
if __name__ == "__main__":
    run_debug2()



