from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TEST_DIR = Path(__file__).resolve().parent
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

CROP_DIR = TEST_DIR / "crop_swords"
OUT_DIR = TEST_DIR / "debug_reframed_swords"
CLAYMORE_CROP_DIR = TEST_DIR / "crop_claymores"
CLAYMORE_OUT_DIR = TEST_DIR / "debug_reframed_claymores"
POLEARM_CROP_DIR = TEST_DIR / "crop_polearms"
POLEARM_OUT_DIR = TEST_DIR / "debug_reframed_polearms"

ALL_REF_DIR = ROOT / "cache" / "enka_ref_weapons"
WEAPONS_JSON_PATH = ROOT / "data" / "weapons.json"

from crop_extractor import (
    extract_object_with_info_from_crop,
    resize_long_side_if_needed,
)

REF_EXTRA_SCALE = 1.05
ALPHA_MASK_THRESHOLD = 32


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def reset_dir(path: Path):
    target = path.resolve()
    test_root = TEST_DIR.resolve()
    if target != test_root and test_root not in target.parents:
        raise ValueError(f"Refusing to clear path outside test dir: {target}")
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)


def save_img(path: Path, img: np.ndarray):
    ensure_dir(path.parent)
    cv2.imwrite(str(path), img)


def build_square_window_compensation(
    source_shape: tuple[int, int],
    applied_cuts: dict | None,
    target_size: int = 256,
) -> tuple[np.ndarray, dict]:
    h, w = source_shape
    cuts = applied_cuts or {}

    left = float(max(int(cuts.get("cut_left", 0)), 0))
    right = float(max(int(cuts.get("cut_right", 0)), 0))
    top = float(max(int(cuts.get("cut_top", 0)), 0))
    bottom = float(max(int(cuts.get("cut_bottom", 0)), 0))

    inner_w = max(float(w) - left - right, 1.0)
    inner_h = max(float(h) - top - bottom, 1.0)

    # Hoyolab card is square. The detected cut_* values describe how much
    # extra window was present around that square after isotropic resize.
    # We therefore recover the effective square window and scale it back to 256.
    effective_side = max(min(inner_w, inner_h), 1.0)

    x0 = left + 0.5 * (inner_w - effective_side)
    y0 = top + 0.5 * (inner_h - effective_side)
    scale = float(target_size) / float(effective_side)

    matrix = np.array(
        [
            [scale, 0.0, -scale * x0],
            [0.0, scale, -scale * y0],
        ],
        dtype=np.float32,
    )

    return matrix, {
        "left": left,
        "right": right,
        "top": top,
        "bottom": bottom,
        "inner_w": inner_w,
        "inner_h": inner_h,
        "effective_side": effective_side,
        "scale": scale,
        "x0": x0,
        "y0": y0,
    }


def warp_bgra_to_square(
    image_bgra: np.ndarray,
    matrix: np.ndarray,
    target_size: int = 256,
) -> np.ndarray:
    out = cv2.warpAffine(
        image_bgra,
        matrix,
        (target_size, target_size),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    out_alpha = out[:, :, 3]
    out[out_alpha == 0, :3] = 0
    return out


def warp_bgr_to_square(
    image_bgr: np.ndarray,
    matrix: np.ndarray,
    target_size: int = 256,
) -> np.ndarray:
    return cv2.warpAffine(
        image_bgr,
        matrix,
        (target_size, target_size),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def load_compensated_crop(
    path: Path,
    target_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, dict]:
    crop_src = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if crop_src is None:
        raise FileNotFoundError(f"Unreadable crop image: {path}")

    crop_scaled = resize_long_side_if_needed(crop_src)
    object_bgra, info = extract_object_with_info_from_crop(crop_src)
    matrix, compensation = build_square_window_compensation(
        object_bgra.shape[:2],
        info.get("applied_cuts"),
        target_size=target_size,
    )

    crop_bg = warp_bgr_to_square(crop_scaled, matrix, target_size=target_size)
    crop_object = warp_bgra_to_square(object_bgra, matrix, target_size=target_size)
    return crop_bg, crop_object, {
        **info,
        "compensation": compensation,
    }


def load_compensated_claymore_crop(
    path: Path,
    target_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, dict]:
    return load_compensated_crop(path, target_size=target_size)


def alpha_to_binary(image_bgra: np.ndarray) -> np.ndarray:
    if image_bgra.ndim != 3 or image_bgra.shape[2] != 4:
        raise ValueError("Expected BGRA image")
    alpha = image_bgra[:, :, 3]
    return np.where(alpha > ALPHA_MASK_THRESHOLD, 255, 0).astype(np.uint8)


def normalize_bgra_alpha(image_bgra: np.ndarray) -> np.ndarray:
    if image_bgra.ndim != 3 or image_bgra.shape[2] != 4:
        raise ValueError("Expected BGRA image")
    out = image_bgra.copy()
    alpha = out[:, :, 3]
    invisible = alpha <= ALPHA_MASK_THRESHOLD
    out[invisible, :3] = 0
    out[invisible, 3] = 0
    return out


def load_ref_paths_by_type(weapon_type: str) -> list[Path]:
    with open(WEAPONS_JSON_PATH, "r", encoding="utf-8") as f:
        weapons = json.load(f)

    out: list[Path] = []
    missing: list[str] = []

    for weapon_id, info in weapons.items():
        if info.get("type") != weapon_type:
            continue

        ref_path = ALL_REF_DIR / f"{weapon_id}.png"
        if ref_path.exists():
            out.append(ref_path)
        else:
            missing.append(str(weapon_id))

    out.sort(key=lambda p: p.stem)

    if missing:
        print(f"[WARN] missing {weapon_type} refs in cache: {len(missing)}")

    return out


def load_all_sword_ref_paths() -> list[Path]:
    return load_ref_paths_by_type("Sword")


def load_all_claymore_ref_paths() -> list[Path]:
    return load_ref_paths_by_type("Claymore")


def load_all_polearm_ref_paths() -> list[Path]:
    return load_ref_paths_by_type("Polearm")


def load_bgra_from_path(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Unreadable image: {path}")

    if img.ndim == 2:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        alpha = np.where(img > 0, 255, 0).astype(np.uint8)
        return normalize_bgra_alpha(np.dstack([bgr, alpha]))

    if img.shape[2] == 3:
        bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = 255
        return normalize_bgra_alpha(bgra)

    if img.shape[2] == 4:
        return normalize_bgra_alpha(img)

    raise ValueError(f"Unsupported image shape: {img.shape}")


def cut_ref_bottom_percent(object_bgra: np.ndarray, cut_ratio: float) -> np.ndarray:
    if object_bgra.ndim != 3 or object_bgra.shape[2] != 4:
        raise ValueError("Expected BGRA image")

    cut_ratio = float(np.clip(cut_ratio, 0.0, 0.95))

    alpha = object_bgra[:, :, 3]
    ys, _ = np.where(alpha > ALPHA_MASK_THRESHOLD)
    if len(ys) == 0:
        raise ValueError("Reference alpha is empty")

    min_y = int(ys.min())
    max_y = int(ys.max())
    obj_h = max_y - min_y + 1

    cut_px = int(round(obj_h * cut_ratio))
    keep_max_y = max(min_y, max_y - cut_px)

    out = object_bgra.copy()
    out[keep_max_y + 1 :, :, :] = 0

    out_alpha = out[:, :, 3]
    out[out_alpha == 0, :3] = 0
    return out


class SwordReferenceReframer:
    TARGET_CANVAS_SIZE = 256
    TARGET_SWORD_LENGTH = 285.0
    TARGET_TOP_ENDPOINT_X = 219.0
    TARGET_BOTTOM_Y = 255.0

    LOWEST_CUT_RATIO = 0.02
    LOW_CUT_RATIO = 0.04
    LOW_PLUS_CUT_RATIO = 0.05
    DEFAULT_CUT_RATIO = 0.08
    MEDIUM_CUT_RATIO = 0.085
    MEDIUM_PLUS_CUT_RATIO = 0.105
    WIDE_THIN_CUT_RATIO = 0.12
    HIGH_CUT_RATIO = 0.13
    VERY_HIGH_CUT_RATIO = 0.16
    MAX_CUT_RATIO = 0.18

    LOW_TOP45_THRESHOLD = 0.40
    LOW_TOP45_BOTTOM_HEAVY_THRESHOLD = 0.32
    LOW_TOP45_WIDE_THRESHOLD = 40.0
    HIGH_TOP35_THRESHOLD = 0.40
    THIN_COMPACTNESS_THRESHOLD = 0.85
    GUARD_FOCUS_TOP45_THRESHOLD = 0.48
    GUARD_FOCUS_COMPACTNESS_THRESHOLD = 1.05
    NARROW_WIDTH_THRESHOLD = 20.0
    NARROW_TOP45_THRESHOLD = 0.44
    SHORT_BOT35_THRESHOLD = 0.26
    SHORT_WIDTH_THRESHOLD = 24.0
    VERY_WIDE_THRESHOLD = 43.0
    EXTREME_WIDE_THRESHOLD = 46.0
    VERY_WIDE_LOW_TOP45_THRESHOLD = 0.42
    VERY_WIDE_THIN_TOP45_THRESHOLD = 0.44
    COMPACT_BLADE_WIDTH_MIN = 20.0
    COMPACT_BLADE_WIDTH_MAX = 25.0
    COMPACT_BLADE_TOP35_MIN = 0.20
    COMPACT_BLADE_TOP35_MAX = 0.30
    COMPACT_BLADE_TOP45_MIN = 0.43
    COMPACT_BLADE_TOP45_MAX = 0.48
    COMPACT_BLADE_BOT35_MIN = 0.26
    COMPACT_BLADE_BOT35_MAX = 0.32
    COMPACT_BLADE_COMPACTNESS_MIN = 0.95

    MIN_SCALE = 0.45
    MAX_SCALE = 1.35

    def select_ref_cut_ratio(self, object_bgra: np.ndarray) -> float:
        features = self._geometry_features(object_bgra)
        top35_ratio = float(features["top35_ratio"])
        top45_ratio = float(features["top45_ratio"])
        bot35_ratio = float(features["bot35_ratio"])
        width_like = float(features["width_like"])
        compactness = float(features["compactness"])

        if (
            width_like >= self.EXTREME_WIDE_THRESHOLD
            and top45_ratio <= self.VERY_WIDE_LOW_TOP45_THRESHOLD
        ):
            return self.LOW_PLUS_CUT_RATIO

        if (
            width_like >= self.VERY_WIDE_THRESHOLD
            and top45_ratio < self.LOW_TOP45_THRESHOLD
        ):
            return self.LOW_CUT_RATIO

        if (
            width_like >= self.VERY_WIDE_THRESHOLD
            and compactness <= self.THIN_COMPACTNESS_THRESHOLD
            and top45_ratio >= self.VERY_WIDE_THIN_TOP45_THRESHOLD
        ):
            return self.WIDE_THIN_CUT_RATIO

        if top45_ratio < self.LOW_TOP45_THRESHOLD:
            if bot35_ratio >= self.LOW_TOP45_BOTTOM_HEAVY_THRESHOLD:
                return self.LOW_CUT_RATIO
            if width_like >= self.LOW_TOP45_WIDE_THRESHOLD:
                return self.MEDIUM_CUT_RATIO
            return self.LOWEST_CUT_RATIO

        if top35_ratio >= self.HIGH_TOP35_THRESHOLD:
            return self.HIGH_CUT_RATIO

        if compactness <= self.THIN_COMPACTNESS_THRESHOLD:
            return self.VERY_HIGH_CUT_RATIO

        if (
            top45_ratio >= self.GUARD_FOCUS_TOP45_THRESHOLD
            and compactness <= self.GUARD_FOCUS_COMPACTNESS_THRESHOLD
        ):
            return self.MAX_CUT_RATIO

        # Compact swords with a dense central module and relatively even blade
        # profile tend to be shown a bit tighter in-game.
        if (
            self.COMPACT_BLADE_WIDTH_MIN <= width_like <= self.COMPACT_BLADE_WIDTH_MAX
            and self.COMPACT_BLADE_TOP35_MIN <= top35_ratio <= self.COMPACT_BLADE_TOP35_MAX
            and self.COMPACT_BLADE_TOP45_MIN <= top45_ratio <= self.COMPACT_BLADE_TOP45_MAX
            and self.COMPACT_BLADE_BOT35_MIN <= bot35_ratio <= self.COMPACT_BLADE_BOT35_MAX
            and compactness >= self.COMPACT_BLADE_COMPACTNESS_MIN
        ):
            return self.HIGH_CUT_RATIO

        if (
            width_like <= self.NARROW_WIDTH_THRESHOLD
            and top45_ratio >= self.NARROW_TOP45_THRESHOLD
        ):
            return self.HIGH_CUT_RATIO + 0.035

        if (
            bot35_ratio <= self.SHORT_BOT35_THRESHOLD
            and width_like <= self.SHORT_WIDTH_THRESHOLD
        ):
            return self.MEDIUM_PLUS_CUT_RATIO

        return self.DEFAULT_CUT_RATIO

    def _geometry_features(self, object_bgra: np.ndarray) -> dict:
        if object_bgra.ndim != 3 or object_bgra.shape[2] != 4:
            raise ValueError("Expected BGRA image")

        alpha = object_bgra[:, :, 3]
        binary = np.where(alpha > ALPHA_MASK_THRESHOLD, 255, 0).astype(np.uint8)

        contour = self._largest_contour(binary)
        if contour is None or len(contour) < 2:
            raise ValueError("Failed to build contour for reference")

        p0, p1 = self._farthest_pair_on_hull(contour)
        pts = self._binary_points(binary)
        if len(pts) == 0:
            raise ValueError("Reference binary mask is empty")

        top, axis_unit, axis_len = self._axis_geometry(p0, p1)
        rel = pts - top[None, :]
        proj = rel @ axis_unit
        perp = np.array([-axis_unit[1], axis_unit[0]], dtype=np.float32)
        side = rel @ perp

        top35 = (proj >= 0.0) & (proj <= axis_len * 0.35)
        top45 = (proj >= 0.0) & (proj <= axis_len * 0.45)
        top35_ratio = float(np.mean(top35)) if len(top35) > 0 else 0.0
        top45_ratio = float(np.mean(top45)) if len(top45) > 0 else 0.0
        bot35_ratio = float(np.mean(proj >= axis_len * 0.65)) if len(proj) > 0 else 0.0

        width_like = float(np.std(side) * 2.0)
        width_like = max(width_like, 1e-6)
        compactness = float(len(pts)) / float(axis_len * width_like)
        return {
            "top35_ratio": top35_ratio,
            "top45_ratio": top45_ratio,
            "bot35_ratio": bot35_ratio,
            "width_like": width_like,
            "compactness": compactness,
        }

    def reframe(self, object_bgra: np.ndarray, extra_scale: float = 1.0) -> np.ndarray:
        if object_bgra.ndim != 3 or object_bgra.shape[2] != 4:
            raise ValueError("Expected BGRA image")

        alpha = object_bgra[:, :, 3]
        binary = np.where(alpha > ALPHA_MASK_THRESHOLD, 255, 0).astype(np.uint8)

        contour = self._largest_contour(binary)
        if contour is None or len(contour) < 2:
            raise ValueError("Failed to build contour for extracted object")

        p0, p1 = self._farthest_pair_on_hull(contour)

        current_length = float(np.linalg.norm(p1 - p0))
        if current_length <= 1e-6:
            raise ValueError("Degenerate sword length")

        current_top_x = self._top_endpoint_x(p0, p1)
        ys, _ = np.where(binary > 0)
        if len(ys) == 0:
            raise ValueError("Binary mask is empty after contour extraction")
        current_max_y = float(ys.max())

        scale = self.TARGET_SWORD_LENGTH / current_length
        scale *= float(extra_scale)
        scale = float(np.clip(scale, self.MIN_SCALE, self.MAX_SCALE))

        tx = float(self.TARGET_TOP_ENDPOINT_X - scale * current_top_x)
        ty = float(self.TARGET_BOTTOM_Y - scale * current_max_y)

        matrix = np.array(
            [
                [scale, 0.0, tx],
                [0.0, scale, ty],
            ],
            dtype=np.float32,
        )

        out = cv2.warpAffine(
            object_bgra,
            matrix,
            (self.TARGET_CANVAS_SIZE, self.TARGET_CANVAS_SIZE),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

        out_alpha = out[:, :, 3]
        out[out_alpha == 0, :3] = 0
        return out

    def _largest_contour(self, binary: np.ndarray):
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None
        return max(contours, key=cv2.contourArea)

    def _farthest_pair_on_hull(self, contour: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hull = cv2.convexHull(contour)
        points = hull[:, 0, :].astype(np.float32)

        if len(points) < 2:
            raise ValueError("Hull has fewer than 2 points")

        diff = points[:, None, :] - points[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        i, j = np.unravel_index(np.argmax(dist2), dist2.shape)

        return points[i], points[j]

    def _binary_points(self, binary: np.ndarray) -> np.ndarray:
        ys, xs = np.where(binary > 0)
        if len(xs) == 0:
            return np.empty((0, 2), dtype=np.float32)
        return np.stack(
            [xs.astype(np.float32), ys.astype(np.float32)],
            axis=1,
        )

    def _axis_geometry(
        self,
        p0: np.ndarray,
        p1: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        top = p0 if p0[1] < p1[1] else p1
        bottom = p1 if p0[1] < p1[1] else p0
        axis = bottom - top
        axis_len = float(np.linalg.norm(axis))
        if axis_len <= 1e-6:
            raise ValueError("Degenerate sword axis")
        axis_unit = axis / axis_len
        return top, axis_unit, axis_len

    def _top_endpoint_x(self, p0: np.ndarray, p1: np.ndarray) -> float:
        top = p0 if p0[1] < p1[1] else p1
        return float(top[0])


class ClaymoreReferenceReframer:
    TARGET_CANVAS_SIZE = 256
    TARGET_CLAYMORE_LENGTH = 309.0
    TARGET_TOP_ENDPOINT_X = 226.0
    TARGET_BOTTOM_Y = 255.0

    WINDOW_PROTOTYPES = {
        "12101": {
            "features": [0.141456, 0.319997, 0.53545, 31.066919, 1.18641, 0.404255, 0.498972, 0.68254, 0.569096, 0.536711, 1.972395],
            "center": 0.488,
            "span": 0.575,
            "length": 335.94,
            "midpoint": [132.5, 115.5],
        },
        "12301": {
            "features": [0.109833, 0.301465, 0.485832, 59.361885, 0.834156, 0.446809, 0.494936, 0.650794, 0.561215, 0.785852, 2.9791],
            "center": 0.4875,
            "span": 0.975,
            "length": 329.61,
            "midpoint": [124.5, 125.0],
        },
        "12302": {
            "features": [0.126041, 0.361831, 0.462243, 38.919651, 0.904912, 0.425532, 0.484288, 0.650794, 0.552029, 0.743326, 3.440809],
            "center": 0.45,
            "span": 0.55,
            "length": 312.43,
            "midpoint": [126.0, 122.0],
        },
        "12305": {
            "features": [0.162031, 0.265908, 0.555275, 39.213173, 1.084394, 0.553191, 0.509099, 0.634921, 0.572406, 0.579075, 2.069947],
            "center": 0.5125,
            "span": 0.975,
            "length": 331.0,
            "midpoint": [118.5, 120.0],
        },
        "12401": {
            "features": [0.123005, 0.329649, 0.470723, 61.374271, 0.78369, 0.446809, 0.518493, 0.634921, 0.5583, 0.752921, 3.234364],
            "center": 0.513,
            "span": 0.925,
            "length": 316.24,
            "midpoint": [120.5, 135.5],
        },
        "12402": {
            "features": [0.165868, 0.287448, 0.521333, 35.259705, 1.158901, 0.468085, 0.496053, 0.619048, 0.566275, 0.579986, 1.97662],
            "center": 0.513,
            "span": 0.575,
            "length": 299.33,
            "midpoint": [116.5, 131.0],
        },
        "12405": {
            "features": [0.144481, 0.346486, 0.482711, 52.50169, 1.04676, 0.361702, 0.509245, 0.603175, 0.553658, 0.612366, 2.165838],
            "center": 0.5,
            "span": 0.95,
            "length": 331.23,
            "midpoint": [119.0, 133.0],
        },
        "12406": {
            "features": [0.176909, 0.3045, 0.529603, 46.534206, 1.096627, 0.510638, 0.52175, 0.603175, 0.563936, 0.517923, 1.757437],
            "center": 0.495,
            "span": 0.88,
            "length": 340.07,
            "midpoint": [122.5, 127.5],
        },
        "12409": {
            "features": [0.082088, 0.242042, 0.533132, 67.235092, 0.95354, 0.531915, 0.552358, 0.539683, 0.577096, 0.754204, 2.170608],
            "center": 0.525,
            "span": 0.95,
            "length": 310.0,
            "midpoint": [127.5, 131.0],
        },
        "12412": {
            "features": [0.263231, 0.384202, 0.475594, 38.914619, 1.44314, 0.468085, 0.513754, 0.68254, 0.525391, 0.329032, 1.514687],
            "center": 0.5125,
            "span": 0.975,
            "length": 253.31,
            "midpoint": [117.0, 166.0],
        },
        "12415": {
            "features": [0.150972, 0.277277, 0.546878, 27.714783, 1.148979, 0.553191, 0.510216, 0.666667, 0.563378, 0.583898, 2.226512],
            "center": 0.5375,
            "span": 0.575,
            "length": 293.51,
            "midpoint": [132.5, 118.0],
        },
        "12417": {
            "features": [0.173224, 0.392424, 0.453379, 33.86739, 1.098511, 0.404255, 0.492805, 0.634921, 0.545604, 0.591254, 2.175574],
            "center": 0.44,
            "span": 0.816,
            "length": 341.4,
            "midpoint": [126.5, 127.5],
        },
        "12424": {
            "features": [0.18173, 0.281805, 0.554959, 34.359905, 1.183723, 0.425532, 0.546835, 0.603175, 0.574993, 0.484487, 1.797893],
            "center": 0.465,
            "span": 0.83,
            "length": 338.1,
            "midpoint": [128.0, 127.5],
        },
        "12430": {
            "features": [0.180936, 0.230559, 0.645218, 34.575542, 0.830196, 0.617021, 0.537038, 0.714286, 0.573177, 0.705588, 2.739482],
            "center": 0.538,
            "span": 0.925,
            "length": 392.19,
            "midpoint": [124.5, 101.0],
        },
        "12431": {
            "features": [0.1456, 0.254217, 0.556696, 33.617168, 0.980933, 0.531915, 0.542461, 0.634921, 0.573024, 0.600058, 2.28654],
            "center": 0.488,
            "span": 0.725,
            "length": 282.97,
            "midpoint": [139.5, 119.0],
        },
        "12432": {
            "features": [0.12637, 0.30206, 0.555842, 47.678684, 1.160818, 0.382979, 0.547418, 0.571429, 0.581871, 0.60993, 1.835269],
            "center": 0.5125,
            "span": 0.975,
            "length": 314.41,
            "midpoint": [116.0, 137.0],
        },
        "12512": {
            "features": [0.339372, 0.606559, 0.223861, 74.310173, 0.650324, 0.382979, 0.450051, 0.571429, 0.432113, 0.784099, 2.4233],
            "center": 0.5,
            "span": 0.9,
            "length": 259.48,
            "midpoint": [113.0, 143.5],
        },
        "12513": {
            "features": [0.151477, 0.30052, 0.530715, 38.131775, 0.924538, 0.425532, 0.520476, 0.650794, 0.573706, 0.533291, 1.919427],
            "center": 0.363,
            "span": 0.725,
            "length": 331.61,
            "midpoint": [136.0, 111.0],
        },
        "12514": {
            "features": [0.186268, 0.382647, 0.482708, 59.058956, 1.01962, 0.404255, 0.519508, 0.555556, 0.549414, 0.563726, 1.912725],
            "center": 0.5,
            "span": 0.95,
            "length": 321.05,
            "midpoint": [123.5, 127.5],
        },
    }
    VARIANT_BENCHMARK_TARGETS = {
        "12101": {"center": 0.488, "span": 0.575, "length": 335.94, "midpoint": [132.5, 115.5]},
        "12401": {"center": 0.513, "span": 0.925, "length": 316.24, "midpoint": [120.5, 135.5]},
        "12402": {"center": 0.513, "span": 0.575, "length": 299.33, "midpoint": [116.5, 131.0]},
        "12405": {"center": 0.500, "span": 0.950, "length": 331.23, "midpoint": [119.0, 133.0]},
        "12406": {"center": 0.495, "span": 0.880, "length": 340.07, "midpoint": [122.5, 127.5]},
        "12409": {"center": 0.525, "span": 0.950, "length": 310.00, "midpoint": [127.5, 131.0]},
        "12412": {"center": 0.5125, "span": 0.975, "length": 253.31, "midpoint": [117.0, 166.0]},
        "12415": {"center": 0.5375, "span": 0.575, "length": 293.51, "midpoint": [132.5, 118.0]},
        "12417": {"center": 0.440, "span": 0.816, "length": 341.40, "midpoint": [126.5, 127.5]},
        "12424": {"center": 0.465, "span": 0.830, "length": 338.10, "midpoint": [128.0, 127.5]},
        "12430": {"center": 0.538, "span": 0.925, "length": 392.19, "midpoint": [124.5, 101.0]},
        "12431": {"center": 0.488, "span": 0.725, "length": 282.97, "midpoint": [139.5, 119.0]},
        "12432": {"center": 0.5125, "span": 0.975, "length": 314.41, "midpoint": [116.0, 137.0]},
        "12513": {"center": 0.363, "span": 0.725, "length": 331.61, "midpoint": [136.0, 111.0]},
        "12514": {"center": 0.500, "span": 0.950, "length": 321.05, "midpoint": [123.5, 127.5]},
        "12301": {"center": 0.4875, "span": 0.975, "length": 329.61, "midpoint": [124.5, 125.0]},
        "12302": {"center": 0.450, "span": 0.550, "length": 312.43, "midpoint": [126.0, 122.0]},
        "12305": {"center": 0.5125, "span": 0.975, "length": 331.0, "midpoint": [118.5, 120.0]},
        "12512": {"center": 0.500, "span": 0.900, "length": 259.48, "midpoint": [113.0, 143.5]},
    }
    PROTOTYPE_FEATURE_SCALES = np.array(
        [0.05, 0.05, 0.05, 8.0, 0.09, 0.05, 0.05, 0.08, 0.05, 0.08, 0.25],
        dtype=np.float32,
    )
    PROTOTYPE_SIGMA = 0.35
    PROTOTYPE_K = 5

    MIN_SCALE = 0.75
    MAX_SCALE = 1.35
    MAX_WINDOW_SCALE = 1.95

    def select_window_params(self, object_bgra: np.ndarray) -> dict:
        features = self._geometry_features(object_bgra)
        saliency = self._saliency_profile_metrics(object_bgra)
        top45 = float(features["top45_ratio"])
        bot45 = float(features["bot45_ratio"])
        compactness = float(features["compactness"])
        width_cv = float(saliency["width_cv"])
        width_peakiness = float(saliency["width_peakiness"])
        feature_vec = np.array(
            [
                features["top35_ratio"],
                features["top45_ratio"],
                features["bot45_ratio"],
                features["width_like"],
                features["compactness"],
                features["peak_width_ratio"],
                saliency["hero_center"],
                saliency["hero_span"],
                saliency["mass_center"],
                saliency["width_cv"],
                saliency["width_peakiness"],
            ],
            dtype=np.float32,
        )
        params = self._blend_from_prototypes(
            feature_vec,
            list(self.WINDOW_PROTOTYPES.keys()),
            "exemplar",
        )
        # For unseen claymores, a small amount of extra visible span is safer
        # than over-cropping. Known calibrated cases converge to max_weight≈1,
        # so this leaves them unchanged while slightly broadening ambiguous refs.
        uncertainty = max(0.0, 1.0 - float(params.get("max_weight", 1.0)))
        if uncertainty > 1e-3 and float(params["span"]) < 0.90:
            params["span"] = float(min(float(params["span"]) + 0.06 * uncertainty, 0.92))
            params["family"] += f", span_guard:{0.06 * uncertainty:.3f}"
        # Compact, smooth-profile claymores with visibly stronger lower blade
        # can be over-cropped by the prototype blend. Broaden them a little and
        # nudge the window down, without touching the guard-focused subtype.
        if (
            float(params["span"]) < 0.76
            and top45 <= 0.32
            and bot45 >= 0.52
            and compactness >= 1.14
            and width_cv <= 0.53
            and width_peakiness <= 1.95
        ):
            params["center"] = float(min(float(params["center"]) + 0.012, 0.56))
            params["span"] = float(min(float(params["span"]) + 0.06, 0.80))
            params["family"] += ", lower_blade_guard"
        return params

    def _saliency_profile_metrics(self, object_bgra: np.ndarray, bins: int = 64) -> dict:
        if object_bgra.ndim != 3 or object_bgra.shape[2] != 4:
            raise ValueError("Expected BGRA image")

        binary = alpha_to_binary(object_bgra)
        contour = self._largest_contour(binary)
        if contour is None or len(contour) < 2:
            raise ValueError("Failed to build contour for saliency profile")

        p0, p1 = self._farthest_pair_on_hull(contour)
        pts = self._binary_points(binary)
        if len(pts) == 0:
            raise ValueError("Empty claymore mask for saliency profile")

        top, axis_unit, axis_len = self._axis_geometry(p0, p1)
        rel = pts - top[None, :]
        proj = rel @ axis_unit
        perp = np.array([-axis_unit[1], axis_unit[0]], dtype=np.float32)
        side = rel @ perp

        bgr = object_bgra[:, :, :3]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
        gmag = cv2.magnitude(gx, gy)

        ys, xs = np.where(binary > 0)
        norm_proj = np.clip(proj / max(axis_len, 1e-6), 0.0, 0.999999)
        bin_ids = np.minimum((norm_proj * bins).astype(np.int32), bins - 1)

        widths = np.zeros(bins, dtype=np.float32)
        masses = np.zeros(bins, dtype=np.float32)
        detail = np.zeros(bins, dtype=np.float32)

        for idx in range(bins):
            mask = bin_ids == idx
            masses[idx] = float(mask.sum())
            if not np.any(mask):
                continue

            slice_side = side[mask]
            lo = float(np.percentile(slice_side, 5))
            hi = float(np.percentile(slice_side, 95))
            widths[idx] = max(0.0, hi - lo)
            detail[idx] = float(np.mean(gmag[ys[mask], xs[mask]]))

        nonzero = masses > 0
        width_cv = 0.0
        width_peakiness = 0.0
        if np.any(nonzero):
            w = widths[nonzero]
            width_cv = float(np.std(w) / max(np.mean(w), 1e-6))
            width_peakiness = float(np.max(w) / max(np.mean(w), 1e-6))

        masses_n = masses / max(masses.sum(), 1e-6)
        widths_n = widths / max(widths.max(), 1e-6)
        detail_n = detail / max(detail.max(), 1e-6)
        dw = np.abs(np.gradient(widths_n))
        dd = np.abs(np.gradient(detail_n))

        sal = (
            0.28 * widths_n
            + 0.16 * np.sqrt(masses_n / max(masses_n.max(), 1e-6))
            + 0.18 * (dw / max(dw.max(), 1e-6))
            + 0.24 * detail_n
            + 0.14 * (dd / max(dd.max(), 1e-6))
        )
        sal = np.convolve(sal, np.array([0.2, 0.6, 0.2], dtype=np.float32), mode="same")
        sal_n = sal / max(sal.sum(), 1e-6)

        pos = np.arange(bins, dtype=np.float32) / max(bins - 1, 1)
        hero_center = float(np.sum(pos * sal_n))
        cdf = np.cumsum(sal_n)
        left = float(pos[np.searchsorted(cdf, 0.15)])
        right = float(pos[min(len(pos) - 1, int(np.searchsorted(cdf, 0.85)))])
        hero_span = max(0.08, right - left)
        mass_center = float(np.sum(pos * masses_n))

        return {
            "hero_center": hero_center,
            "hero_span": hero_span,
            "mass_center": mass_center,
            "width_cv": width_cv,
            "width_peakiness": width_peakiness,
        }

    def reframe_window_autonomous(self, object_bgra: np.ndarray) -> tuple[np.ndarray, dict]:
        params = self.select_window_params(object_bgra)
        out = self.reframe_window(
            object_bgra,
            target_midpoint=params["midpoint"],
            target_visible_length=float(params["length"]),
            window_center_ratio=float(params["center"]),
            window_span_ratio=float(params["span"]),
            extra_scale=1.0,
        )
        return out, params

    def reframe(self, object_bgra: np.ndarray, extra_scale: float = 1.0) -> np.ndarray:
        if object_bgra.ndim != 3 or object_bgra.shape[2] != 4:
            raise ValueError("Expected BGRA image")

        alpha = object_bgra[:, :, 3]
        binary = np.where(alpha > ALPHA_MASK_THRESHOLD, 255, 0).astype(np.uint8)

        contour = self._largest_contour(binary)
        if contour is None or len(contour) < 2:
            raise ValueError("Failed to build contour for extracted object")

        p0, p1 = self._farthest_pair_on_hull(contour)

        current_length = float(np.linalg.norm(p1 - p0))
        if current_length <= 1e-6:
            raise ValueError("Degenerate claymore length")

        current_top_x = self._top_endpoint_x(p0, p1)
        ys, _ = np.where(binary > 0)
        if len(ys) == 0:
            raise ValueError("Binary mask is empty after contour extraction")
        current_max_y = float(ys.max())

        scale = self.TARGET_CLAYMORE_LENGTH / current_length
        scale *= float(extra_scale)
        scale = float(np.clip(scale, self.MIN_SCALE, self.MAX_SCALE))

        tx = float(self.TARGET_TOP_ENDPOINT_X - scale * current_top_x)
        ty = float(self.TARGET_BOTTOM_Y - scale * current_max_y)

        matrix = np.array(
            [
                [scale, 0.0, tx],
                [0.0, scale, ty],
            ],
            dtype=np.float32,
        )

        out = cv2.warpAffine(
            object_bgra,
            matrix,
            (self.TARGET_CANVAS_SIZE, self.TARGET_CANVAS_SIZE),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

        out_alpha = out[:, :, 3]
        out[out_alpha == 0, :3] = 0
        return out

    def reframe_window(
        self,
        object_bgra: np.ndarray,
        target_midpoint: np.ndarray,
        target_visible_length: float,
        window_center_ratio: float,
        window_span_ratio: float,
        extra_scale: float = 1.0,
    ) -> np.ndarray:
        if object_bgra.ndim != 3 or object_bgra.shape[2] != 4:
            raise ValueError("Expected BGRA image")

        alpha = object_bgra[:, :, 3]
        binary = np.where(alpha > ALPHA_MASK_THRESHOLD, 255, 0).astype(np.uint8)

        contour = self._largest_contour(binary)
        if contour is None or len(contour) < 2:
            raise ValueError("Failed to build contour for extracted object")

        p0, p1 = self._farthest_pair_on_hull(contour)
        top, axis_unit, axis_len = self._axis_geometry(p0, p1)

        span_ratio = float(np.clip(window_span_ratio, 0.05, 1.0))
        center_ratio = float(np.clip(window_center_ratio, span_ratio * 0.5, 1.0 - span_ratio * 0.5))

        visible_length = axis_len * span_ratio
        if visible_length <= 1e-6:
            raise ValueError("Degenerate visible claymore window")

        visible_mid = top + axis_unit * (axis_len * center_ratio)

        scale = float(target_visible_length) / float(visible_length)
        scale *= float(extra_scale)
        scale = float(np.clip(scale, self.MIN_SCALE, self.MAX_WINDOW_SCALE))

        tx = float(target_midpoint[0] - scale * visible_mid[0])
        ty = float(target_midpoint[1] - scale * visible_mid[1])

        matrix = np.array(
            [
                [scale, 0.0, tx],
                [0.0, scale, ty],
            ],
            dtype=np.float32,
        )

        out = cv2.warpAffine(
            object_bgra,
            matrix,
            (self.TARGET_CANVAS_SIZE, self.TARGET_CANVAS_SIZE),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

        out_alpha = out[:, :, 3]
        out[out_alpha == 0, :3] = 0
        return out

    def _largest_contour(self, binary: np.ndarray):
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None
        return max(contours, key=cv2.contourArea)

    def _farthest_pair_on_hull(self, contour: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hull = cv2.convexHull(contour)
        points = hull[:, 0, :].astype(np.float32)

        if len(points) < 2:
            raise ValueError("Hull has fewer than 2 points")

        diff = points[:, None, :] - points[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        i, j = np.unravel_index(np.argmax(dist2), dist2.shape)

        return points[i], points[j]

    def _binary_points(self, binary: np.ndarray) -> np.ndarray:
        ys, xs = np.where(binary > 0)
        if len(xs) == 0:
            return np.empty((0, 2), dtype=np.float32)
        return np.stack(
            [xs.astype(np.float32), ys.astype(np.float32)],
            axis=1,
        )

    def _axis_geometry(
        self,
        p0: np.ndarray,
        p1: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        top = p0 if p0[1] < p1[1] else p1
        bottom = p1 if p0[1] < p1[1] else p0
        axis = bottom - top
        axis_len = float(np.linalg.norm(axis))
        if axis_len <= 1e-6:
            raise ValueError("Degenerate claymore axis")
        axis_unit = axis / axis_len
        return top, axis_unit, axis_len

    def _top_endpoint_x(self, p0: np.ndarray, p1: np.ndarray) -> float:
        top = p0 if p0[1] < p1[1] else p1
        return float(top[0])

    def _peak_width_ratio(
        self,
        proj: np.ndarray,
        side: np.ndarray,
        axis_len: float,
        bins: int = 48,
    ) -> float:
        norm_proj = np.clip(proj / max(axis_len, 1e-6), 0.0, 0.999999)
        bin_ids = np.minimum((norm_proj * bins).astype(np.int32), bins - 1)
        widths = np.zeros(bins, dtype=np.float32)

        for idx in range(bins):
            mask = bin_ids == idx
            if not np.any(mask):
                continue
            slice_side = side[mask]
            lo = float(np.percentile(slice_side, 5))
            hi = float(np.percentile(slice_side, 95))
            widths[idx] = max(0.0, hi - lo)

        return float(np.argmax(widths) / max(bins - 1, 1))

    def _geometry_features(self, object_bgra: np.ndarray) -> dict:
        if object_bgra.ndim != 3 or object_bgra.shape[2] != 4:
            raise ValueError("Expected BGRA image")

        binary = alpha_to_binary(object_bgra)
        contour = self._largest_contour(binary)
        if contour is None or len(contour) < 2:
            raise ValueError("Failed to build contour for reference")

        p0, p1 = self._farthest_pair_on_hull(contour)
        pts = self._binary_points(binary)
        if len(pts) == 0:
            raise ValueError("Reference binary mask is empty")

        top, axis_unit, axis_len = self._axis_geometry(p0, p1)
        rel = pts - top[None, :]
        proj = rel @ axis_unit
        perp = np.array([-axis_unit[1], axis_unit[0]], dtype=np.float32)
        side = rel @ perp

        top35_ratio = float(np.mean(proj <= axis_len * 0.35))
        top45_ratio = float(np.mean(proj <= axis_len * 0.45))
        top55_ratio = float(np.mean(proj <= axis_len * 0.55))
        bot35_ratio = float(np.mean(proj >= axis_len * 0.65))
        bot45_ratio = float(np.mean(proj >= axis_len * 0.55))
        width_like = float(np.std(side) * 2.0)
        width_like = max(width_like, 1e-6)
        compactness = float(len(pts)) / float(axis_len * width_like)
        peak_width_ratio = self._peak_width_ratio(proj, side, axis_len)

        return {
            "top35_ratio": top35_ratio,
            "top45_ratio": top45_ratio,
            "top55_ratio": top55_ratio,
            "bot35_ratio": bot35_ratio,
            "bot45_ratio": bot45_ratio,
            "width_like": width_like,
            "compactness": compactness,
            "peak_width_ratio": peak_width_ratio,
        }

    def _blend_from_prototypes(
        self,
        feature_vec: np.ndarray,
        proto_ids: list[str],
        label: str,
    ) -> dict:
        rows = []
        for proto_id in proto_ids:
            proto = self.WINDOW_PROTOTYPES[proto_id]
            proto_feat = np.array(proto["features"], dtype=np.float32)
            dist = np.linalg.norm((feature_vec - proto_feat) / self.PROTOTYPE_FEATURE_SCALES)
            rows.append((float(dist), proto_id, proto))

        rows.sort(key=lambda item: item[0])
        selected = rows[: min(self.PROTOTYPE_K, len(rows))]

        distances = np.array([row[0] for row in selected], dtype=np.float32)
        d0 = float(distances.min()) if len(distances) > 0 else 0.0
        rel_distances = np.maximum(distances - d0, 0.0)
        weights = np.exp(-0.5 * (rel_distances / self.PROTOTYPE_SIGMA) ** 2)
        weights = weights / np.maximum(weights.sum(), 1e-6)

        center = 0.0
        span = 0.0
        length = 0.0
        midpoint = np.zeros(2, dtype=np.float32)
        family_parts: list[str] = []

        for weight, (_, proto_id, proto) in zip(weights.tolist(), selected):
            center += weight * float(proto["center"])
            span += weight * float(proto["span"])
            length += weight * float(proto["length"])
            midpoint += weight * np.array(proto["midpoint"], dtype=np.float32)
            family_parts.append(f"{proto_id}:{weight:.2f}")

        return {
            "family": label + "[" + ", ".join(family_parts) + "]",
            "center": float(center),
            "span": float(span),
            "length": float(length),
            "midpoint": midpoint.astype(np.float32),
            "max_weight": float(weights.max()) if len(weights) else 1.0,
        }


class PolearmReferenceReframer(SwordReferenceReframer):
    TARGET_SWORD_LENGTH = 292.0
    TARGET_TOP_ENDPOINT_X = 222.0
    TARGET_BOTTOM_Y = 255.0

    MIN_SCALE = 0.30
    MAX_SCALE = 1.60

    MIN_CUT_RATIO = 0.00
    MAX_CUT_RATIO = 0.30

    PROTOTYPE_FEATURE_SCALES = np.array([0.05, 0.06, 0.05, 8.0, 0.12], dtype=np.float32)
    PROTOTYPE_SIGMA = 0.35
    PROTOTYPE_K = 4

    PROTOTYPES = {
        "13511": {"features": [0.439907, 0.604186, 0.141014, 28.664707, 1.243745], "cut": 0.22, "length": 280.0, "top_x": 216.0},
        "13501": {"features": [0.406259, 0.623069, 0.120657, 36.307171, 0.939473], "cut": 0.16, "length": 296.0, "top_x": 222.0},
        "13514": {"features": [0.323273, 0.442837, 0.191236, 45.957703, 0.878663], "cut": 0.02, "length": 296.0, "top_x": 228.0},
        "13426": {"features": [0.624539, 0.727357, 0.114810, 57.364456, 0.845045], "cut": 0.00, "length": 256.0, "top_x": 210.0},
        "13414": {"features": [0.288578, 0.395582, 0.169720, 40.011082, 0.773317], "cut": 0.20, "length": 304.0, "top_x": 222.0},
        "13415": {"features": [0.316499, 0.490834, 0.180405, 36.570786, 0.775634], "cut": 0.00, "length": 280.0, "top_x": 230.0},
        "13507": {"features": [0.295904, 0.544451, 0.170976, 27.466476, 0.834979], "cut": 0.28, "length": 296.0, "top_x": 234.0},
        "13424": {"features": [0.383434, 0.656382, 0.164512, 34.849743, 0.754313], "cut": 0.16, "length": 288.0, "top_x": 222.0},
        "13403": {"features": [0.207275, 0.467068, 0.214808, 19.684872, 0.774870], "cut": 0.28, "length": 312.0, "top_x": 234.0},
        "13404": {"features": [0.283807, 0.547301, 0.192590, 25.818962, 0.681046], "cut": 0.24, "length": 296.0, "top_x": 228.0},
        "13402": {"features": [0.242189, 0.334719, 0.249028, 34.518402, 0.721668], "cut": 0.04, "length": 296.0, "top_x": 228.0},
        "13416": {"features": [0.345537, 0.461264, 0.245852, 20.693382, 1.142021], "cut": 0.28, "length": 264.0, "top_x": 234.0},
        "13509": {"features": [0.295957, 0.375990, 0.248020, 29.067814, 0.790845], "cut": 0.02, "length": 288.0, "top_x": 234.0},
        "13406": {"features": [0.387725, 0.546645, 0.183961, 16.868135, 1.201162], "cut": 0.28, "length": 312.0, "top_x": 234.0},
        "13405": {"features": [0.389880, 0.654380, 0.149690, 29.563349, 0.846707], "cut": 0.28, "length": 296.0, "top_x": 228.0},
        "13431": {"features": [0.308500, 0.468002, 0.129824, 36.812336, 0.739592], "cut": 0.10, "length": 300.0, "top_x": 230.0},
        "13515": {"features": [0.478033, 0.665133, 0.112027, 64.990166, 1.019545], "cut": 0.055, "length": 294.0, "top_x": 216.0},
    }

    def _feature_vec(self, object_bgra: np.ndarray) -> np.ndarray:
        features = self._geometry_features(object_bgra)
        return np.array(
            [
                float(features["top35_ratio"]),
                float(features["top45_ratio"]),
                float(features["bot35_ratio"]),
                float(features["width_like"]),
                float(features["compactness"]),
            ],
            dtype=np.float32,
        )

    def _blend_from_prototypes(self, feature_vec: np.ndarray) -> dict:
        rows = []
        for proto_id, proto in self.PROTOTYPES.items():
            proto_feat = np.array(proto["features"], dtype=np.float32)
            dist = np.linalg.norm((feature_vec - proto_feat) / self.PROTOTYPE_FEATURE_SCALES)
            rows.append((float(dist), proto_id, proto))

        rows.sort(key=lambda item: item[0])
        selected = rows[: min(self.PROTOTYPE_K, len(rows))]

        distances = np.array([row[0] for row in selected], dtype=np.float32)
        d0 = float(distances.min()) if len(distances) > 0 else 0.0
        rel_distances = np.maximum(distances - d0, 0.0)
        weights = np.exp(-0.5 * (rel_distances / self.PROTOTYPE_SIGMA) ** 2)
        weights = weights / np.maximum(weights.sum(), 1e-6)

        cut = 0.0
        length = 0.0
        top_x = 0.0
        family_parts: list[str] = []
        for weight, (_, proto_id, proto) in zip(weights.tolist(), selected):
            cut += weight * float(proto["cut"])
            length += weight * float(proto["length"])
            top_x += weight * float(proto["top_x"])
            family_parts.append(f"{proto_id}:{weight:.2f}")

        return {
            "family": "polearm[" + ", ".join(family_parts) + "]",
            "cut": float(cut),
            "length": float(length),
            "top_x": float(top_x),
            "max_weight": float(weights.max()) if len(weights) else 1.0,
        }

    def _apply_guards(self, feature_vec: np.ndarray, params: dict) -> dict:
        top35_ratio, top45_ratio, bot35_ratio, width_like, compactness = [float(v) for v in feature_vec.tolist()]

        # Very broad / full-bodied polearms are usually already readable and
        # should not be aggressively shortened.
        if width_like >= 50.0 and bot35_ratio >= 0.21 and compactness <= 0.82:
            params["cut"] = min(float(params["cut"]), 0.02)
            params["family"] += ", full_broad_guard"

        # Extremely broad upper-heavy heads also tend to be shown more fully.
        if width_like >= 55.0 and top45_ratio >= 0.60 and bot35_ratio <= 0.13:
            params["cut"] = min(float(params["cut"]), 0.05)
            params["length"] = min(float(params["length"]), 294.0)
            params["top_x"] = min(float(params["top_x"]), 216.0)
            params["family"] += ", top_heavy_full_guard"

        params["cut"] = float(np.clip(float(params["cut"]), self.MIN_CUT_RATIO, self.MAX_CUT_RATIO))
        return params

    def _inferred_params(self, object_bgra: np.ndarray) -> dict:
        feature_vec = self._feature_vec(object_bgra)
        params = self._blend_from_prototypes(feature_vec)
        params = self._apply_guards(feature_vec, params)
        return params

    def select_ref_cut_ratio(self, object_bgra: np.ndarray) -> float:
        return float(self._inferred_params(object_bgra)["cut"])

    def reframe(self, object_bgra: np.ndarray, extra_scale: float = 1.0) -> np.ndarray:
        if object_bgra.ndim != 3 or object_bgra.shape[2] != 4:
            raise ValueError("Expected BGRA image")

        alpha = object_bgra[:, :, 3]
        binary = np.where(alpha > ALPHA_MASK_THRESHOLD, 255, 0).astype(np.uint8)

        contour = self._largest_contour(binary)
        if contour is None or len(contour) < 2:
            raise ValueError("Failed to build contour for extracted object")

        p0, p1 = self._farthest_pair_on_hull(contour)
        current_length = float(np.linalg.norm(p1 - p0))
        if current_length <= 1e-6:
            raise ValueError("Degenerate polearm length")

        params = self._inferred_params(object_bgra)
        target_length = float(params["length"])
        target_top_x = float(params["top_x"])
        target_bottom_y = self.TARGET_BOTTOM_Y

        current_top_x = self._top_endpoint_x(p0, p1)
        ys, _ = np.where(binary > 0)
        if len(ys) == 0:
            raise ValueError("Binary mask is empty after contour extraction")
        current_max_y = float(ys.max())

        scale = target_length / current_length
        scale *= float(extra_scale)
        scale = float(np.clip(scale, self.MIN_SCALE, self.MAX_SCALE))

        tx = float(target_top_x - scale * current_top_x)
        ty = float(target_bottom_y - scale * current_max_y)

        matrix = np.array(
            [
                [scale, 0.0, tx],
                [0.0, scale, ty],
            ],
            dtype=np.float32,
        )

        out = cv2.warpAffine(
            object_bgra,
            matrix,
            (self.TARGET_CANVAS_SIZE, self.TARGET_CANVAS_SIZE),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

        out_alpha = out[:, :, 3]
        out[out_alpha == 0, :3] = 0
        return out


def run_sword_reframe_debug():
    reset_dir(OUT_DIR)

    crop_files = sorted(CROP_DIR.glob("*.png"))
    ref_files = load_all_sword_ref_paths()

    if not crop_files and not ref_files:
        raise FileNotFoundError(f"No PNG files found in {CROP_DIR} and {ALL_REF_DIR}")

    reframer = SwordReferenceReframer()

    print(f"[INFO] crops: {len(crop_files)}")
    print(f"[INFO] refs:  {len(ref_files)}")
    print(f"[OUT]  {OUT_DIR}")
    print(f"[SRC]  sword refs from {ALL_REF_DIR}")
    print("[CFG]  REF_CUT_MODE=CALIBRATED_GEOMETRY_TREE")
    print(f"[CFG]  REF_EXTRA_SCALE={REF_EXTRA_SCALE:.3f}")

    for path in crop_files:
        print(f"[CROP] {path.name}")

        try:
            _, reframed, meta = load_compensated_crop(path)
            comp = meta["compensation"]
            cuts = meta["applied_cuts"]
            print(
                f"[CROPWIN] {path.name} -> "
                f"L{cuts.get('cut_left', 0)} R{cuts.get('cut_right', 0)} "
                f"T{cuts.get('cut_top', 0)} B{cuts.get('cut_bottom', 0)} "
                f"scale={comp['scale']:.3f}"
            )
        except Exception as exc:
            print(f"[WARN] skip crop {path.name}: {exc}")
            continue

        save_img(OUT_DIR / path.name, reframed)

    for path in ref_files:
        print(f"[REF]  {path.name}")

        try:
            ref_bgra = load_bgra_from_path(path)
            ref_cut_ratio = reframer.select_ref_cut_ratio(ref_bgra)
            print(f"[CUT]  {path.name} -> {ref_cut_ratio:.3f}")
            ref_cut = cut_ref_bottom_percent(ref_bgra, ref_cut_ratio)
            reframed_ref = reframer.reframe(ref_cut, extra_scale=REF_EXTRA_SCALE)
        except Exception as exc:
            print(f"[WARN] skip ref {path.name}: {exc}")
            continue

        save_img(OUT_DIR / path.name, reframed_ref)

    print("[DONE]")


def run_claymore_reframe_debug():
    reset_dir(CLAYMORE_OUT_DIR)

    crop_files = sorted(CLAYMORE_CROP_DIR.glob("*.png"))
    ref_files = load_all_claymore_ref_paths()

    if not crop_files and not ref_files:
        raise FileNotFoundError(
            f"No PNG files found in {CLAYMORE_CROP_DIR} and {ALL_REF_DIR}"
        )

    reframer = ClaymoreReferenceReframer()

    print(f"[INFO] crops: {len(crop_files)}")
    print(f"[INFO] refs:  {len(ref_files)}")
    print(f"[OUT]  {CLAYMORE_OUT_DIR}")
    print(f"[SRC]  claymore refs from {ALL_REF_DIR}")
    print("[CFG]  REF_MODE=CLAYMORE_WINDOW_AUTONOMOUS")

    for path in crop_files:
        print(f"[CROP] {path.name}")

        try:
            _, reframed, meta = load_compensated_claymore_crop(path)
            comp = meta["compensation"]
            cuts = meta["applied_cuts"]
            print(
                f"[CROPWIN] {path.name} -> "
                f"L{cuts.get('cut_left', 0)} R{cuts.get('cut_right', 0)} "
                f"T{cuts.get('cut_top', 0)} B{cuts.get('cut_bottom', 0)} "
                f"scale={comp['scale']:.3f}"
            )
        except Exception as exc:
            print(f"[WARN] skip crop {path.name}: {exc}")
            continue

        save_img(CLAYMORE_OUT_DIR / path.name, reframed)

    for path in ref_files:
        print(f"[REF]  {path.name}")

        try:
            ref_bgra = load_bgra_from_path(path)
            reframed_ref, params = reframer.reframe_window_autonomous(ref_bgra)
            print(
                f"[MODEL] {path.name} -> {params['family']} "
                f"center={params['center']:.3f} span={params['span']:.3f}"
            )
        except Exception as exc:
            print(f"[WARN] skip ref {path.name}: {exc}")
            continue

        save_img(CLAYMORE_OUT_DIR / path.name, reframed_ref)

    print("[DONE]")


def run_polearm_reframe_debug():
    reset_dir(POLEARM_OUT_DIR)

    crop_files = sorted(POLEARM_CROP_DIR.glob("*.png"))
    ref_files = load_all_polearm_ref_paths()

    if not crop_files and not ref_files:
        raise FileNotFoundError(
            f"No PNG files found in {POLEARM_CROP_DIR} and {ALL_REF_DIR}"
        )

    reframer = PolearmReferenceReframer()

    print(f"[INFO] crops: {len(crop_files)}")
    print(f"[INFO] refs:  {len(ref_files)}")
    print(f"[OUT]  {POLEARM_OUT_DIR}")
    print(f"[SRC]  polearm refs from {ALL_REF_DIR}")
    print("[CFG]  REF_MODE=POLEARM_SWORD_LIKE")
    print(f"[CFG]  REF_EXTRA_SCALE={REF_EXTRA_SCALE:.3f}")

    for path in crop_files:
        print(f"[CROP] {path.name}")

        try:
            _, reframed, meta = load_compensated_crop(path)
            comp = meta["compensation"]
            cuts = meta["applied_cuts"]
            print(
                f"[CROPWIN] {path.name} -> "
                f"L{cuts.get('cut_left', 0)} R{cuts.get('cut_right', 0)} "
                f"T{cuts.get('cut_top', 0)} B{cuts.get('cut_bottom', 0)} "
                f"scale={comp['scale']:.3f}"
            )
        except Exception as exc:
            print(f"[WARN] skip crop {path.name}: {exc}")
            continue

        save_img(POLEARM_OUT_DIR / path.name, reframed)

    for path in ref_files:
        print(f"[REF]  {path.name}")

        try:
            ref_bgra = load_bgra_from_path(path)
            ref_cut_ratio = reframer.select_ref_cut_ratio(ref_bgra)
            print(f"[CUT]  {path.name} -> {ref_cut_ratio:.3f}")
            ref_cut = cut_ref_bottom_percent(ref_bgra, ref_cut_ratio)
            reframed_ref = reframer.reframe(ref_cut, extra_scale=REF_EXTRA_SCALE)
        except Exception as exc:
            print(f"[WARN] skip ref {path.name}: {exc}")
            continue

        save_img(POLEARM_OUT_DIR / path.name, reframed_ref)

    print("[DONE]")


if __name__ == "__main__":
    run_polearm_reframe_debug()
