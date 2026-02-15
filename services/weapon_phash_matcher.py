import os
import json
import shutil
import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _p(*parts: str) -> str:
    return os.path.join(ROOT, *parts)


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _alpha_composite_to_bg(img_rgba: np.ndarray, bg_bgr: tuple[int, int, int]) -> np.ndarray:
    if img_rgba.shape[2] != 4:
        return img_rgba
    b, g, r, a = cv2.split(img_rgba)
    alpha = a.astype(np.float32) / 255.0

    bg_b = np.full_like(b, bg_bgr[0], dtype=np.uint8)
    bg_g = np.full_like(g, bg_bgr[1], dtype=np.uint8)
    bg_r = np.full_like(r, bg_bgr[2], dtype=np.uint8)

    out_b = (b.astype(np.float32) * alpha + bg_b.astype(np.float32) * (1 - alpha)).astype(np.uint8)
    out_g = (g.astype(np.float32) * alpha + bg_g.astype(np.float32) * (1 - alpha)).astype(np.uint8)
    out_r = (r.astype(np.float32) * alpha + bg_r.astype(np.float32) * (1 - alpha)).astype(np.uint8)
    return cv2.merge([out_b, out_g, out_r])


# простые фоны "на глаз". Они почти не важны, потому что редкость мы фильтруем по кропу.
RARITY_BG = {
    1: (120, 120, 120),
    2: (120, 150, 110),
    3: (170, 130, 90),
    4: (160, 110, 160),
    5: (80, 140, 190),
}


def _render_weapon_ref(ref_png_path: str, rarity: int, size: int = 64) -> np.ndarray | None:
    img = cv2.imread(ref_png_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    bg = RARITY_BG.get(int(rarity), (128, 128, 128))
    canvas = np.full((size, size, 3), bg, dtype=np.uint8)

    if len(img.shape) == 3 and img.shape[2] == 4:
        img = _alpha_composite_to_bg(img, bg)
    elif len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    h, w = img.shape[:2]
    if h < 2 or w < 2:
        return None

    target = int(size * 0.86)
    scale = min(target / float(w), target / float(h))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    img2 = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    x = (size - nw) // 2
    y = (size - nh) // 2
    canvas[y:y + nh, x:x + nw] = img2
    return canvas


def _phash_64(img_bgr: np.ndarray, hash_size: int = 8, highfreq_factor: int = 4) -> np.ndarray:
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # без edges — на твоих данных так стабильнее после фильтра type+rarity
    size = hash_size * highfreq_factor  # 32
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32)

    dct = cv2.dct(img)
    dct_low = dct[:hash_size, :hash_size]

    med = np.median(dct_low[1:, 1:])
    bits = (dct_low > med).astype(np.uint8)
    return bits


def _hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def build_weapon_phash_index(
    data_dir: str = "data",
    cache_weapons_dir: str = "cache/enka_ref_weapons",
    out_ref_dir: str = "cache/ref_icons/weapons_64",
    out_index_path: str = "cache/ref_index/weapons_phash_64.json",
    size: int = 64,
    force: bool = False,
) -> dict:
    weapons = _load_json(_p(data_dir, "weapons.json"))
    cache_dir = _p(cache_weapons_dir)
    ref_dir = _p(out_ref_dir)
    idx_path = _p(out_index_path)

    _ensure_dir(ref_dir)
    _ensure_dir(os.path.dirname(idx_path))

    if os.path.exists(idx_path) and not force:
        return {"status": "exists", "index": idx_path}

    index: dict[str, str] = {}
    rendered = 0
    skipped = 0

    for wid, meta in weapons.items():
        ref_png = os.path.join(cache_dir, f"{wid}.png")
        if not os.path.exists(ref_png):
            skipped += 1
            continue

        rarity = int(meta.get("rarity", 1))
        out_png = os.path.join(ref_dir, f"{wid}.png")

        if (not force) and os.path.exists(out_png):
            img = cv2.imread(out_png, cv2.IMREAD_COLOR)
            if img is None:
                skipped += 1
                continue
        else:
            img = _render_weapon_ref(ref_png, rarity, size=size)
            if img is None:
                skipped += 1
                continue
            cv2.imwrite(out_png, img)
            rendered += 1

        bits = _phash_64(img)
        index[str(wid)] = "".join("1" if x else "0" for x in bits.flatten().tolist())

    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump({"size": size, "index": index}, f, ensure_ascii=False, indent=2)

    return {"status": "built", "rendered": rendered, "skipped": skipped, "index": idx_path}


def load_weapon_phash_index_bits(index_path: str = "cache/ref_index/weapons_phash_64.json") -> dict[str, np.ndarray]:
    idx_path = _p(index_path)
    if not os.path.exists(idx_path):
        return {}
    with open(idx_path, "r", encoding="utf-8") as f:
        blob = json.load(f)

    raw_index: dict[str, str] = blob.get("index", {})
    index_bits: dict[str, np.ndarray] = {}
    for wid, bitstr in raw_index.items():
        arr = np.array([1 if ch == "1" else 0 for ch in bitstr], dtype=np.uint8).reshape(8, 8)
        index_bits[str(wid)] = arr
    return index_bits


def match_weapon_crop_phash_filtered(
    crop_path: str,
    index_bits: dict[str, np.ndarray],
    candidate_ids: list[str],
    size: int = 64,
) -> tuple[str | None, int, int]:
    crop = cv2.imread(crop_path, cv2.IMREAD_COLOR)
    if crop is None:
        return None, 999, 999

    # режем рамку
    h, w = crop.shape[:2]
    p = int(min(h, w) * 0.18)
    if p > 0 and h - 2 * p >= 10 and w - 2 * p >= 10:
        crop = crop[p:h - p, p:w - p]

    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    cbits = _phash_64(crop)

    best_id = None
    best = 999
    second = 999

    for wid in candidate_ids:
        dbits = index_bits.get(str(wid))
        if dbits is None:
            continue
        d = _hamming(cbits, dbits)
        if d < best:
            second = best
            best = d
            best_id = str(wid)
        elif d < second:
            second = d

    return best_id, best, second


def detect_weapon_rarity_from_crop(crop_bgr: np.ndarray) -> int | None:
    h, w = crop_bgr.shape[:2]
    if h < 10 or w < 10:
        return None

    p = int(min(h, w) * 0.18)
    if p > 0 and h - 2 * p >= 10 and w - 2 * p >= 10:
        img = crop_bgr[p:h - p, p:w - p]
    else:
        img = crop_bgr

    hh, ww = img.shape[:2]
    y1, y2 = int(hh * 0.25), int(hh * 0.45)
    x1, x2 = int(ww * 0.25), int(ww * 0.45)
    patch = img[y1:y2, x1:x2]
    if patch.size == 0:
        return None

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    mask = V > 30
    if np.count_nonzero(mask) < 10:
        return None

    Hm = float(np.median(H[mask]))
    Sm = float(np.median(S[mask]))

    if Sm < 35:
        return 1

    if 10 <= Hm <= 40 and Sm > 80:
        return 5

    if 125 <= Hm <= 170:
        return 4

    if 90 <= Hm < 125:
        return 3

    if 40 <= Hm < 90:
        return 2

    return None


def copy_weapon_hd_from_cache(best_id: str, out_hd_weap_dir: str = "assets/hd/weapons", cache_dir: str = "cache/enka_ref_weapons") -> bool:
    src = _p(cache_dir, f"{best_id}.png")
    dst = _p(out_hd_weap_dir, f"{best_id}.png")
    _ensure_dir(os.path.dirname(dst))
    if not os.path.exists(src):
        return False
    if os.path.exists(dst):
        return True
    try:
        shutil.copyfile(src, dst)
        return True
    except Exception:
        return False

