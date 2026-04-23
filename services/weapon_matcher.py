import os
import json
import shutil
import torch
import timm
from PIL import Image
from dataclasses import dataclass

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _p(*parts: str) -> str:
    return os.path.join(ROOT, *parts)


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _load_image(path: str, flags=cv2.IMREAD_COLOR):
    return cv2.imread(path, flags)



def _alpha_composite_to_bg(img_rgba: np.ndarray, bg_bgr: tuple[int, int, int]) -> np.ndarray:
    if img_rgba is None:
        return None
    if len(img_rgba.shape) == 2:
        return cv2.cvtColor(img_rgba, cv2.COLOR_GRAY2BGR)
    if img_rgba.shape[2] != 4:
        return img_rgba

    b, g, r, a = cv2.split(img_rgba)
    alpha = a.astype(np.float32) / 255.0

    bg_b = np.full_like(b, bg_bgr[0], dtype=np.uint8)
    bg_g = np.full_like(g, bg_bgr[1], dtype=np.uint8)
    bg_r = np.full_like(r, bg_bgr[2], dtype=np.uint8)

    out_b = (b.astype(np.float32) * alpha + bg_b.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
    out_g = (g.astype(np.float32) * alpha + bg_g.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
    out_r = (r.astype(np.float32) * alpha + bg_r.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
    return cv2.merge([out_b, out_g, out_r])

REF_VIEW_PARAMS = [
    {"scale": 0.76, "dx": 0.00, "dy": 0.00},
    {"scale": 0.84, "dx": 0.00, "dy": 0.00},
    {"scale": 0.92, "dx": 0.00, "dy": 0.00},
    {"scale": 0.84, "dx": -0.03, "dy": 0.00},
    {"scale": 0.84, "dx": 0.03, "dy": 0.00},
    {"scale": 0.84, "dx": 0.00, "dy": -0.03},
    {"scale": 0.84, "dx": 0.00, "dy": 0.03},
]


@dataclass
class MatchDecision:
    best_id: str | None
    top1: float
    top2: float
    accepted: bool
    method: str
    extra: dict


_DEVICE = None
_DINO = None
_DINO_TRANSFORM = None


def _get_dino():
    global _DEVICE, _DINO, _DINO_TRANSFORM
    if _DINO is not None:
        return _DINO, _DINO_TRANSFORM, _DEVICE

    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    _DINO = timm.create_model(
        "vit_base_patch14_dinov2.lvd142m",
        pretrained=True,
        num_classes=0,
    )
    _DINO.eval()
    _DINO.to(_DEVICE)

    data_cfg = timm.data.resolve_model_data_config(_DINO)
    _DINO_TRANSFORM = timm.data.create_transform(**data_cfg, is_training=False)

    print(f"[DINO] device={_DEVICE}")
    return _DINO, _DINO_TRANSFORM, _DEVICE


def _l2norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 1e-12:
        return v
    return v / n


def _encode_bgr_dino(img_bgr: np.ndarray) -> np.ndarray:
    model, transform, device = _get_dino()

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    x = transform(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model(x)

    if isinstance(feat, (tuple, list)):
        feat = feat[0]

    emb = feat.squeeze(0).detach().cpu().numpy().astype(np.float32)
    return _l2norm(emb)

def build_weapon_embedding_index(
    weaps_db: dict,
    cache_weapons_dir: str = "cache/enka_ref_weapons",
    out_index_path: str = "cache/ref_index/weapons_dino_index.npz",
    size: int = 224,
    force: bool = False,
) -> dict:
    out_path = _p(out_index_path)
    _ensure_dir(os.path.dirname(out_path))

    if os.path.exists(out_path) and not force:
        return {"status": "exists", "index": out_path}

    embs = []
    weapon_ids = []
    rarities = []
    view_names = []

    total_refs = 0
    skipped_missing_png = 0
    skipped_bad_ref = 0
    skipped_encode_error = 0

    items = list(weaps_db.items())
    total = len(items)

    for i, (wid, meta) in enumerate(items):
        if i % 5 == 0:
            print(f"[INDEX] {i}/{total}")
        ref_path = _p(cache_weapons_dir, f"{wid}.png")
        if not os.path.exists(ref_path):
            skipped_missing_png += 1
            continue

        rarity = meta.get("rarity", 1)
        views = _make_ref_views(ref_path, rarity=rarity, size=size)
        if not views:
            skipped_bad_ref += 1
            continue

        total_refs += 1
        print(f"   ↳ weapon {wid} views: {len(views)}")

        for j, view_bgr in enumerate(views):
            try:
                emb = _encode_bgr_dino(view_bgr)
            except Exception:
                skipped_encode_error += 1
                continue

            embs.append(emb.astype(np.float32))
            weapon_ids.append(str(wid))
            rarities.append(int(rarity))
            view_names.append(f"view_{j}")

    if not embs:
        return {
            "status": "empty",
            "index": out_path,
            "total_refs": total_refs,
            "skipped_missing_png": skipped_missing_png,
            "skipped_bad_ref": skipped_bad_ref,
            "skipped_encode_error": skipped_encode_error,
        }

    emb_matrix = np.stack(embs, axis=0).astype(np.float32)
    weapon_ids_arr = np.array(weapon_ids, dtype="<U32")
    rarities_arr = np.array(rarities, dtype=np.int16)
    view_names_arr = np.array(view_names, dtype="<U16")

    np.savez_compressed(
        out_path,
        embeddings=emb_matrix,
        weapon_ids=weapon_ids_arr,
        rarities=rarities_arr,
        view_names=view_names_arr,
    )

    return {
        "status": "built",
        "index": out_path,
        "rows": int(len(weapon_ids)),
        "unique_weapons": int(len(set(weapon_ids))),
        "skipped_missing_png": skipped_missing_png,
        "skipped_bad_ref": skipped_bad_ref,
        "skipped_encode_error": skipped_encode_error,
    }


def load_weapon_embedding_index(
    index_path: str = "cache/ref_index/weapons_dino_index.npz",
) -> dict | None:
    path = _p(index_path)
    if not os.path.exists(path):
        return None

    blob = np.load(path, allow_pickle=False)

    embeddings = blob["embeddings"].astype(np.float32)
    weapon_ids = blob["weapon_ids"]
    rarities = blob["rarities"]
    view_names = blob["view_names"]

    return {
        "path": path,
        "embeddings": embeddings,
        "weapon_ids": weapon_ids,
        "rarities": rarities,
        "view_names": view_names,
    }


def retrieve_top_candidates(
    crop_bgr: np.ndarray,
    candidate_ids: list[str],
    emb_index: dict,
    size: int = 224,
    top_n: int = 5,
) -> list[tuple[str, float]]:
    if crop_bgr is None or not candidate_ids or not emb_index:
        return []

    crop_prep = _prepare_crop_for_match(crop_bgr, size=size)
    crop_emb = _encode_bgr_dino(crop_prep)

    ref_embs = emb_index["embeddings"]
    ref_ids = emb_index["weapon_ids"]

    candidate_set = set(str(x) for x in candidate_ids)
    mask = np.array([wid in candidate_set for wid in ref_ids], dtype=bool)
    if not np.any(mask):
        return []

    ref_embs_sub = ref_embs[mask]
    ref_ids_sub = ref_ids[mask]

    sims = ref_embs_sub @ crop_emb

    best_by_id: dict[str, float] = {}
    for wid, sim in zip(ref_ids_sub.tolist(), sims.tolist()):
        wid = str(wid)
        sim = float(sim)
        if wid not in best_by_id or sim > best_by_id[wid]:
            best_by_id[wid] = sim

    ranked = sorted(best_by_id.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]


def _make_ref_views(ref_png_path: str, rarity: int | None, size: int = 224) -> list[np.ndarray]:
    views = []

    for p in REF_VIEW_PARAMS:
        ref_view = _render_ref_view(
            ref_png_path=ref_png_path,
            rarity=rarity,
            size=size,
            scale=float(p["scale"]),
            dx=float(p["dx"]),
            dy=float(p["dy"]),
        )
        if ref_view is not None:
            views.append(ref_view)

    return views

def _render_ref_view_and_mask(
    ref_png_path: str,
    size: int = 160,
    scale: float = 0.84,
    dx: float = 0.0,
    dy: float = 0.0,
    bg_bgr: tuple[int, int, int] = (127, 127, 127),
) -> tuple[np.ndarray | None, np.ndarray | None]:
    img = _load_image(ref_png_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, None

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)

    if img.shape[2] == 4:
        alpha = img[:, :, 3]
    else:
        gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
        _, alpha = cv2.threshold(gray, 8, 255, cv2.THRESH_BINARY)
        img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2BGRA)
        img[:, :, 3] = alpha

    bbox = _alpha_bbox(img)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        img = img[y1:y2, x1:x2].copy()

    if img is None or len(img.shape) != 3 or img.shape[2] < 4:
        return None, None

    alpha = img[:, :, 3]
    bgr = _alpha_composite_to_bg(img, bg_bgr)
    if bgr is None:
        return None, None

    h, w = bgr.shape[:2]
    if h < 2 or w < 2:
        return None, None

    canvas = np.full((size, size, 3), bg_bgr, dtype=np.uint8)
    mask_canvas = np.zeros((size, size), dtype=np.uint8)

    target = int(size * scale)
    scale_k = min(target / float(w), target / float(h))
    nw, nh = max(1, int(round(w * scale_k))), max(1, int(round(h * scale_k)))

    img2 = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    alpha2 = cv2.resize(alpha, (nw, nh), interpolation=cv2.INTER_AREA)

    x = int(round((size - nw) / 2.0 + dx * size))
    y = int(round((size - nh) / 2.0 + dy * size))

    x = max(0, min(size - nw, x))
    y = max(0, min(size - nh, y))

    canvas[y:y + nh, x:x + nw] = img2
    mask_canvas[y:y + nh, x:x + nw] = alpha2

    _, mask_canvas = cv2.threshold(mask_canvas, 16, 255, cv2.THRESH_BINARY)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_canvas = cv2.morphologyEx(mask_canvas, cv2.MORPH_OPEN, k, iterations=1)
    mask_canvas = cv2.morphologyEx(mask_canvas, cv2.MORPH_CLOSE, k, iterations=1)

    return canvas, mask_canvas


def _masked_appearance_features(img_bgr: np.ndarray, mask: np.ndarray) -> dict:
    if img_bgr is None or mask is None:
        return {"valid": False, "reason": "empty"}

    if img_bgr.shape[:2] != mask.shape[:2]:
        return {"valid": False, "reason": "shape_mismatch"}

    m = (mask > 0)
    count = int(np.count_nonzero(m))
    if count < 32:
        return {"valid": False, "reason": "too_few_pixels", "count": count}

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0][m].astype(np.float32)
    S = hsv[:, :, 1][m].astype(np.float32)
    V = hsv[:, :, 2][m].astype(np.float32)

    hue_hist = cv2.calcHist([hsv], [0], mask.astype(np.uint8), [12], [0, 180]).flatten().astype(np.float32)
    hist_sum = float(hue_hist.sum())
    if hist_sum > 1e-6:
        hue_hist /= hist_sum

    return {
        "valid": True,
        "count": count,
        "sat_p50": float(np.percentile(S, 50)),
        "sat_p75": float(np.percentile(S, 75)),
        "val_p50": float(np.percentile(V, 50)),
        "val_p75": float(np.percentile(V, 75)),
        "bright_ratio": float(np.mean(V >= 200)),
        "dark_ratio": float(np.mean(V <= 70)),
        "vivid_ratio": float(np.mean((S >= 80) & (V >= 160))),
        "hue_hist": hue_hist.tolist(),
    }


def _hist_intersection(a: list[float], b: list[float]) -> float:
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    if aa.size == 0 or bb.size == 0 or aa.shape != bb.shape:
        return 0.0
    return float(np.minimum(aa, bb).sum())


def _score_closeness(a: float, b: float, scale: float) -> float:
    if scale <= 1e-6:
        return 0.0
    d = abs(float(a) - float(b)) / float(scale)
    return float(max(0.0, 1.0 - min(1.0, d)))


def _appearance_match_score(
    crop_bgr: np.ndarray,
    crop_mask: np.ndarray,
    ref_bgr: np.ndarray,
    ref_mask: np.ndarray,
) -> tuple[float, dict]:
    crop_feat = _masked_appearance_features(crop_bgr, crop_mask)
    ref_feat = _masked_appearance_features(ref_bgr, ref_mask)

    if not crop_feat.get("valid") or not ref_feat.get("valid"):
        return 0.0, {
            "appearance_score": 0.0,
            "crop_feat": crop_feat,
            "ref_feat": ref_feat,
        }

    hue_score = _hist_intersection(crop_feat["hue_hist"], ref_feat["hue_hist"])
    sat_score = 0.5 * _score_closeness(crop_feat["sat_p50"], ref_feat["sat_p50"], 80.0) + \
                0.5 * _score_closeness(crop_feat["sat_p75"], ref_feat["sat_p75"], 80.0)
    val_score = 0.5 * _score_closeness(crop_feat["val_p50"], ref_feat["val_p50"], 80.0) + \
                0.5 * _score_closeness(crop_feat["val_p75"], ref_feat["val_p75"], 80.0)
    bright_score = _score_closeness(crop_feat["bright_ratio"], ref_feat["bright_ratio"], 0.35)
    dark_score = _score_closeness(crop_feat["dark_ratio"], ref_feat["dark_ratio"], 0.35)
    vivid_score = _score_closeness(crop_feat["vivid_ratio"], ref_feat["vivid_ratio"], 0.35)
    hue_guard = max(0.0, min(1.0, (hue_score - 0.35) / 0.25))
    bright_score *= (0.35 + 0.65 * hue_guard)
    vivid_score *= (0.25 + 0.75 * hue_guard)

    appearance_score = (
            0.50 * hue_score +
            0.18 * sat_score +
            0.14 * val_score +
            0.06 * bright_score +
            0.06 * dark_score +
            0.06 * vivid_score
    )

    return float(appearance_score), {
        "appearance_score": float(appearance_score),
        "hue_score": float(hue_score),
        "sat_score": float(sat_score),
        "val_score": float(val_score),
        "bright_score": float(bright_score),
        "dark_score": float(dark_score),
        "vivid_score": float(vivid_score),
        "crop_feat": crop_feat,
        "ref_feat": ref_feat,
    }


def _score_candidate_masked(
    crop_bgr: np.ndarray,
    ref_png_path: str,
    rarity: int | None,
    size: int,
    emb_sim: float,
) -> tuple[float, dict]:
    best_align_score = -1.0
    best_ref_view = None
    best_ref_mask = None
    best_align_extra = {}
    best_view_params = None

    for p in REF_VIEW_PARAMS:
        ref_view, ref_mask = _render_ref_view_and_mask(
            ref_png_path=ref_png_path,
            size=size,
            scale=float(p["scale"]),
            dx=float(p["dx"]),
            dy=float(p["dy"]),
        )

        if ref_view is None or ref_mask is None:
            continue

        align_score, align_extra = _score_crop_vs_ref(crop_bgr, ref_view)

        if align_score > best_align_score:
            best_align_score = float(align_score)
            best_ref_view = ref_view
            best_ref_mask = ref_mask
            best_align_extra = align_extra
            best_view_params = dict(p)

    if best_ref_view is None or best_ref_mask is None:
        return 0.0, {
            "emb_sim": float(emb_sim),
            "align_score": 0.0,
            "appearance_score": 0.0,
            "reason": "no_valid_ref_view",
        }

    crop_mask = _build_crop_mask(crop_bgr)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    crop_mask = cv2.erode(crop_mask, k, iterations=1)

    appearance_score, appearance_extra = _appearance_match_score(
        crop_bgr=crop_bgr,
        crop_mask=crop_mask,
        ref_bgr=best_ref_view,
        ref_mask=best_ref_mask,
    )

    final_score = (
        0.25 * float(emb_sim) +
        0.20 * float(best_align_score) +
        0.55 * float(appearance_score)
    )

    return float(final_score), {
        "emb_sim": float(emb_sim),
        "align_score": float(best_align_score),
        "appearance_score": float(appearance_score),
        "crop_mask_area": int(np.count_nonzero(crop_mask > 0)),
        "ref_mask_area": int(np.count_nonzero(best_ref_mask > 0)),
        "view_params": best_view_params,
        **best_align_extra,
        **appearance_extra,
    }

def detect_weapon_rarity_from_crop(crop_bgr: np.ndarray) -> int | None:
    h, w = crop_bgr.shape[:2]
    if h < 20 or w < 20:
        return None

    img = crop_bgr.copy()

    p = int(min(h, w) * 0.08)
    if p > 0 and h - 2 * p >= 16 and w - 2 * p >= 16:
        img = img[p:h - p, p:w - p]

    hh, ww = img.shape[:2]
    if hh < 16 or ww < 16:
        return None

    patch_size = int(min(hh, ww) * 0.22)
    patch_size = max(8, patch_size)

    patches = [
        img[0:patch_size, 0:patch_size],
        img[0:patch_size, ww - patch_size:ww],
        img[hh - patch_size:hh, 0:patch_size],
        img[hh - patch_size:hh, ww - patch_size:ww],
    ]

    hsv_pixels = []

    for patch in patches:
        if patch.size == 0:
            continue

        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        H = hsv[:, :, 0]
        S = hsv[:, :, 1]
        V = hsv[:, :, 2]

        mask = (V > 40) & (S > 25)
        if np.count_nonzero(mask) < 6:
            continue

        hs = np.stack([H[mask], S[mask], V[mask]], axis=1)
        hsv_pixels.append(hs)

    if not hsv_pixels:
        return None

    hsv_all = np.concatenate(hsv_pixels, axis=0)
    Hm = float(np.median(hsv_all[:, 0]))
    Sm = float(np.median(hsv_all[:, 1]))
    Vm = float(np.median(hsv_all[:, 2]))

    if Sm < 35 and Vm > 60:
        return 1

    if 8 <= Hm <= 28 and Sm >= 70:
        return 5

    if 125 <= Hm <= 170 and Sm >= 45:
        return 4

    if 90 <= Hm < 125 and Sm >= 45:
        return 3

    if 40 <= Hm < 90 and Sm >= 40:
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


def build_weapon_candidate_index(weaps_db: dict) -> dict[tuple[str, int], list[str]]:
    out: dict[tuple[str, int], list[str]] = {}
    for wid, meta in weaps_db.items():
        weapon_type = meta.get("type")
        rarity = meta.get("rarity")
        if not weapon_type or rarity is None:
            continue
        key = (str(weapon_type), int(rarity))
        out.setdefault(key, []).append(str(wid))
    return out


def get_candidate_ids(
    weapon_type: str,
    rarity: int | None,
    type_rarity_to_ids: dict[tuple[str, int], list[str]],
    weaps_db: dict,
) -> list[str]:
    if rarity is not None:
        return list(type_rarity_to_ids.get((str(weapon_type), int(rarity)), []))

    out: list[str] = []
    for wid, meta in weaps_db.items():
        if str(meta.get("type")) == str(weapon_type):
            out.append(str(wid))
    return out

def _letterbox_to_square(
    img_bgr: np.ndarray,
    size: int = 224,
    bg_bgr: tuple[int, int, int] = (127, 127, 127),
) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    if h < 1 or w < 1:
        return np.full((size, size, 3), bg_bgr, dtype=np.uint8)

    scale = min(size / float(w), size / float(h))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))

    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), bg_bgr, dtype=np.uint8)

    x = (size - nw) // 2
    y = (size - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas

def _prepare_crop_for_match(crop_bgr: np.ndarray, size: int = 160) -> np.ndarray:
    return _letterbox_to_square(crop_bgr, size=size, bg_bgr=(127, 127, 127))

def _alpha_bbox(img_rgba: np.ndarray):
    if img_rgba is None or len(img_rgba.shape) != 3 or img_rgba.shape[2] < 4:
        return None

    alpha = img_rgba[:, :, 3]
    ys, xs = np.where(alpha > 8)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x1 = int(xs.min())
    x2 = int(xs.max()) + 1
    y1 = int(ys.min())
    y2 = int(ys.max()) + 1
    return x1, y1, x2, y2

def _render_ref_view(
    ref_png_path: str,
    size: int = 160,
    scale: float = 0.84,
    dx: float = 0.0,
    dy: float = 0.0,
) -> np.ndarray | None:
    view, _ = _render_ref_view_and_mask(
        ref_png_path=ref_png_path,
        size=size,
        scale=scale,
        dx=dx,
        dy=dy,
    )
    return view


def _gray(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def _edge_map(img_bgr: np.ndarray) -> np.ndarray:
    g = _gray(img_bgr)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    e = cv2.Canny(g, 60, 140)
    return e

def _build_crop_mask(img_bgr: np.ndarray, bg_bgr: tuple[int, int, int] = (127, 127, 127)) -> np.ndarray:
    if img_bgr is None:
        return None

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    diff = np.max(np.abs(img_bgr.astype(np.int16) - np.array(bg_bgr, dtype=np.int16).reshape(1, 1, 3)), axis=2)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    mask = ((diff >= 18) | (s >= 35)) & (v >= 25)
    mask = mask.astype(np.uint8) * 255

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

    if np.count_nonzero(mask) < 24:
        h, w = mask.shape[:2]
        pad = max(2, int(min(h, w) * 0.08))
        yy1, yy2 = pad, max(pad + 1, h - pad)
        xx1, xx2 = pad, max(pad + 1, w - pad)
        mask = np.zeros_like(mask)
        mask[yy1:yy2, xx1:xx2] = 255

    return mask

def _fg_mask(img_bgr: np.ndarray) -> np.ndarray:
    return _build_crop_mask(img_bgr)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = a > 0
    bb = b > 0
    inter = int(np.count_nonzero(aa & bb))
    union = int(np.count_nonzero(aa | bb))
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def _patch_score(crop_bgr: np.ndarray, ref_bgr: np.ndarray) -> float:
    h, w = crop_bgr.shape[:2]
    bands = [
        (0.05, 0.35),
        (0.35, 0.70),
        (0.70, 0.95),
    ]

    scores = []
    crop_edges = _edge_map(crop_bgr)
    ref_edges = _edge_map(ref_bgr)

    for y1r, y2r in bands:
        y1 = int(h * y1r)
        y2 = int(h * y2r)
        if y2 <= y1:
            continue
        ce = crop_edges[y1:y2, :]
        re = ref_edges[y1:y2, :]
        scores.append(_iou(ce, re))

    if not scores:
        return 0.0
    return float(np.mean(scores))

def _score_crop_vs_ref(crop_bgr: np.ndarray, ref_bgr: np.ndarray) -> tuple[float, dict]:
    crop_edges = _edge_map(crop_bgr)
    ref_edges = _edge_map(ref_bgr)
    edge = _iou(crop_edges, ref_edges)
    patch = _patch_score(crop_bgr, ref_bgr)

    score = (
        0.60 * edge +
        0.40 * patch
    )

    return float(score), {
        "edge": float(edge),
        "patch": float(patch),
    }


def match_weapon_crop(
    crop_bgr: np.ndarray,
    candidate_ids: list[str],
    weaps_db: dict,
    emb_index: dict,
    cache_weapons_dir: str = "cache/enka_ref_weapons",
    rarity: int | None = None,
    size: int = 224,
) -> MatchDecision:
    if crop_bgr is None or not candidate_ids:
        return MatchDecision(
            best_id=None,
            top1=0.0,
            top2=0.0,
            accepted=False,
            method="dino_index_masked_rerank",
            extra={"reason": "empty_input"},
        )

    top_retrieval = retrieve_top_candidates(
        crop_bgr=crop_bgr,
        candidate_ids=candidate_ids,
        emb_index=emb_index,
        size=size,
        top_n=10,
    )

    if not top_retrieval:
        return MatchDecision(
            best_id=None,
            top1=0.0,
            top2=0.0,
            accepted=False,
            method="dino_index_masked_rerank",
            extra={"reason": "no_retrieval_candidates"},
        )

    crop_prep = _prepare_crop_for_match(crop_bgr, size=size)

    retrieval_margin = 1.0
    if len(top_retrieval) >= 2:
        retrieval_margin = float(top_retrieval[0][1] - top_retrieval[1][1])

    top_n_rerank = 5 if retrieval_margin >= 0.06 else min(10, len(top_retrieval))

    reranked = []

    for wid, emb_sim in top_retrieval[:top_n_rerank]:
        ref_path = _p(cache_weapons_dir, f"{wid}.png")
        wrarity = weaps_db.get(str(wid), {}).get("rarity", rarity)

        cand_score, cand_extra = _score_candidate_masked(
            crop_bgr=crop_prep,
            ref_png_path=ref_path,
            rarity=wrarity,
            size=size,
            emb_sim=float(emb_sim),
        )

        reranked.append((str(wid), float(cand_score), cand_extra))

    if not reranked:
        return MatchDecision(
            best_id=None,
            top1=0.0,
            top2=0.0,
            accepted=False,
            method="dino_index_masked_rerank",
            extra={"reason": "rerank_empty", "retrieval_top": top_retrieval},
        )

    reranked.sort(key=lambda x: x[1], reverse=True)
    retrieval_best_id = str(top_retrieval[0][0])
    reranked_by_id = {wid: (score, extra) for wid, score, extra in reranked}

    guard_applied = False

    best_id = reranked[0][0]
    best_score = reranked[0][1]
    best_extra = reranked[0][2]

    retrieval_best_sim = float(top_retrieval[0][1])

    if (
            retrieval_best_id in reranked_by_id
            and (
            retrieval_margin >= 0.03
            or retrieval_best_sim >= 0.70
    )
    ):
        r_score, r_extra = reranked_by_id[retrieval_best_id]

        if best_id != retrieval_best_id:
            appearance_gap = float(best_extra.get("appearance_score", 0.0)) - float(
                r_extra.get("appearance_score", 0.0)
            )
            align_gap = float(best_extra.get("align_score", 0.0)) - float(
                r_extra.get("align_score", 0.0)
            )
            score_gap = float(best_score - r_score)

            # retrieval почти всегда лучше по форме; rerank разрешаем переворот
            # только если он очень явно доминирует
            if not (
                    appearance_gap >= 0.12
                    and align_gap >= 0.015
                    and score_gap >= 0.05
            ):
                best_id = retrieval_best_id
                best_score = float(r_score)
                best_extra = r_extra
                guard_applied = True

    second_score = 0.0
    for wid, score, extra in reranked:
        if wid != best_id:
            second_score = float(score)
            break

    if guard_applied:
        margin = float(retrieval_margin)
    else:
        margin = float(best_score - second_score)

    retrieval_best_id = str(top_retrieval[0][0])
    retrieval_best_sim = float(top_retrieval[0][1])

    if guard_applied:
        accepted = (
                best_id is not None
                and rarity is not None
                and retrieval_best_sim >= 0.60
        )
    else:
        accepted = (
                best_id is not None
                and rarity is not None
                and (
                        margin >= 0.02
                        or (
                                best_extra.get("appearance_score", 0.0) >= 0.72
                                and best_extra.get("align_score", 0.0) >= 0.055
                        )
                        or (
                                best_id == retrieval_best_id
                                and retrieval_best_sim >= 0.54
                                and best_extra.get("align_score", 0.0) >= 0.04
                        )
                )
        )

    return MatchDecision(
        best_id=best_id,
        top1=float(best_score),
        top2=float(second_score),
        accepted=bool(accepted),
        method="dino_index_masked_rerank",
        extra={
            "margin": float(margin),
            "retrieval_margin": float(retrieval_margin),
            "rerank_top_n": int(top_n_rerank),
            "guard_applied": bool(guard_applied),
            "retrieval_top": [{"id": wid, "score": float(score)} for wid, score in top_retrieval],
            "reranked_top": [
                {
                    "id": wid,
                    "score": float(score),
                    "emb_sim": float(extra.get("emb_sim", 0.0)),
                    "align_score": float(extra.get("align_score", 0.0)),
                    "appearance_score": float(extra.get("appearance_score", 0.0)),
                }
                for wid, score, extra in reranked
            ],
            **best_extra,
        },
    )


def _save_accept_debug(
    crop_bgr: np.ndarray,
    weapon_crop_name: str,
    best_id: str,
    decision: MatchDecision,
    weapon_type: str,
    rarity: int | None,
    debug_dir: str,
    cache_weapons_dir: str,
):
    acc_dir = _p(debug_dir, "accepted")
    _ensure_dir(acc_dir)

    base = (
        f"{os.path.splitext(weapon_crop_name)[0]}"
        f"__id_{best_id}"
        f"__s1_{decision.top1:.4f}"
        f"__s2_{decision.top2:.4f}"
        f"__t_{weapon_type}"
        f"__r_{rarity}"
    )

    try:
        cv2.imwrite(os.path.join(acc_dir, base + ".png"), crop_bgr)
    except Exception:
        pass

    try:
        ref_path = _p(cache_weapons_dir, f"{best_id}.png")
        ref_img = _load_image(ref_path, cv2.IMREAD_UNCHANGED)
        if ref_img is not None:
            cv2.imwrite(os.path.join(acc_dir, base + "__ref.png"), ref_img)
    except Exception:
        pass

    try:
        with open(os.path.join(acc_dir, base + "__meta.json"), "w", encoding="utf-8") as f:
            json.dump({
                "best_id": best_id,
                "top1": decision.top1,
                "top2": decision.top2,
                "accepted": decision.accepted,
                "method": decision.method,
                "weapon_type": weapon_type,
                "rarity": rarity,
                "extra": decision.extra,
            }, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _save_reject_debug(
    crop_bgr: np.ndarray,
    weapon_crop_name: str,
    decision: MatchDecision,
    weapon_type: str,
    rarity: int | None,
    debug_dir: str,
):
    rej_dir = _p(debug_dir, "rejected")
    _ensure_dir(rej_dir)

    base = (
        f"{os.path.splitext(weapon_crop_name)[0]}"
        f"__best_{decision.best_id}"
        f"__s1_{decision.top1:.4f}"
        f"__s2_{decision.top2:.4f}"
        f"__t_{weapon_type}"
        f"__r_{rarity}"
    )

    try:
        cv2.imwrite(os.path.join(rej_dir, base + ".png"), crop_bgr)
    except Exception:
        pass

    try:
        with open(os.path.join(rej_dir, base + "__meta.json"), "w", encoding="utf-8") as f:
            json.dump({
                "best_id": decision.best_id,
                "top1": decision.top1,
                "top2": decision.top2,
                "accepted": decision.accepted,
                "method": decision.method,
                "weapon_type": weapon_type,
                "rarity": rarity,
                "extra": decision.extra,
            }, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def match_weapons(
    parsed: dict,
    char_map: dict[str, str],
    chars_db: dict,
    weaps_db: dict,
    crops_weap_dir: str = "assets/weapons",
    out_hd_weap_dir: str = "assets/hd/weapons",
    cache_weapons_dir: str = "cache/enka_ref_weapons",
    debug_dir: str = "debug/weapons",
) -> dict:
    _ensure_dir(_p(out_hd_weap_dir))
    _ensure_dir(_p(debug_dir, "accepted"))
    _ensure_dir(_p(debug_dir, "rejected"))

    index_info = build_weapon_embedding_index(
        weaps_db=weaps_db,
        cache_weapons_dir=cache_weapons_dir,
        out_index_path="cache/ref_index/weapons_dino_index.npz",
        size=224,
        force=False,
    )
    print("WEAPON EMB INDEX:", index_info)

    emb_index = load_weapon_embedding_index("cache/ref_index/weapons_dino_index.npz")
    if not emb_index:
        return {
            "accepted_new_hd": 0,
            "accepted_total": 0,
            "rejected": 0,
            "skipped_no_char": 0,
            "skipped_no_weapon_type": 0,
            "skipped_no_crop": 0,
            "skipped_no_candidates": 0,
            "skipped_low_rarity": 0,
            "pairs_total": 0,
            "debug_dir": debug_dir,
            "method": "dino_index_rerank",
            "note": "embedding_index_missing",
        }

    type_rarity_to_ids = build_weapon_candidate_index(weaps_db)

    accepted_new_hd = 0
    accepted_total = 0
    rejected = 0
    skipped_no_char = 0
    skipped_no_weapon_type = 0
    skipped_no_crop = 0
    skipped_no_candidates = 0
    skipped_low_rarity = 0

    pairs = parsed.get("pairs", [])
    if not pairs:
        return {
            "accepted_new_hd": 0,
            "accepted_total": 0,
            "rejected": 0,
            "skipped_no_char": 0,
            "skipped_no_weapon_type": 0,
            "skipped_no_crop": 0,
            "skipped_no_candidates": 0,
            "skipped_low_rarity": 0,
            "pairs_total": 0,
            "debug_dir": debug_dir,
            "method": "dino_index_rerank",
            "note": "pairs_empty",
        }

    for pair in pairs:
        ci = int(pair["char_index"])
        wi = int(pair["weapon_index"])

        char_crop_name = f"char_{ci:03d}.png"
        weapon_crop_name = f"weapon_{wi:03d}.png"

        char_id = char_map.get(char_crop_name)
        if not char_id:
            skipped_no_char += 1
            continue

        weapon_type = chars_db.get(str(char_id), {}).get("weapon_type")
        if not weapon_type:
            skipped_no_weapon_type += 1
            continue

        crop_path = _p(crops_weap_dir, weapon_crop_name)
        crop_bgr = _load_image(crop_path, cv2.IMREAD_COLOR)
        if crop_bgr is None:
            skipped_no_crop += 1
            continue

        rarity = detect_weapon_rarity_from_crop(crop_bgr)
        if rarity in (1, 2):
            skipped_low_rarity += 1
            continue

        candidate_ids = get_candidate_ids(
            weapon_type=str(weapon_type),
            rarity=rarity,
            type_rarity_to_ids=type_rarity_to_ids,
            weaps_db=weaps_db,
        )

        if not candidate_ids:
            skipped_no_candidates += 1
            _save_reject_debug(
                crop_bgr=crop_bgr,
                weapon_crop_name=weapon_crop_name,
                decision=MatchDecision(
                    best_id=None,
                    top1=0.0,
                    top2=0.0,
                    accepted=False,
                    method="dino_index_rerank",
                    extra={"reason": "no_candidates"},
                ),
                weapon_type=str(weapon_type),
                rarity=rarity,
                debug_dir=debug_dir,
            )
            continue

        if len(candidate_ids) < 2:
            rejected += 1
            _save_reject_debug(
                crop_bgr=crop_bgr,
                weapon_crop_name=weapon_crop_name,
                decision=MatchDecision(
                    best_id=None,
                    top1=0.0,
                    top2=0.0,
                    accepted=False,
                    method="dino_index_rerank",
                    extra={"reason": "singleton_bucket", "candidate_ids": candidate_ids},
                ),
                weapon_type=str(weapon_type),
                rarity=rarity,
                debug_dir=debug_dir,
            )
            continue

        decision = match_weapon_crop(
            crop_bgr=crop_bgr,
            candidate_ids=candidate_ids,
            weaps_db=weaps_db,
            emb_index=emb_index,
            cache_weapons_dir=cache_weapons_dir,
            rarity=rarity,
            size=224,
        )

        if not decision.accepted or decision.best_id is None:
            rejected += 1
            _save_reject_debug(
                crop_bgr=crop_bgr,
                weapon_crop_name=weapon_crop_name,
                decision=decision,
                weapon_type=str(weapon_type),
                rarity=rarity,
                debug_dir=debug_dir,
            )
            continue

        accepted_total += 1
        _save_accept_debug(
            crop_bgr=crop_bgr,
            weapon_crop_name=weapon_crop_name,
            best_id=decision.best_id,
            decision=decision,
            weapon_type=str(weapon_type),
            rarity=rarity,
            debug_dir=debug_dir,
            cache_weapons_dir=cache_weapons_dir,
        )

        dst = _p(out_hd_weap_dir, f"{decision.best_id}.png")
        existed_before = os.path.exists(dst)

        copied_ok = copy_weapon_hd_from_cache(
            best_id=decision.best_id,
            out_hd_weap_dir=out_hd_weap_dir,
            cache_dir=cache_weapons_dir,
        )

        exists_after = os.path.exists(dst)
        if copied_ok and (not existed_before) and exists_after:
            accepted_new_hd += 1

    return {
        "accepted_new_hd": accepted_new_hd,
        "accepted_total": accepted_total,
        "rejected": rejected,
        "skipped_no_char": skipped_no_char,
        "skipped_no_weapon_type": skipped_no_weapon_type,
        "skipped_no_crop": skipped_no_crop,
        "skipped_no_candidates": skipped_no_candidates,
        "skipped_low_rarity": skipped_low_rarity,
        "pairs_total": len(pairs),
        "debug_dir": debug_dir,
        "method": "dino_index_rerank",
    }