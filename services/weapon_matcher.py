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


def _load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


RARITY_BG = {
    1: (120, 120, 120),
    2: (120, 150, 110),
    3: (170, 130, 90),
    4: (160, 110, 160),
    5: (80, 140, 190),
}


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

def _make_ref_views(ref_png_path: str, rarity: int | None, size: int = 224) -> list[tuple[str, np.ndarray]]:
    views = []
    params = [
        ("base",   {"scale": 0.84, "dx": 0.00, "dy": 0.00}),
        ("zoom",   {"scale": 0.92, "dx": 0.00, "dy": 0.00}),
        ("small",  {"scale": 0.76, "dx": 0.00, "dy": 0.00}),
        ("left",   {"scale": 0.84, "dx": -0.03, "dy": 0.00}),
        ("right",  {"scale": 0.84, "dx": 0.03, "dy": 0.00}),
    ]

    for view_name, p in params:
        ref_view = _render_ref_view(
            ref_png_path=ref_png_path,
            rarity=rarity,
            size=size,
            scale=float(p["scale"]),
            dx=float(p["dx"]),
            dy=float(p["dy"]),
        )
        if ref_view is not None:
            views.append((view_name, ref_view))

    return views

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

        for j, view_bgr in enumerate(views):
            if j == 0:
                print(f"   ↳ weapon {wid} views: {len(views)}")

            try:
                emb = _encode_bgr_dino(view_bgr)
            except Exception:
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

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def _make_ref_views(ref_png_path: str, rarity: int | None, size: int = 224) -> list[np.ndarray]:
    views = []
    params = [
        {"scale": 0.76, "dx": 0.00, "dy": 0.00},
        {"scale": 0.84, "dx": 0.00, "dy": 0.00},
        {"scale": 0.92, "dx": 0.00, "dy": 0.00},
        {"scale": 0.84, "dx": -0.03, "dy": 0.00},
        {"scale": 0.84, "dx": 0.03, "dy": 0.00},
        {"scale": 0.84, "dx": 0.00, "dy": -0.03},
        {"scale": 0.84, "dx": 0.00, "dy": 0.03},
    ]

    for p in params:
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


def detect_weapon_rarity_from_crop(crop_bgr: np.ndarray) -> int | None:
    h, w = crop_bgr.shape[:2]
    if h < 20 or w < 20:
        return None

    img = crop_bgr.copy()

    # Мягко подрезаем рамку, но не слишком сильно
    p = int(min(h, w) * 0.08)
    if p > 0 and h - 2 * p >= 16 and w - 2 * p >= 16:
        img = img[p:h - p, p:w - p]

    hh, ww = img.shape[:2]
    if hh < 16 or ww < 16:
        return None

    # Берём несколько угловых зон, а не центр
    patch_size = int(min(hh, ww) * 0.22)
    patch_size = max(8, patch_size)

    patches = [
        img[0:patch_size, 0:patch_size],                         # top-left
        img[0:patch_size, ww - patch_size:ww],                  # top-right
        img[hh - patch_size:hh, 0:patch_size],                  # bottom-left
        img[hh - patch_size:hh, ww - patch_size:ww],            # bottom-right
    ]

    hsv_pixels = []

    for patch in patches:
        if patch.size == 0:
            continue

        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        H = hsv[:, :, 0]
        S = hsv[:, :, 1]
        V = hsv[:, :, 2]

        # Отбрасываем слишком тёмное и почти серое
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

    # Очень низкая насыщенность — обычно 1*
    if Sm < 35 and Vm > 60:
        return 1

    # 5* — золотисто-оранжевый
    if 8 <= Hm <= 28 and Sm >= 70:
        return 5

    # 4* — фиолетовый / сиреневый
    if 125 <= Hm <= 170 and Sm >= 45:
        return 4

    # 3* — синий / голубой
    if 90 <= Hm < 125 and Sm >= 45:
        return 3

    # 2* — зелёный
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


def _crop_inner(img_bgr: np.ndarray, pad_ratio: float = 0.18) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    p = int(min(h, w) * pad_ratio)
    if p > 0 and h - 2 * p >= 10 and w - 2 * p >= 10:
        return img_bgr[p:h - p, p:w - p].copy()
    return img_bgr.copy()


def _prepare_crop_for_match(crop_bgr: np.ndarray, size: int = 160) -> np.ndarray:
    crop = _crop_inner(crop_bgr, pad_ratio=0.18)
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return crop


def _render_ref_view(
    ref_png_path: str,
    rarity: int | None,
    size: int = 160,
    scale: float = 0.84,
    dx: float = 0.0,
    dy: float = 0.0,
) -> np.ndarray | None:
    img = _load_image(ref_png_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    bg = RARITY_BG.get(int(rarity or 1), (128, 128, 128))
    canvas = np.full((size, size, 3), bg, dtype=np.uint8)

    img = _alpha_composite_to_bg(img, bg)
    if img is None:
        return None

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    h, w = img.shape[:2]
    if h < 2 or w < 2:
        return None

    target = int(size * scale)
    scale_k = min(target / float(w), target / float(h))
    nw, nh = max(1, int(round(w * scale_k))), max(1, int(round(h * scale_k)))
    img2 = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    x = int(round((size - nw) / 2.0 + dx * size))
    y = int(round((size - nh) / 2.0 + dy * size))

    x = max(0, min(size - nw, x))
    y = max(0, min(size - nh, y))

    canvas[y:y + nh, x:x + nw] = img2
    return canvas


def _gray(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def _edge_map(img_bgr: np.ndarray) -> np.ndarray:
    g = _gray(img_bgr)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    e = cv2.Canny(g, 60, 140)
    return e


def _fg_mask(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    mask = ((s > 35) | (v > 70)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    return mask


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


def _color_score(crop_bgr: np.ndarray, ref_bgr: np.ndarray) -> float:
    crop_hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    ref_hsv = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2HSV)

    crop_hist = cv2.calcHist([crop_hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
    ref_hist = cv2.calcHist([ref_hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])

    crop_hist = cv2.normalize(crop_hist, None).flatten()
    ref_hist = cv2.normalize(ref_hist, None).flatten()

    score = cv2.compareHist(crop_hist, ref_hist, cv2.HISTCMP_CORREL)
    return float((score + 1.0) / 2.0)


def _score_crop_vs_ref(crop_bgr: np.ndarray, ref_bgr: np.ndarray) -> tuple[float, dict]:
    crop_edges = _edge_map(crop_bgr)
    ref_edges = _edge_map(ref_bgr)

    crop_mask = _fg_mask(crop_bgr)
    ref_mask = _fg_mask(ref_bgr)

    edge = _iou(crop_edges, ref_edges)
    mask = _iou(crop_mask, ref_mask)
    patch = _patch_score(crop_bgr, ref_bgr)
    color = _color_score(crop_bgr, ref_bgr)

    score = (
        0.40 * edge +
        0.30 * mask +
        0.20 * patch +
        0.10 * color
    )

    return float(score), {
        "edge": float(edge),
        "mask": float(mask),
        "patch": float(patch),
        "color": float(color),
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
            method="dino_index_rerank",
            extra={"reason": "empty_input"},
        )

    top_retrieval = retrieve_top_candidates(
        crop_bgr=crop_bgr,
        candidate_ids=candidate_ids,
        emb_index=emb_index,
        size=size,
        top_n=5,
    )

    if not top_retrieval:
        return MatchDecision(
            best_id=None,
            top1=0.0,
            top2=0.0,
            accepted=False,
            method="dino_index_rerank",
            extra={"reason": "no_retrieval_candidates"},
        )

    crop_prep = _prepare_crop_for_match(crop_bgr, size=size)

    reranked = []

    for wid, emb_sim in top_retrieval:
        ref_path = _p(cache_weapons_dir, f"{wid}.png")
        wrarity = weaps_db.get(str(wid), {}).get("rarity", rarity)

        local_best = -1.0
        local_extra = {}

        for _, ref_view in _make_ref_views(ref_path, rarity=wrarity, size=size):
            local_score, score_extra = _score_crop_vs_ref(crop_prep, ref_view)
            final_score = 0.70 * float(emb_sim) + 0.30 * float(local_score)

            if final_score > local_best:
                local_best = final_score
                local_extra = {
                    "emb_sim": float(emb_sim),
                    "local_score": float(local_score),
                    **score_extra,
                }

        reranked.append((str(wid), float(local_best), local_extra))

    reranked.sort(key=lambda x: x[1], reverse=True)

    best_id = reranked[0][0]
    best_score = reranked[0][1]
    best_extra = reranked[0][2]

    second_score = reranked[1][1] if len(reranked) > 1 else 0.0
    margin = best_score - second_score

    accepted = (
        best_id is not None
        and best_score >= 0.58
        and margin >= 0.03
    )

    return MatchDecision(
        best_id=best_id,
        top1=float(best_score),
        top2=float(second_score),
        accepted=bool(accepted),
        method="dino_index_rerank",
        extra={
            "margin": float(margin),
            "retrieval_top": [{"id": wid, "score": score} for wid, score in top_retrieval],
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
            "pairs_total": 0,
            "debug_dir": debug_dir,
            "method": "edge_mask_mv",
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
                    method="edge_mask_mv",
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
                    method="edge_mask_mv",
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

        if copy_weapon_hd_from_cache(
            best_id=decision.best_id,
            out_hd_weap_dir=out_hd_weap_dir,
            cache_dir=cache_weapons_dir,
        ):
            dst = _p(out_hd_weap_dir, f"{decision.best_id}.png")
            if os.path.exists(dst):
                accepted_new_hd += 1

    return {
        "accepted_new_hd": accepted_new_hd,
        "accepted_total": accepted_total,
        "rejected": rejected,
        "skipped_no_char": skipped_no_char,
        "skipped_no_weapon_type": skipped_no_weapon_type,
        "skipped_no_crop": skipped_no_crop,
        "skipped_no_candidates": skipped_no_candidates,
        "pairs_total": len(pairs),
        "debug_dir": debug_dir,
        "method": "edge_mask_mv",
    }