import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services import weapon_matcher as wm


TEST_DIR = ROOT / "test"
CROP_DIR = TEST_DIR / "crop"
REF_DIR = TEST_DIR / "ref"
OUT_DIR = TEST_DIR / "debug_weapon_match"

SIZE = 224


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_img(path: Path, img: np.ndarray):
    ensure_dir(path.parent)
    cv2.imwrite(str(path), img)


def load_crop() -> tuple[Path, np.ndarray]:
    files = sorted(CROP_DIR.glob("*.png"))
    if not files:
        raise FileNotFoundError(f"В {CROP_DIR} нет crop png")
    if len(files) > 1:
        print(f"[WARN] В {CROP_DIR} несколько файлов, беру первый: {files[0].name}")
    path = files[0]
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Не удалось прочитать crop: {path}")
    return path, img


def load_refs() -> list[tuple[str, Path]]:
    files = sorted(REF_DIR.glob("*.png"))
    if not files:
        raise FileNotFoundError(f"В {REF_DIR} нет ref png")
    out = []
    for p in files:
        out.append((p.stem, p))
    return out


def mask_to_bgr(mask: np.ndarray) -> np.ndarray | None:
    if mask is None:
        return None
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def overlay_mask(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    if img_bgr is None or mask is None:
        return None
    out = img_bgr.copy()
    green = np.zeros_like(out)
    green[:, :, 1] = 255
    m = (mask > 0).astype(np.uint8)
    m3 = np.repeat(m[:, :, None], 3, axis=2)
    out = np.where(m3 > 0, cv2.addWeighted(out, 0.65, green, 0.35, 0), out)
    return out.astype(np.uint8)


def edge_vis(img_bgr: np.ndarray) -> np.ndarray:
    e = wm._edge_map(img_bgr)
    return cv2.cvtColor(e, cv2.COLOR_GRAY2BGR)


def side_by_side(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape[:2] != b.shape[:2]:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.hstack([a, b])


def safe_json(obj):
    if isinstance(obj, dict):
        return {str(k): safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [safe_json(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def normalize_mask(mask: np.ndarray) -> np.ndarray:
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def keep_largest_component(mask: np.ndarray, min_area: int = 24) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8),
        connectivity=8,
    )
    if n <= 1:
        return normalize_mask(mask)

    best_label = None
    best_area = 0
    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area > best_area:
            best_area = area
            best_label = label

    out = np.zeros_like(mask, dtype=np.uint8)
    if best_label is not None and best_area >= min_area:
        out[labels == best_label] = 255
    return normalize_mask(out)


def edge_mask_base(prep_bgr: np.ndarray) -> np.ndarray:
    edges = wm._edge_map(prep_bgr)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(edges, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = keep_largest_component(mask, min_area=24)
    return normalize_mask(mask)


def _sample_positions(length: int) -> list[int]:
    raw = [0.15, 0.30, 0.50, 0.70, 0.85]
    out = []
    for r in raw:
        p = int(round((length - 1) * r))
        p = max(0, min(length - 1, p))
        out.append(p)
    return sorted(set(out))


def _run_white_from_left(mask: np.ndarray, y: int) -> int:
    h, w = mask.shape
    cnt = 0
    for x in range(w):
        if mask[y, x] > 0:
            cnt += 1
        else:
            break
    return cnt


def _run_white_from_right(mask: np.ndarray, y: int) -> int:
    h, w = mask.shape
    cnt = 0
    for x in range(w - 1, -1, -1):
        if mask[y, x] > 0:
            cnt += 1
        else:
            break
    return cnt


def _run_white_from_top(mask: np.ndarray, x: int) -> int:
    h, w = mask.shape
    cnt = 0
    for y in range(h):
        if mask[y, x] > 0:
            cnt += 1
        else:
            break
    return cnt


def _run_white_from_bottom(mask: np.ndarray, x: int) -> int:
    h, w = mask.shape
    cnt = 0
    for y in range(h - 1, -1, -1):
        if mask[y, x] > 0:
            cnt += 1
        else:
            break
    return cnt


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


def detect_border_stripes(mask: np.ndarray) -> dict:
    h, w = mask.shape

    y_samples = _sample_positions(h)
    x_samples = _sample_positions(w)

    left_runs = [_run_white_from_left(mask, y) for y in y_samples]
    right_runs = [_run_white_from_right(mask, y) for y in y_samples]
    top_runs = [_run_white_from_top(mask, x) for x in x_samples]
    bottom_runs = [_run_white_from_bottom(mask, x) for x in x_samples]

    left = _robust_border_width(left_runs)
    right = _robust_border_width(right_runs)
    top = _robust_border_width(top_runs)
    bottom = _robust_border_width(bottom_runs)

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


def cleanup_mask_by_detected_border_hard(mask: np.ndarray) -> tuple[np.ndarray, dict]:
    mask = normalize_mask(mask)
    info = detect_border_stripes(mask)

    out = mask.copy()
    h, w = out.shape

    l = int(info["left"]) + 1 if int(info["left"]) > 0 else 0
    r = int(info["right"]) + 1 if int(info["right"]) > 0 else 0
    t = int(info["top"]) + 1 if int(info["top"]) > 0 else 0
    b = int(info["bottom"]) + 1 if int(info["bottom"]) > 0 else 0

    if l > 0:
        out[:, :min(l, w)] = 0
    if r > 0:
        out[:, max(0, w - r):w] = 0
    if t > 0:
        out[:min(t, h), :] = 0
    if b > 0:
        out[max(0, h - b):h, :] = 0

    out = normalize_mask(out)
    out = keep_largest_component(out, min_area=24)
    out = normalize_mask(out)

    info["applied"] = {
        "left": l,
        "right": r,
        "top": t,
        "bottom": b,
    }
    return out, info


def _appearance_match_score_with_fixed_crop_mask(
    crop_bgr: np.ndarray,
    crop_mask: np.ndarray,
    ref_bgr: np.ndarray,
    ref_mask: np.ndarray,
) -> tuple[float, dict]:
    crop_feat = wm._masked_appearance_features(crop_bgr, crop_mask)
    ref_feat = wm._masked_appearance_features(ref_bgr, ref_mask)

    if not crop_feat.get("valid") or not ref_feat.get("valid"):
        return 0.0, {
            "appearance_score": 0.0,
            "crop_feat": crop_feat,
            "ref_feat": ref_feat,
        }

    hue_score = wm._hist_intersection(crop_feat["hue_hist"], ref_feat["hue_hist"])
    sat_score = 0.5 * wm._score_closeness(crop_feat["sat_p50"], ref_feat["sat_p50"], 80.0) + \
                0.5 * wm._score_closeness(crop_feat["sat_p75"], ref_feat["sat_p75"], 80.0)
    val_score = 0.5 * wm._score_closeness(crop_feat["val_p50"], ref_feat["val_p50"], 80.0) + \
                0.5 * wm._score_closeness(crop_feat["val_p75"], ref_feat["val_p75"], 80.0)
    bright_score = wm._score_closeness(crop_feat["bright_ratio"], ref_feat["bright_ratio"], 0.35)
    dark_score = wm._score_closeness(crop_feat["dark_ratio"], ref_feat["dark_ratio"], 0.35)
    vivid_score = wm._score_closeness(crop_feat["vivid_ratio"], ref_feat["vivid_ratio"], 0.35)
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
        "crop_feat": safe_json(crop_feat),
        "ref_feat": safe_json(ref_feat),
    }


def _score_candidate_masked_with_fixed_crop_mask(
    crop_bgr: np.ndarray,
    crop_mask: np.ndarray,
    ref_png_path: str,
    size: int,
    emb_sim: float,
) -> tuple[float, dict, np.ndarray | None, np.ndarray | None]:
    best_align_score = -1.0
    best_ref_view = None
    best_ref_mask = None
    best_align_extra = {}
    best_view_params = None

    for p in wm.REF_VIEW_PARAMS:
        ref_view, ref_mask = wm._render_ref_view_and_mask(
            ref_png_path=ref_png_path,
            size=size,
            scale=float(p["scale"]),
            dx=float(p["dx"]),
            dy=float(p["dy"]),
        )

        if ref_view is None or ref_mask is None:
            continue

        align_score, align_extra = wm._score_crop_vs_ref(crop_bgr, ref_view)

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
        }, None, None

    appearance_score, appearance_extra = _appearance_match_score_with_fixed_crop_mask(
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
    }, best_ref_view, best_ref_mask


def run():
    ensure_dir(OUT_DIR)

    crop_path, crop_src = load_crop()
    refs = load_refs()

    print(f"[CROP] {crop_path.name}")
    print(f"[REFS] {len(refs)} files")

    crop_prep = wm._prepare_crop_for_match(crop_src, size=SIZE)
    crop_mask_base_edges = edge_mask_base(crop_prep)
    crop_mask, border_info = cleanup_mask_by_detected_border_hard(crop_mask_base_edges)
    crop_emb = wm._encode_bgr_dino(crop_prep)

    save_img(OUT_DIR / "crop__original.png", crop_src)
    save_img(OUT_DIR / "crop__prepared.png", crop_prep)
    save_img(OUT_DIR / "crop__mask_base_edges.png", crop_mask_base_edges)
    save_img(OUT_DIR / "crop__mask_base_edges_overlay.png", overlay_mask(crop_prep, crop_mask_base_edges))
    save_img(OUT_DIR / "crop__mask_cleaned.png", crop_mask)
    save_img(OUT_DIR / "crop__mask_cleaned_overlay.png", overlay_mask(crop_prep, crop_mask))
    save_img(OUT_DIR / "crop__edges.png", edge_vis(crop_prep))

    report = {
        "crop_file": crop_path.name,
        "size": SIZE,
        "crop_shape_original": list(crop_src.shape),
        "crop_shape_prepared": list(crop_prep.shape),
        "crop_mask_area": int(np.count_nonzero(crop_mask > 0)),
        "crop_mask_area_ratio": float(np.count_nonzero(crop_mask > 0) / float(crop_mask.shape[0] * crop_mask.shape[1])),
        "crop_mask_base_edges_area": int(np.count_nonzero(crop_mask_base_edges > 0)),
        "crop_mask_base_edges_area_ratio": float(
            np.count_nonzero(crop_mask_base_edges > 0) / float(crop_mask_base_edges.shape[0] * crop_mask_base_edges.shape[1])
        ),
        "border_cleanup": safe_json(border_info),
        "refs_total": len(refs),
        "retrieval_ranking": [],
        "rerank_ranking": [],
        "refs": [],
    }

    rerank_rows = []

    for ref_id, ref_path in refs:
        print(f"[REF] {ref_id}")

        ref_best_view_name = None
        ref_best_view_params = None
        ref_best_emb_sim = None
        ref_best_align_score = -1.0
        ref_best_align_extra = None
        ref_best_ref_view = None
        ref_best_ref_mask = None

        ref_view_rows = []

        for i, p in enumerate(wm.REF_VIEW_PARAMS):
            ref_view, ref_mask = wm._render_ref_view_and_mask(
                ref_png_path=str(ref_path),
                size=SIZE,
                scale=float(p["scale"]),
                dx=float(p["dx"]),
                dy=float(p["dy"]),
            )

            if ref_view is None or ref_mask is None:
                continue

            view_tag = f"view_{i}__s{p['scale']:.2f}__dx{p['dx']:.2f}__dy{p['dy']:.2f}"

            ref_emb = wm._encode_bgr_dino(ref_view)
            emb_sim = float(ref_emb @ crop_emb)

            align_score, align_extra = wm._score_crop_vs_ref(crop_prep, ref_view)

            save_img(OUT_DIR / "refs" / ref_id / f"{view_tag}__ref_view.png", ref_view)
            save_img(OUT_DIR / "refs" / ref_id / f"{view_tag}__ref_mask.png", ref_mask)
            save_img(OUT_DIR / "refs" / ref_id / f"{view_tag}__ref_mask_overlay.png", overlay_mask(ref_view, ref_mask))
            save_img(OUT_DIR / "refs" / ref_id / f"{view_tag}__ref_edges.png", edge_vis(ref_view))
            save_img(
                OUT_DIR / "refs" / ref_id / f"{view_tag}__compare_crop_vs_ref.png",
                side_by_side(crop_prep, ref_view),
            )
            save_img(
                OUT_DIR / "refs" / ref_id / f"{view_tag}__compare_cropmask_vs_refmask.png",
                side_by_side(mask_to_bgr(crop_mask), mask_to_bgr(ref_mask)),
            )

            row = {
                "view_index": i,
                "view_name": view_tag,
                "view_params": {
                    "scale": float(p["scale"]),
                    "dx": float(p["dx"]),
                    "dy": float(p["dy"]),
                },
                "emb_sim": emb_sim,
                "align_score": float(align_score),
                "edge": float(align_extra.get("edge", 0.0)),
                "patch": float(align_extra.get("patch", 0.0)),
                "ref_mask_area": int(np.count_nonzero(ref_mask > 0)),
                "ref_mask_area_ratio": float(np.count_nonzero(ref_mask > 0) / float(ref_mask.shape[0] * ref_mask.shape[1])),
                "files": {
                    "ref_view": str((Path("refs") / ref_id / f"{view_tag}__ref_view.png").as_posix()),
                    "ref_mask": str((Path("refs") / ref_id / f"{view_tag}__ref_mask.png").as_posix()),
                    "ref_mask_overlay": str((Path("refs") / ref_id / f"{view_tag}__ref_mask_overlay.png").as_posix()),
                    "ref_edges": str((Path("refs") / ref_id / f"{view_tag}__ref_edges.png").as_posix()),
                    "compare_crop_vs_ref": str((Path("refs") / ref_id / f"{view_tag}__compare_crop_vs_ref.png").as_posix()),
                    "compare_cropmask_vs_refmask": str((Path("refs") / ref_id / f"{view_tag}__compare_cropmask_vs_refmask.png").as_posix()),
                },
            }
            ref_view_rows.append(row)

            if align_score > ref_best_align_score:
                ref_best_align_score = float(align_score)
                ref_best_align_extra = dict(align_extra)
                ref_best_ref_view = ref_view
                ref_best_ref_mask = ref_mask
                ref_best_view_name = view_tag
                ref_best_view_params = {
                    "scale": float(p["scale"]),
                    "dx": float(p["dx"]),
                    "dy": float(p["dy"]),
                }
                ref_best_emb_sim = emb_sim

        if not ref_view_rows or ref_best_ref_view is None or ref_best_ref_mask is None:
            report["refs"].append({
                "ref_id": ref_id,
                "error": "no_valid_ref_views",
            })
            continue

        appearance_score, appearance_extra = _appearance_match_score_with_fixed_crop_mask(
            crop_bgr=crop_prep,
            crop_mask=crop_mask,
            ref_bgr=ref_best_ref_view,
            ref_mask=ref_best_ref_mask,
        )

        final_score, final_extra, final_best_ref_view, final_best_ref_mask = _score_candidate_masked_with_fixed_crop_mask(
            crop_bgr=crop_prep,
            crop_mask=crop_mask,
            ref_png_path=str(ref_path),
            size=SIZE,
            emb_sim=float(ref_best_emb_sim),
        )

        best_ref_view_for_save = final_best_ref_view if final_best_ref_view is not None else ref_best_ref_view
        best_ref_mask_for_save = final_best_ref_mask if final_best_ref_mask is not None else ref_best_ref_mask

        save_img(OUT_DIR / "refs" / ref_id / "best__ref_view.png", best_ref_view_for_save)
        save_img(OUT_DIR / "refs" / ref_id / "best__ref_mask.png", best_ref_mask_for_save)
        save_img(OUT_DIR / "refs" / ref_id / "best__ref_mask_overlay.png", overlay_mask(best_ref_view_for_save, best_ref_mask_for_save))
        save_img(
            OUT_DIR / "refs" / ref_id / "best__compare_crop_vs_ref.png",
            side_by_side(crop_prep, best_ref_view_for_save),
        )
        save_img(
            OUT_DIR / "refs" / ref_id / "best__compare_cropmask_vs_refmask.png",
            side_by_side(mask_to_bgr(crop_mask), mask_to_bgr(best_ref_mask_for_save)),
        )

        ref_entry = {
            "ref_id": ref_id,
            "retrieval_emb_sim_best_view": float(ref_best_emb_sim),
            "best_align_view_name": ref_best_view_name,
            "best_align_view_params": ref_best_view_params,
            "best_align_score": float(ref_best_align_score),
            "appearance_score_on_best_align_view": float(appearance_score),
            "final_score": float(final_score),
            "crop_mask_area": int(np.count_nonzero(crop_mask > 0)),
            "ref_mask_area_best_view": int(np.count_nonzero(best_ref_mask_for_save > 0)),
            "edge": float(ref_best_align_extra.get("edge", 0.0)),
            "patch": float(ref_best_align_extra.get("patch", 0.0)),
            "appearance": {
                "appearance_score": float(appearance_extra.get("appearance_score", 0.0)),
                "hue_score": float(appearance_extra.get("hue_score", 0.0)),
                "sat_score": float(appearance_extra.get("sat_score", 0.0)),
                "val_score": float(appearance_extra.get("val_score", 0.0)),
                "bright_score": float(appearance_extra.get("bright_score", 0.0)),
                "dark_score": float(appearance_extra.get("dark_score", 0.0)),
                "vivid_score": float(appearance_extra.get("vivid_score", 0.0)),
                "crop_feat": safe_json(appearance_extra.get("crop_feat", {})),
                "ref_feat": safe_json(appearance_extra.get("ref_feat", {})),
            },
            "final_extra": safe_json(final_extra),
            "files": {
                "best_ref_view": str((Path("refs") / ref_id / "best__ref_view.png").as_posix()),
                "best_ref_mask": str((Path("refs") / ref_id / "best__ref_mask.png").as_posix()),
                "best_ref_mask_overlay": str((Path("refs") / ref_id / "best__ref_mask_overlay.png").as_posix()),
                "best_compare_crop_vs_ref": str((Path("refs") / ref_id / "best__compare_crop_vs_ref.png").as_posix()),
                "best_compare_cropmask_vs_refmask": str((Path("refs") / ref_id / "best__compare_cropmask_vs_refmask.png").as_posix()),
            },
            "views": ref_view_rows,
        }

        report["refs"].append(ref_entry)

        report["retrieval_ranking"].append({
            "ref_id": ref_id,
            "emb_sim": float(ref_best_emb_sim),
            "best_view_name": ref_best_view_name,
            "best_view_params": ref_best_view_params,
        })

        rerank_rows.append({
            "ref_id": ref_id,
            "final_score": float(final_score),
            "emb_sim": float(final_extra.get("emb_sim", 0.0)),
            "align_score": float(final_extra.get("align_score", 0.0)),
            "appearance_score": float(final_extra.get("appearance_score", 0.0)),
            "crop_mask_area": int(final_extra.get("crop_mask_area", 0)),
            "ref_mask_area": int(final_extra.get("ref_mask_area", 0)),
            "view_params": safe_json(final_extra.get("view_params")),
        })

    report["retrieval_ranking"].sort(key=lambda x: x["emb_sim"], reverse=True)
    rerank_rows.sort(key=lambda x: x["final_score"], reverse=True)
    report["rerank_ranking"] = rerank_rows

    report["summary"] = {
        "retrieval_best_id": report["retrieval_ranking"][0]["ref_id"] if report["retrieval_ranking"] else None,
        "rerank_best_id": report["rerank_ranking"][0]["ref_id"] if report["rerank_ranking"] else None,
        "retrieval_vs_rerank_same": (
            report["retrieval_ranking"][0]["ref_id"] == report["rerank_ranking"][0]["ref_id"]
            if report["retrieval_ranking"] and report["rerank_ranking"]
            else None
        ),
        "crop_mask_area_ratio": report["crop_mask_area_ratio"],
        "crop_mask_base_edges_area_ratio": report["crop_mask_base_edges_area_ratio"],
    }

    with open(OUT_DIR / "report.json", "w", encoding="utf-8") as f:
        json.dump(safe_json(report), f, ensure_ascii=False, indent=2)

    print(f"[DONE] {OUT_DIR / 'report.json'}")
    print(f"[DONE] images -> {OUT_DIR}")


if __name__ == "__main__":
    run()