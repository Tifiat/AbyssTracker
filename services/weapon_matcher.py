from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass

import cv2
import numpy as np

from services.weapon_dino import (
    DinoTopKItem,
    composite_bgra_on_background,
    encode_weapon_bgra_dino,
    cosine_topk,
)
from services.weapon_ref_preparer import (
    prepare_compensated_crop,
    prepare_weapon_reference,
)


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _p(*parts: str) -> str:
    path = os.path.join(*parts)
    if os.path.isabs(path):
        return path
    return os.path.join(ROOT, path)


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _load_image(path: str, flags=cv2.IMREAD_COLOR):
    return cv2.imread(path, flags)


@dataclass
class DinoMatchDecision:
    best_id: str | None
    best_score: float
    accepted: bool
    top: list[DinoTopKItem]
    extra: dict


@dataclass
class RefEmbeddingRecord:
    weapon_id: str
    weapon_type: str
    embedding: np.ndarray
    meta: dict


_REF_EMBED_CACHE: dict[tuple[str, str, float, int], RefEmbeddingRecord] = {}


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


def _ref_cache_key(ref_path: str, weapon_id: str, weapon_type: str) -> tuple[str, str, float, int]:
    stat = os.stat(ref_path)
    return (
        str(weapon_id),
        str(weapon_type),
        float(stat.st_mtime),
        int(stat.st_size),
    )


def _encode_reference_weapon(
    weapon_id: str,
    weapon_type: str,
    cache_weapons_dir: str,
) -> RefEmbeddingRecord | None:
    ref_path = _p(cache_weapons_dir, f"{weapon_id}.png")
    if not os.path.exists(ref_path):
        return None

    key = _ref_cache_key(ref_path, weapon_id, weapon_type)
    cached = _REF_EMBED_CACHE.get(key)
    if cached is not None:
        return cached

    ref_img = _load_image(ref_path, cv2.IMREAD_UNCHANGED)
    if ref_img is None:
        return None

    prepared_ref, meta = prepare_weapon_reference(ref_img, weapon_type=weapon_type)
    embedding = encode_weapon_bgra_dino(prepared_ref)
    record = RefEmbeddingRecord(
        weapon_id=str(weapon_id),
        weapon_type=str(weapon_type),
        embedding=embedding,
        meta=meta,
    )
    _REF_EMBED_CACHE[key] = record
    return record


def _encode_reference_candidates(
    candidate_ids: list[str],
    weapon_type: str,
    cache_weapons_dir: str,
) -> tuple[np.ndarray, list[str], dict[str, dict], int]:
    embeddings = []
    ref_ids = []
    ref_meta = {}
    skipped_refs = 0

    for weapon_id in candidate_ids:
        try:
            record = _encode_reference_weapon(
                weapon_id=str(weapon_id),
                weapon_type=str(weapon_type),
                cache_weapons_dir=cache_weapons_dir,
            )
        except Exception as exc:
            ref_meta[str(weapon_id)] = {"error": str(exc)}
            skipped_refs += 1
            continue

        if record is None:
            skipped_refs += 1
            continue

        embeddings.append(record.embedding.astype(np.float32))
        ref_ids.append(record.weapon_id)
        ref_meta[record.weapon_id] = record.meta

    if not embeddings:
        return np.empty((0, 0), dtype=np.float32), [], ref_meta, skipped_refs

    return np.stack(embeddings, axis=0).astype(np.float32), ref_ids, ref_meta, skipped_refs


def match_prepared_weapon_crop(
    prepared_crop_bgra: np.ndarray,
    crop_meta: dict,
    candidate_ids: list[str],
    weapon_type: str,
    cache_weapons_dir: str = "cache/enka_ref_weapons",
    top_k: int = 10,
    min_score: float = 0.0,
) -> DinoMatchDecision:
    if prepared_crop_bgra is None:
        return DinoMatchDecision(
            best_id=None,
            best_score=0.0,
            accepted=False,
            top=[],
            extra={"reason": "empty_crop"},
        )

    if not candidate_ids:
        return DinoMatchDecision(
            best_id=None,
            best_score=0.0,
            accepted=False,
            top=[],
            extra={"reason": "no_candidates"},
        )

    crop_embedding = encode_weapon_bgra_dino(prepared_crop_bgra)

    ref_embeddings, ref_ids, ref_meta, skipped_refs = _encode_reference_candidates(
        candidate_ids=candidate_ids,
        weapon_type=weapon_type,
        cache_weapons_dir=cache_weapons_dir,
    )
    if len(ref_ids) == 0:
        return DinoMatchDecision(
            best_id=None,
            best_score=0.0,
            accepted=False,
            top=[],
            extra={
                "reason": "no_encoded_refs",
                "candidate_count": len(candidate_ids),
                "skipped_refs": skipped_refs,
                "crop_meta": crop_meta,
            },
        )

    top = cosine_topk(
        query_embedding=crop_embedding,
        ref_embeddings=ref_embeddings,
        ref_ids=ref_ids,
        top_k=top_k,
    )
    if not top:
        return DinoMatchDecision(
            best_id=None,
            best_score=0.0,
            accepted=False,
            top=[],
            extra={
                "reason": "empty_topk",
                "candidate_count": len(candidate_ids),
                "encoded_ref_count": len(ref_ids),
                "skipped_refs": skipped_refs,
                "crop_meta": crop_meta,
            },
        )

    best = top[0]
    return DinoMatchDecision(
        best_id=best.ref_id,
        best_score=float(best.score),
        accepted=bool(best.score >= float(min_score)),
        top=top,
        extra={
            "candidate_count": len(candidate_ids),
            "encoded_ref_count": len(ref_ids),
            "skipped_refs": skipped_refs,
            "crop_meta": crop_meta,
            "ref_meta": {item.ref_id: ref_meta.get(item.ref_id, {}) for item in top},
        },
    )


def match_weapon_crop(
    crop_bgr: np.ndarray,
    candidate_ids: list[str],
    weapon_type: str,
    cache_weapons_dir: str = "cache/enka_ref_weapons",
    top_k: int = 10,
    min_score: float = 0.0,
) -> DinoMatchDecision:
    if crop_bgr is None:
        return DinoMatchDecision(
            best_id=None,
            best_score=0.0,
            accepted=False,
            top=[],
            extra={"reason": "empty_crop"},
        )

    _, prepared_crop, crop_meta = prepare_compensated_crop(crop_bgr)
    return match_prepared_weapon_crop(
        prepared_crop_bgra=prepared_crop,
        crop_meta=crop_meta,
        candidate_ids=candidate_ids,
        weapon_type=weapon_type,
        cache_weapons_dir=cache_weapons_dir,
        top_k=top_k,
        min_score=min_score,
    )


def copy_weapon_hd_from_cache(
    best_id: str,
    out_hd_weap_dir: str = "assets/hd/weapons",
    cache_dir: str = "cache/enka_ref_weapons",
) -> bool:
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


def _decision_to_report_item(
    weapon_crop_name: str,
    char_crop_name: str,
    weapon_type: str,
    rarity: int | None,
    decision: DinoMatchDecision,
) -> dict:
    return {
        "crop": weapon_crop_name,
        "char_crop": char_crop_name,
        "weapon_type": weapon_type,
        "rarity": rarity,
        "best_id": decision.best_id,
        "best_score": float(decision.best_score),
        "accepted": bool(decision.accepted),
        "top": [
            {
                "id": item.ref_id,
                "score": float(item.score),
                "row_index": int(item.row_index),
            }
            for item in decision.top
        ],
        "extra": decision.extra,
    }


def _write_report(debug_dir: str, report: dict):
    _ensure_dir(_p(debug_dir))
    path = _p(debug_dir, "report_v2_dino.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _debug_base_name(
    weapon_crop_name: str,
    weapon_type: str,
    rarity: int | None,
    best_id: str | None = None,
    score: float | None = None,
) -> str:
    base = os.path.splitext(weapon_crop_name)[0]
    parts = [base, f"t_{weapon_type}", f"r_{rarity}"]
    if best_id is not None:
        parts.append(f"id_{best_id}")
    if score is not None:
        parts.append(f"s_{score:.4f}")
    return "__".join(parts)


def _put_label(img_bgr: np.ndarray, text: str) -> np.ndarray:
    out = img_bgr.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (0, 0, 0), thickness=-1)
    cv2.putText(
        out,
        text[:42],
        (5, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def _make_side_by_side(
    crop_bgr: np.ndarray,
    ref_bgr: np.ndarray | None,
    crop_label: str,
    ref_label: str,
) -> np.ndarray:
    crop_vis = _put_label(crop_bgr, crop_label)
    if ref_bgr is None:
        ref_vis = np.full_like(crop_vis, 127, dtype=np.uint8)
        ref_vis = _put_label(ref_vis, ref_label)
    else:
        ref_vis = _put_label(ref_bgr, ref_label)

    sep = np.full((crop_vis.shape[0], 6, 3), 30, dtype=np.uint8)
    return np.hstack([crop_vis, sep, ref_vis])


def _load_ref_dino_input(
    weapon_id: str,
    weapon_type: str,
    cache_weapons_dir: str,
) -> tuple[np.ndarray | None, dict]:
    ref_path = _p(cache_weapons_dir, f"{weapon_id}.png")
    ref_img = _load_image(ref_path, cv2.IMREAD_UNCHANGED)
    if ref_img is None:
        return None, {"error": "missing_ref"}

    prepared_ref, meta = prepare_weapon_reference(ref_img, weapon_type=weapon_type)
    ref_dino = composite_bgra_on_background(prepared_ref)
    return ref_dino, meta


def _save_match_debug_images(
    prepared_crop_bgra: np.ndarray,
    weapon_crop_name: str,
    weapon_type: str,
    rarity: int | None,
    decision: DinoMatchDecision,
    debug_dir: str,
    cache_weapons_dir: str,
) -> dict:
    crops_dir = _p(debug_dir, "crops_dino")
    refs_dir = _p(debug_dir, "refs_dino", os.path.splitext(weapon_crop_name)[0])
    bucket_dir = _p(debug_dir, "accepted" if decision.accepted else "rejected")
    _ensure_dir(crops_dir)
    _ensure_dir(refs_dir)
    _ensure_dir(bucket_dir)

    crop_dino = composite_bgra_on_background(prepared_crop_bgra)

    base = _debug_base_name(
        weapon_crop_name=weapon_crop_name,
        weapon_type=weapon_type,
        rarity=rarity,
        best_id=decision.best_id,
        score=decision.best_score,
    )

    crop_path = os.path.join(crops_dir, base + ".png")
    cv2.imwrite(crop_path, crop_dino)

    best_ref_dino = None
    ref_paths = []
    for rank, item in enumerate(decision.top, start=1):
        ref_dino, _ = _load_ref_dino_input(
            weapon_id=item.ref_id,
            weapon_type=weapon_type,
            cache_weapons_dir=cache_weapons_dir,
        )
        if ref_dino is None:
            continue

        if rank == 1:
            best_ref_dino = ref_dino

        ref_name = f"rank_{rank:02d}__id_{item.ref_id}__s_{item.score:.4f}.png"
        ref_path = os.path.join(refs_dir, ref_name)
        cv2.imwrite(ref_path, ref_dino)
        ref_paths.append(ref_path)

    side_by_side = _make_side_by_side(
        crop_bgr=crop_dino,
        ref_bgr=best_ref_dino,
        crop_label=f"{os.path.splitext(weapon_crop_name)[0]} {weapon_type} r={rarity}",
        ref_label=f"best={decision.best_id} s={decision.best_score:.4f}",
    )
    pair_path = os.path.join(bucket_dir, base + "__pair.png")
    cv2.imwrite(pair_path, side_by_side)
    return {
        "crop_dino": crop_path,
        "refs_dino_dir": refs_dir,
        "ref_dino_top": ref_paths,
        "pair": pair_path,
    }


def match_weapons(
    parsed: dict,
    char_map: dict[str, str],
    chars_db: dict,
    weaps_db: dict,
    crops_weap_dir: str = "assets/weapons",
    out_hd_weap_dir: str = "assets/hd/weapons",
    cache_weapons_dir: str = "cache/enka_ref_weapons",
    debug_dir: str = "debug/weapons",
    top_k: int = 10,
    min_score: float = 0.0,
) -> dict:
    _ensure_dir(_p(out_hd_weap_dir))
    _ensure_dir(_p(debug_dir))
    _ensure_dir(_p(debug_dir, "crops_dino"))
    _ensure_dir(_p(debug_dir, "refs_dino"))
    _ensure_dir(_p(debug_dir, "accepted"))
    _ensure_dir(_p(debug_dir, "rejected"))

    type_rarity_to_ids = build_weapon_candidate_index(weaps_db)

    accepted_new_hd = 0
    accepted_total = 0
    rejected = 0
    skipped_no_char = 0
    skipped_no_weapon_type = 0
    skipped_no_crop = 0
    skipped_no_candidates = 0

    report = {
        "method": "dino_v2_prepared_refs_no_rerank",
        "accepted": [],
        "rejected": [],
        "skipped": [],
    }

    pairs = parsed.get("pairs", [])
    for pair in pairs:
        ci = int(pair["char_index"])
        wi = int(pair["weapon_index"])

        char_crop_name = f"char_{ci:03d}.png"
        weapon_crop_name = f"weapon_{wi:03d}.png"

        char_id = char_map.get(char_crop_name)
        if not char_id:
            skipped_no_char += 1
            report["skipped"].append({
                "crop": weapon_crop_name,
                "char_crop": char_crop_name,
                "reason": "no_char_match",
            })
            continue

        weapon_type = chars_db.get(str(char_id), {}).get("weapon_type")
        if not weapon_type:
            skipped_no_weapon_type += 1
            report["skipped"].append({
                "crop": weapon_crop_name,
                "char_crop": char_crop_name,
                "char_id": str(char_id),
                "reason": "no_weapon_type",
            })
            continue

        crop_path = _p(crops_weap_dir, weapon_crop_name)
        crop_bgr = _load_image(crop_path, cv2.IMREAD_COLOR)
        if crop_bgr is None:
            skipped_no_crop += 1
            report["skipped"].append({
                "crop": weapon_crop_name,
                "char_crop": char_crop_name,
                "weapon_type": str(weapon_type),
                "reason": "missing_crop",
            })
            continue

        try:
            _, prepared_crop, crop_meta = prepare_compensated_crop(crop_bgr)
        except Exception as exc:
            rejected += 1
            report["rejected"].append({
                "crop": weapon_crop_name,
                "char_crop": char_crop_name,
                "weapon_type": str(weapon_type),
                "reason": "crop_prepare_error",
                "error": str(exc),
            })
            continue

        rarity = crop_meta.get("rarity")
        if rarity is not None:
            try:
                rarity = int(rarity)
            except (TypeError, ValueError):
                rarity = None

        candidate_ids = get_candidate_ids(
            weapon_type=str(weapon_type),
            rarity=rarity,
            type_rarity_to_ids=type_rarity_to_ids,
            weaps_db=weaps_db,
        )
        if not candidate_ids:
            skipped_no_candidates += 1
            report["skipped"].append({
                "crop": weapon_crop_name,
                "char_crop": char_crop_name,
                "weapon_type": str(weapon_type),
                "rarity": rarity,
                "crop_meta": crop_meta,
                "reason": "no_candidates",
            })
            continue

        try:
            decision = match_prepared_weapon_crop(
                prepared_crop_bgra=prepared_crop,
                crop_meta=crop_meta,
                candidate_ids=candidate_ids,
                weapon_type=str(weapon_type),
                cache_weapons_dir=cache_weapons_dir,
                top_k=top_k,
                min_score=min_score,
            )
        except Exception as exc:
            rejected += 1
            report["rejected"].append({
                "crop": weapon_crop_name,
                "char_crop": char_crop_name,
                "weapon_type": str(weapon_type),
                "rarity": rarity,
                "crop_meta": crop_meta,
                "reason": "match_error",
                "error": str(exc),
            })
            continue

        report_item = _decision_to_report_item(
            weapon_crop_name=weapon_crop_name,
            char_crop_name=char_crop_name,
            weapon_type=str(weapon_type),
            rarity=rarity,
            decision=decision,
        )

        try:
            report_item["debug_images"] = _save_match_debug_images(
                prepared_crop_bgra=prepared_crop,
                weapon_crop_name=weapon_crop_name,
                weapon_type=str(weapon_type),
                rarity=rarity,
                decision=decision,
                debug_dir=debug_dir,
                cache_weapons_dir=cache_weapons_dir,
            )
        except Exception as exc:
            report_item["debug_save_error"] = str(exc)

        if not decision.accepted or decision.best_id is None:
            rejected += 1
            report["rejected"].append(report_item)
            continue

        accepted_total += 1
        report["accepted"].append(report_item)

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

    result = {
        "accepted_new_hd": accepted_new_hd,
        "accepted_total": accepted_total,
        "rejected": rejected,
        "skipped_no_char": skipped_no_char,
        "skipped_no_weapon_type": skipped_no_weapon_type,
        "skipped_no_crop": skipped_no_crop,
        "skipped_no_candidates": skipped_no_candidates,
        "skipped_low_rarity": 0,
        "pairs_total": len(pairs),
        "debug_dir": debug_dir,
        "method": "dino_v2_prepared_refs_no_rerank",
        "note": "temporary_ref_embeddings_are_computed_locally",
    }
    report["summary"] = result
    _write_report(debug_dir, report)
    return result


"""
README / TODO

This is the clean temporary weapon matcher v2.

Current behavior:
- Uses the parser pairs and character recognition results from the existing UI.
- Prepares runtime weapon crops through weapon_ref_preparer.prepare_compensated_crop.
- Prepares reference images through weapon_ref_preparer.prepare_weapon_reference.
- Runs DINOv2 only, without color rerank or edge/HSV legacy rerank.
- Computes reference embeddings locally from cache/enka_ref_weapons for now.
- Saves visual debug images in debug/weapons:
  crops_dino/ contains the exact crop images passed to DINO after hard alpha
  and gray compositing.
  refs_dino/<crop>/ contains the top reference images passed to DINO.
  accepted/ and rejected/ contain side-by-side crop/ref previews.
- Copies the matched raw Enka ref into assets/hd/weapons for current UI
  compatibility.

Important temporary parts:
- _encode_reference_candidates and _REF_EMBED_CACHE are debug-stage helpers.
  They should disappear once reference embeddings come from SQLite.
- _save_match_debug_images re-prepares top refs only for visual debugging. It
  should be disabled or guarded behind a debug flag in release builds.
- cache_weapons_dir currently points to raw Enka refs. Release code should use
  prepared refs from the data-pack/cache instead.
- min_score defaults to 0.0 so DINO top-1 is accepted for investigation. Real
  acceptance thresholds must be calibrated after debug runs.
- Weapon rarity comes from weapon_crop_extractor info. Do not re-add the old
  legacy HSV corner detector here.

Production work still needed:
- Load ref embeddings from the SQLite data-pack instead of computing them here.
- Lazy-download prepared refs for color rerank/UI when they are missing locally.
- Add optional color rerank as a separate module after DINO top-N.
- Add batch crop encoding so many weapon crops do not run one-by-one.
- Stop copying raw Enka refs to assets/hd/weapons once prepared refs become the
  canonical display/cache artifact.
- Keep legacy matchers out of runtime imports; they are only reference material.
"""
