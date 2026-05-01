from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass

import cv2
import numpy as np

from services.weapon_dino import (
    composite_bgra_on_background,
    cosine_topk,
    encode_weapon_bgra_dino,
)
from services.weapon_patch_dino_matcher import (
    PatchDinoFeatures,
    compare_patch_dino_features,
    encode_weapon_bgra_patch_dino,
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


def _alpha_mask_from_bgra(image_bgra: np.ndarray, alpha_threshold: int = 32) -> np.ndarray:
    if image_bgra is None or image_bgra.ndim != 3 or image_bgra.shape[2] != 4:
        return np.zeros((0, 0), dtype=np.uint8)
    return np.where(image_bgra[:, :, 3] >= int(alpha_threshold), 255, 0).astype(np.uint8)


@dataclass
class HybridMatchItem:
    ref_id: str
    score: float
    row_index: int
    metrics: dict


@dataclass
class HybridMatchDecision:
    best_id: str | None
    best_score: float
    accepted: bool
    top: list[HybridMatchItem]
    extra: dict


@dataclass
class RefShapeRecord:
    weapon_id: str
    weapon_type: str
    mask: np.ndarray
    prepared_bgra: np.ndarray
    dino_embedding: np.ndarray
    meta: dict


_REF_SHAPE_CACHE: dict[tuple[str, str, float, int], RefShapeRecord] = {}
_REF_PATCH_DINO_CACHE: dict[tuple[str, str], PatchDinoFeatures] = {}
GLOBAL_DINO_RERANK_TOP_K = 10
BOW_GLOBAL_DINO_RERANK_TOP_K = 15
PATCH_COLOR_RERANK_WINDOW = 0.020
RED_CLONE_RERANK_WINDOW = 0.045
PATCH_COLOR_SCORE_WEIGHT = 0.075
PATCH_COLOR_REJECT_PENALTY = 0.0
GLOBAL_DINO_TIEBREAK_WEIGHT = 0.03
DOMINANT_COLOR_MIN_FRACTION = 0.25
DOMINANT_COLOR_MATCH_BONUS = 0.18
DOMINANT_COLOR_MISMATCH_PENALTY = 0.25
DOMINANT_COLOR_MISMATCH_SCORE_PENALTY = 0.015
RED_CLONE_MISMATCH_PENALTY = 0.035
POLEARM_COLOR_STRUCTURE_PENALTY = 0.035


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


def _prepare_reference_shape(
    weapon_id: str,
    weapon_type: str,
    cache_weapons_dir: str,
) -> RefShapeRecord | None:
    ref_path = _p(cache_weapons_dir, f"{weapon_id}.png")
    if not os.path.exists(ref_path):
        return None

    key = _ref_cache_key(ref_path, weapon_id, weapon_type)
    cached = _REF_SHAPE_CACHE.get(key)
    if cached is not None:
        return cached

    ref_img = _load_image(ref_path, cv2.IMREAD_UNCHANGED)
    if ref_img is None:
        return None

    prepared_ref, meta = prepare_weapon_reference(ref_img, weapon_type=weapon_type)
    mask = _alpha_mask_from_bgra(prepared_ref)
    dino_embedding = encode_weapon_bgra_dino(prepared_ref)
    record = RefShapeRecord(
        weapon_id=str(weapon_id),
        weapon_type=str(weapon_type),
        mask=mask,
        prepared_bgra=prepared_ref,
        dino_embedding=dino_embedding,
        meta=meta,
    )
    _REF_SHAPE_CACHE[key] = record
    return record


def _prepare_reference_candidates(
    candidate_ids: list[str],
    weapon_type: str,
    cache_weapons_dir: str,
) -> tuple[list[np.ndarray], np.ndarray, list[str], dict[str, dict], int]:
    prepared_refs: list[np.ndarray] = []
    embeddings: list[np.ndarray] = []
    ref_ids = []
    ref_meta = {}
    skipped_refs = 0

    for weapon_id in candidate_ids:
        try:
            record = _prepare_reference_shape(
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

        prepared_refs.append(record.prepared_bgra)
        embeddings.append(record.dino_embedding.astype(np.float32))
        ref_ids.append(record.weapon_id)
        ref_meta[record.weapon_id] = record.meta

    if embeddings:
        ref_embeddings = np.stack(embeddings, axis=0).astype(np.float32)
    else:
        ref_embeddings = np.empty((0, 0), dtype=np.float32)

    return prepared_refs, ref_embeddings, ref_ids, ref_meta, skipped_refs


def _masked_hsv_stats(image_bgra: np.ndarray, alpha_threshold: int = 32) -> dict:
    if image_bgra.ndim != 3 or image_bgra.shape[2] != 4:
        raise ValueError("Expected BGRA image")

    mask = image_bgra[:, :, 3] >= int(alpha_threshold)
    pixels = image_bgra[:, :, :3][mask]
    if pixels.size == 0:
        return {
            "pixels": 0,
            "color_fractions": {},
            "median_hsv": [0.0, 0.0, 0.0],
            "hist": np.zeros((8, 4), dtype=np.float32),
        }

    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    h = hsv[:, 0].astype(np.float32)
    s = hsv[:, 1].astype(np.float32)
    v = hsv[:, 2].astype(np.float32)
    # Low-value shadows often get unstable blue/purple hue in HoYoLAB crops.
    # Keep only clear chroma pixels for color gating.
    colored = (s >= 55.0) & (v >= 70.0)
    denom = max(int(len(hsv)), 1)

    color_masks = {
        "red": colored & ((h <= 10.0) | (h >= 170.0)),
        "orange": colored & (h > 10.0) & (h < 22.0),
        "yellow": colored & (h >= 22.0) & (h <= 40.0),
        "green": colored & (h >= 45.0) & (h <= 88.0),
        "cyan": colored & (h > 88.0) & (h < 100.0),
        "blue": colored & (h >= 100.0) & (h <= 132.0),
        "purple": colored & (h > 132.0) & (h < 160.0),
        "pink": colored & (h >= 160.0) & (h < 170.0),
    }

    hsv_for_hist = hsv.astype(np.uint8).reshape(-1, 1, 3)
    hist = cv2.calcHist(
        [hsv_for_hist],
        [0, 1],
        None,
        [8, 4],
        [0, 180, 0, 256],
    ).astype(np.float32)
    hist_sum = float(hist.sum())
    if hist_sum > 1e-6:
        hist /= hist_sum

    return {
        "pixels": int(denom),
        "color_fractions": {
            name: float(np.count_nonzero(mask)) / float(denom)
            for name, mask in color_masks.items()
        },
        "colored_fraction": float(np.count_nonzero(colored)) / float(denom),
        "median_hsv": [
            float(np.median(h)),
            float(np.median(s)),
            float(np.median(v)),
        ],
        "hist": hist,
    }


def _color_family_fractions(color_fractions: dict[str, float]) -> dict[str, float]:
    return {
        "warm": (
            float(color_fractions.get("red", 0.0))
            + float(color_fractions.get("orange", 0.0))
            + float(color_fractions.get("yellow", 0.0))
        ),
        "green": float(color_fractions.get("green", 0.0)),
        "blue": (
            float(color_fractions.get("cyan", 0.0))
            + float(color_fractions.get("blue", 0.0))
        ),
        "purple": (
            float(color_fractions.get("purple", 0.0))
            + float(color_fractions.get("pink", 0.0))
        ),
    }


def _missing_color_accents(
    crop_families: dict[str, float],
    ref_families: dict[str, float],
) -> dict:
    thresholds = {
        # Warm/gold/wood is common and unstable after crop compression, so it
        # needs a bigger gap before it becomes a hard reject.
        "warm": {
            "ref_strong": 0.34,
            "ref_weak": 0.055,
            "ref_gap": 0.26,
            "crop_strong": 0.65,
            "crop_weak": 0.060,
            "crop_gap": 0.50,
        },
        "green": {
            "ref_strong": 0.16,
            "ref_weak": 0.035,
            "ref_gap": 0.12,
            "crop_strong": 0.45,
            "crop_weak": 0.040,
            "crop_gap": 0.35,
        },
        "blue": {
            "ref_strong": 0.28,
            "ref_weak": 0.18,
            "ref_gap": 0.30,
            "crop_strong": 0.55,
            "crop_weak": 0.140,
            "crop_gap": 0.45,
        },
        "purple": {
            "ref_strong": 0.16,
            "ref_weak": 0.045,
            "ref_gap": 0.12,
            "crop_strong": 0.45,
            "crop_weak": 0.050,
            "crop_gap": 0.35,
        },
    }

    missing = {}
    for name, rule in thresholds.items():
        crop_value = float(crop_families.get(name, 0.0))
        ref_value = float(ref_families.get(name, 0.0))

        if (
            ref_value >= float(rule["ref_strong"])
            and crop_value <= float(rule["ref_weak"])
            and (ref_value - crop_value) >= float(rule["ref_gap"])
        ):
            missing[f"ref_{name}"] = {
                "crop_fraction": crop_value,
                "ref_fraction": ref_value,
                "gap": ref_value - crop_value,
                "direction": "ref_has_color_missing_on_crop",
            }
        if (
            crop_value >= float(rule["crop_strong"])
            and ref_value <= float(rule["crop_weak"])
            and (crop_value - ref_value) >= float(rule["crop_gap"])
        ):
            missing[f"crop_{name}"] = {
                "crop_fraction": crop_value,
                "ref_fraction": ref_value,
                "gap": crop_value - ref_value,
                "direction": "crop_has_color_missing_on_ref",
            }

    return missing


def _color_metrics(crop_bgra: np.ndarray, ref_bgra: np.ndarray) -> dict:
    crop = _masked_hsv_stats(crop_bgra)
    ref = _masked_hsv_stats(ref_bgra)

    hist_intersection = float(np.minimum(crop["hist"], ref["hist"]).sum())
    crop_fractions = crop["color_fractions"]
    ref_fractions = ref["color_fractions"]
    crop_families = _color_family_fractions(crop_fractions)
    ref_families = _color_family_fractions(ref_fractions)

    missing_accents = _missing_color_accents(crop_families, ref_families)
    bidirectional_gaps = [
        abs(float(crop_families.get(name, 0.0)) - float(ref_families.get(name, 0.0)))
        for name in ("warm", "green", "blue", "purple")
    ]
    max_family_total = sum(
        max(float(crop_families.get(name, 0.0)), float(ref_families.get(name, 0.0)))
        for name in ("warm", "green", "blue", "purple")
    )
    family_overlap = sum(
        min(float(crop_families.get(name, 0.0)), float(ref_families.get(name, 0.0)))
        for name in ("warm", "green", "blue", "purple")
    )
    family_similarity = 1.0
    if max_family_total > 1e-6:
        family_similarity = family_overlap / max_family_total

    hard_rejected = bool(missing_accents)
    accent_penalty = min(0.45, 0.65 * max(bidirectional_gaps or [0.0]))
    color_score = float(np.clip((0.35 * hist_intersection + 0.65 * family_similarity) - accent_penalty, 0.0, 1.0))

    return {
        "score": color_score,
        "hist_intersection": hist_intersection,
        "family_similarity": float(family_similarity),
        "hard_rejected": hard_rejected,
        "missing_ref_accents": missing_accents,
        "max_family_gap": float(max(bidirectional_gaps or [0.0])),
        "crop": {
            "color_fractions": {
                name: float(value)
                for name, value in crop_fractions.items()
            },
            "color_families": {
                name: float(value)
                for name, value in crop_families.items()
            },
            "colored_fraction": float(crop.get("colored_fraction", 0.0)),
            "median_hsv": crop["median_hsv"],
        },
        "ref": {
            "color_fractions": {
                name: float(value)
                for name, value in ref_fractions.items()
            },
            "color_families": {
                name: float(value)
                for name, value in ref_families.items()
            },
            "colored_fraction": float(ref.get("colored_fraction", 0.0)),
            "median_hsv": ref["median_hsv"],
        },
    }


def _get_ref_patch_dino_features(
    ref_id: str,
    ref_bgra: np.ndarray,
    weapon_type: str,
) -> PatchDinoFeatures:
    key = (str(ref_id), str(weapon_type))
    cached = _REF_PATCH_DINO_CACHE.get(key)
    if cached is not None:
        return cached
    features = encode_weapon_bgra_patch_dino(ref_bgra, weapon_type=weapon_type)
    _REF_PATCH_DINO_CACHE[key] = features
    return features


def _global_dino_rerank_top_k(weapon_type: str) -> int:
    # Bow crops are the most device-sensitive in the current debug set, so let
    # patch-DINO see a slightly wider candidate pool for them.
    if str(weapon_type).lower() == "bow":
        return int(BOW_GLOBAL_DINO_RERANK_TOP_K)
    return int(GLOBAL_DINO_RERANK_TOP_K)


def _patch_color_adjusted_score(
    patch_score: float,
    dino_score: float,
    color: dict,
    best_patch_score: float,
    color_weight: float = PATCH_COLOR_SCORE_WEIGHT,
    weapon_type: str = "",
) -> tuple[float, dict]:
    patch_score = float(patch_score)
    dino_score = float(dino_score)
    best_patch_score = float(best_patch_score)
    patch_delta = max(0.0, best_patch_score - patch_score)
    in_window = patch_delta <= float(PATCH_COLOR_RERANK_WINDOW)
    in_red_clone_window = patch_delta <= float(RED_CLONE_RERANK_WINDOW)

    color_score = float(color.get("score", 0.0))
    effective_color_score = color_score
    dominant = _dominant_color_adjustment(color)
    effective_color_score = float(np.clip(color_score + dominant["adjustment"], 0.0, 1.0))

    global_bonus = float(GLOBAL_DINO_TIEBREAK_WEIGHT) * dino_score
    color_bonus = 0.0
    color_penalty = 0.0
    polearm_color_structure_penalty = 0.0
    if in_window:
        color_bonus = float(color_weight) * effective_color_score
        color_penalty += float(dominant.get("dominant_mismatch_penalty", 0.0))
        if str(weapon_type).lower() == "polearm":
            polearm_color_structure_penalty = _polearm_color_structure_penalty(color)
            color_penalty += polearm_color_structure_penalty
        if bool(color.get("hard_rejected", False)):
            color_penalty += float(PATCH_COLOR_REJECT_PENALTY)
    if in_red_clone_window:
        color_penalty += float(dominant.get("clone_penalty", 0.0))

    final_score = patch_score + global_bonus + color_bonus - color_penalty
    return float(final_score), {
        "enabled": bool(in_window),
        "red_clone_enabled": bool(in_red_clone_window),
        "patch_delta_from_best": float(patch_delta),
        "window": float(PATCH_COLOR_RERANK_WINDOW),
        "red_clone_window": float(RED_CLONE_RERANK_WINDOW),
        "global_bonus": float(global_bonus),
        "color_bonus": float(color_bonus),
        "color_weight": float(color_weight),
        "color_penalty": float(color_penalty),
        "polearm_color_structure_penalty": float(polearm_color_structure_penalty),
        "raw_color_score": float(color_score),
        "effective_color_score": float(effective_color_score),
        "dominant_color": dominant,
        "base_patch_score": float(patch_score),
    }


def _dominant_family(families: dict) -> tuple[str | None, float]:
    if not families:
        return None, 0.0
    name, value = max(
        ((str(k), float(v)) for k, v in families.items()),
        key=lambda item: item[1],
    )
    return name, float(value)


def _dominant_color_adjustment(color: dict) -> dict:
    crop_families = color.get("crop", {}).get("color_families", {}) or {}
    ref_families = color.get("ref", {}).get("color_families", {}) or {}
    crop_fractions = color.get("crop", {}).get("color_fractions", {}) or {}
    ref_fractions = color.get("ref", {}).get("color_fractions", {}) or {}
    crop_name, crop_value = _dominant_family(crop_families)
    ref_name, ref_value = _dominant_family(ref_families)

    adjustment = 0.0
    clone_penalty = 0.0
    dominant_mismatch_penalty = 0.0
    reason = "disabled"
    if (
        crop_name is not None
        and ref_name is not None
        and crop_value >= float(DOMINANT_COLOR_MIN_FRACTION)
        and ref_value >= float(DOMINANT_COLOR_MIN_FRACTION)
    ):
        if crop_name == ref_name:
            adjustment = float(DOMINANT_COLOR_MATCH_BONUS)
            reason = "dominant_family_match"
        else:
            adjustment = -float(DOMINANT_COLOR_MISMATCH_PENALTY)
            dominant_mismatch_penalty = float(DOMINANT_COLOR_MISMATCH_SCORE_PENALTY)
            reason = "dominant_family_mismatch"

    crop_red = float(crop_fractions.get("red", 0.0))
    ref_red = float(ref_fractions.get("red", 0.0))
    crop_nonred_warm = float(crop_fractions.get("orange", 0.0)) + float(crop_fractions.get("yellow", 0.0))
    ref_nonred_warm = float(ref_fractions.get("orange", 0.0)) + float(ref_fractions.get("yellow", 0.0))
    red_clone_reason = "none"
    if ref_red >= 0.18 and crop_red <= 0.04 and crop_nonred_warm >= 0.10:
        clone_penalty = max(clone_penalty, float(RED_CLONE_MISMATCH_PENALTY))
        red_clone_reason = "ref_red_missing_on_crop"
    elif crop_red >= 0.18 and ref_red <= 0.04 and ref_nonred_warm >= 0.10:
        clone_penalty = max(clone_penalty, float(RED_CLONE_MISMATCH_PENALTY))
        red_clone_reason = "crop_red_missing_on_ref"

    return {
        "adjustment": float(adjustment),
        "clone_penalty": float(clone_penalty),
        "dominant_mismatch_penalty": float(dominant_mismatch_penalty),
        "reason": reason,
        "red_clone_reason": red_clone_reason,
        "crop_red": float(crop_red),
        "ref_red": float(ref_red),
        "crop_nonred_warm": float(crop_nonred_warm),
        "ref_nonred_warm": float(ref_nonred_warm),
        "crop_family": crop_name,
        "crop_fraction": float(crop_value),
        "ref_family": ref_name,
        "ref_fraction": float(ref_value),
        "min_fraction": float(DOMINANT_COLOR_MIN_FRACTION),
    }


def _polearm_color_structure_penalty(color: dict) -> float:
    crop_families = color.get("crop", {}).get("color_families", {}) or {}
    ref_families = color.get("ref", {}).get("color_families", {}) or {}
    crop_nonwarm = (
        float(crop_families.get("green", 0.0))
        + float(crop_families.get("blue", 0.0))
        + float(crop_families.get("purple", 0.0))
    )
    ref_nonwarm = (
        float(ref_families.get("green", 0.0))
        + float(ref_families.get("blue", 0.0))
        + float(ref_families.get("purple", 0.0))
    )
    crop_warm = float(crop_families.get("warm", 0.0))
    ref_warm = float(ref_families.get("warm", 0.0))

    penalty = 0.0
    if crop_nonwarm >= 0.20 and ref_nonwarm <= 0.04:
        penalty += float(POLEARM_COLOR_STRUCTURE_PENALTY)
    if ref_warm - crop_warm >= 0.30:
        penalty += float(POLEARM_COLOR_STRUCTURE_PENALTY)
    return float(penalty)


def match_prepared_weapon_crop(
    prepared_crop_bgra: np.ndarray,
    crop_meta: dict,
    candidate_ids: list[str],
    weapon_type: str,
    cache_weapons_dir: str = "cache/enka_ref_weapons",
    top_k: int = 10,
    min_score: float = 0.0,
) -> HybridMatchDecision:
    if prepared_crop_bgra is None:
        return HybridMatchDecision(
            best_id=None,
            best_score=0.0,
            accepted=False,
            top=[],
            extra={"reason": "empty_crop"},
        )

    if not candidate_ids:
        return HybridMatchDecision(
            best_id=None,
            best_score=0.0,
            accepted=False,
            top=[],
            extra={"reason": "no_candidates"},
        )

    crop_embedding = encode_weapon_bgra_dino(prepared_crop_bgra)

    ref_bgras, ref_embeddings, ref_ids, ref_meta, skipped_refs = _prepare_reference_candidates(
        candidate_ids=candidate_ids,
        weapon_type=weapon_type,
        cache_weapons_dir=cache_weapons_dir,
    )
    if len(ref_ids) == 0:
        return HybridMatchDecision(
            best_id=None,
            best_score=0.0,
            accepted=False,
            top=[],
            extra={
                "reason": "no_prepared_refs",
                "candidate_count": len(candidate_ids),
                "skipped_refs": skipped_refs,
                "crop_meta": crop_meta,
            },
        )

    dino_top = cosine_topk(
        query_embedding=crop_embedding,
        ref_embeddings=ref_embeddings,
        ref_ids=ref_ids,
        top_k=min(_global_dino_rerank_top_k(weapon_type), len(ref_ids)),
    )
    if not dino_top:
        return HybridMatchDecision(
            best_id=None,
            best_score=0.0,
            accepted=False,
            top=[],
            extra={
                "reason": "empty_dino_topk",
                "candidate_count": len(candidate_ids),
                "prepared_ref_count": len(ref_ids),
                "skipped_refs": skipped_refs,
                "crop_meta": crop_meta,
            },
        )

    id_to_index = {ref_id: idx for idx, ref_id in enumerate(ref_ids)}
    hybrid_items: list[HybridMatchItem] = []
    crop_patch_features = encode_weapon_bgra_patch_dino(
        prepared_crop_bgra,
        weapon_type=weapon_type,
    )
    for dino_item in dino_top:
        ref_idx = id_to_index.get(dino_item.ref_id)
        if ref_idx is None:
            continue

        ref_patch_features = _get_ref_patch_dino_features(
            ref_id=dino_item.ref_id,
            ref_bgra=ref_bgras[ref_idx],
            weapon_type=weapon_type,
        )
        patch_dino = compare_patch_dino_features(crop_patch_features, ref_patch_features)
        color = _color_metrics(prepared_crop_bgra, ref_bgras[ref_idx])

        hybrid_items.append(
            HybridMatchItem(
                ref_id=dino_item.ref_id,
                score=float(patch_dino["score"]),
                row_index=int(dino_item.row_index),
                metrics={
                    "dino_score": float(dino_item.score),
                    "patch_dino": patch_dino,
                    "color": color,
                    "color_rejected": bool(color.get("hard_rejected", False)),
                    "stage": "dino_top10_patch_dino_color_tiebreak",
                },
            )
        )

    if hybrid_items:
        best_patch_score = max(float(item.metrics["patch_dino"]["score"]) for item in hybrid_items)
        for item in hybrid_items:
            color_weight = 0.0 if str(weapon_type).lower() == "polearm" else float(PATCH_COLOR_SCORE_WEIGHT)
            adjusted_score, color_rerank = _patch_color_adjusted_score(
                patch_score=float(item.metrics["patch_dino"]["score"]),
                dino_score=float(item.metrics["dino_score"]),
                color=item.metrics["color"],
                best_patch_score=best_patch_score,
                color_weight=color_weight,
                weapon_type=weapon_type,
            )
            item.score = adjusted_score
            item.metrics["color_rerank"] = color_rerank

    hybrid_items.sort(key=lambda item: item.score, reverse=True)
    top = hybrid_items[:max(0, int(top_k))]
    if not top:
        return HybridMatchDecision(
            best_id=None,
            best_score=0.0,
            accepted=False,
            top=[],
            extra={
                "reason": "empty_hybrid_topk",
                "candidate_count": len(candidate_ids),
                "prepared_ref_count": len(ref_ids),
                "skipped_refs": skipped_refs,
                "crop_meta": crop_meta,
            },
        )

    best = top[0]
    return HybridMatchDecision(
        best_id=best.ref_id,
        best_score=float(best.score),
        accepted=bool(best.score >= float(min_score)),
        top=top,
        extra={
            "candidate_count": len(candidate_ids),
            "prepared_ref_count": len(ref_ids),
            "skipped_refs": skipped_refs,
            "crop_meta": crop_meta,
            "dino_top": [
                {
                    "id": item.ref_id,
                    "score": float(item.score),
                    "row_index": int(item.row_index),
                }
                for item in dino_top
            ],
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
) -> HybridMatchDecision:
    if crop_bgr is None:
        return HybridMatchDecision(
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
    decision: HybridMatchDecision,
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
                "metrics": item.metrics,
            }
            for item in decision.top
        ],
        "extra": decision.extra,
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return value
    return value


def _write_report(debug_dir: str, report: dict):
    _ensure_dir(_p(debug_dir))
    path = _p(debug_dir, "report_v2_hybrid.json")
    try:
        text = json.dumps(_json_safe(report), ensure_ascii=False, indent=2)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
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


def _composite_bgra_on_gray(image_bgra: np.ndarray, bg_bgr: tuple[int, int, int] = (127, 127, 127)) -> np.ndarray:
    if image_bgra.ndim != 3 or image_bgra.shape[2] != 4:
        raise ValueError("Expected BGRA image")
    bgr = image_bgra[:, :, :3].astype(np.float32)
    alpha = (image_bgra[:, :, 3] >= 32).astype(np.float32)
    bg = np.empty_like(bgr, dtype=np.float32)
    bg[:, :, 0] = float(bg_bgr[0])
    bg[:, :, 1] = float(bg_bgr[1])
    bg[:, :, 2] = float(bg_bgr[2])
    out = bgr * alpha[:, :, None] + bg * (1.0 - alpha[:, :, None])
    return np.clip(out, 0, 255).astype(np.uint8)


def _load_ref_shape_input(
    weapon_id: str,
    weapon_type: str,
    cache_weapons_dir: str,
) -> tuple[np.ndarray | None, dict]:
    record = _prepare_reference_shape(
        weapon_id=weapon_id,
        weapon_type=weapon_type,
        cache_weapons_dir=cache_weapons_dir,
    )
    if record is None:
        return None, {"error": "missing_ref"}
    return _composite_bgra_on_gray(record.prepared_bgra), record.meta


def _save_match_debug_images(
    prepared_crop_bgra: np.ndarray,
    weapon_crop_name: str,
    weapon_type: str,
    rarity: int | None,
    decision: HybridMatchDecision,
    debug_dir: str,
    cache_weapons_dir: str,
) -> dict:
    crops_dir = _p(debug_dir, "crops_shape")
    crop_masks_dir = _p(debug_dir, "crop_masks")
    crops_dino_dir = _p(debug_dir, "crops_dino")
    refs_dir = _p(debug_dir, "refs_shape", os.path.splitext(weapon_crop_name)[0])
    ref_masks_dir = _p(debug_dir, "ref_masks", os.path.splitext(weapon_crop_name)[0])
    refs_dino_dir = _p(debug_dir, "refs_dino", os.path.splitext(weapon_crop_name)[0])
    bucket_dir = _p(debug_dir, "accepted" if decision.accepted else "rejected")
    _ensure_dir(crops_dir)
    _ensure_dir(crop_masks_dir)
    _ensure_dir(crops_dino_dir)
    _ensure_dir(refs_dir)
    _ensure_dir(ref_masks_dir)
    _ensure_dir(refs_dino_dir)
    _ensure_dir(bucket_dir)

    crop_shape = _composite_bgra_on_gray(prepared_crop_bgra)
    crop_mask = _alpha_mask_from_bgra(prepared_crop_bgra)
    crop_dino = composite_bgra_on_background(prepared_crop_bgra)

    base = _debug_base_name(
        weapon_crop_name=weapon_crop_name,
        weapon_type=weapon_type,
        rarity=rarity,
        best_id=decision.best_id,
        score=decision.best_score,
    )

    crop_path = os.path.join(crops_dir, base + ".png")
    crop_mask_path = os.path.join(crop_masks_dir, base + ".png")
    crop_dino_path = os.path.join(crops_dino_dir, base + ".png")
    cv2.imwrite(crop_path, crop_shape)
    cv2.imwrite(crop_mask_path, crop_mask)
    cv2.imwrite(crop_dino_path, crop_dino)

    best_ref_shape = None
    ref_paths = []
    ref_mask_paths = []
    ref_dino_paths = []
    for rank, item in enumerate(decision.top, start=1):
        ref_shape, _ = _load_ref_shape_input(
            weapon_id=item.ref_id,
            weapon_type=weapon_type,
            cache_weapons_dir=cache_weapons_dir,
        )
        record = _prepare_reference_shape(item.ref_id, weapon_type, cache_weapons_dir)
        if ref_shape is None or record is None:
            continue

        if rank == 1:
            best_ref_shape = ref_shape

        ref_name = f"rank_{rank:02d}__id_{item.ref_id}__s_{item.score:.4f}.png"
        ref_path = os.path.join(refs_dir, ref_name)
        ref_mask_path = os.path.join(ref_masks_dir, ref_name)
        ref_dino_path = os.path.join(refs_dino_dir, ref_name)
        cv2.imwrite(ref_path, ref_shape)
        cv2.imwrite(ref_mask_path, record.mask)
        cv2.imwrite(ref_dino_path, composite_bgra_on_background(record.prepared_bgra))
        ref_paths.append(ref_path)
        ref_mask_paths.append(ref_mask_path)
        ref_dino_paths.append(ref_dino_path)

    side_by_side = _make_side_by_side(
        crop_bgr=crop_shape,
        ref_bgr=best_ref_shape,
        crop_label=f"{os.path.splitext(weapon_crop_name)[0]} {weapon_type} r={rarity}",
        ref_label=f"best={decision.best_id} s={decision.best_score:.4f}",
    )
    pair_path = os.path.join(bucket_dir, base + "__pair.png")
    cv2.imwrite(pair_path, side_by_side)
    return {
        "crop_shape": crop_path,
        "crop_mask": crop_mask_path,
        "crop_dino": crop_dino_path,
        "refs_shape_dir": refs_dir,
        "ref_masks_dir": ref_masks_dir,
        "refs_dino_dir": refs_dino_dir,
        "ref_shape_top": ref_paths,
        "ref_mask_top": ref_mask_paths,
        "ref_dino_top": ref_dino_paths,
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
    _ensure_dir(_p(debug_dir, "crops_shape"))
    _ensure_dir(_p(debug_dir, "crop_masks"))
    _ensure_dir(_p(debug_dir, "crops_dino"))
    _ensure_dir(_p(debug_dir, "refs_shape"))
    _ensure_dir(_p(debug_dir, "ref_masks"))
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
        "method": "dino_top10_patch_dino_color_tiebreak",
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
        "method": "dino_top10_patch_dino_color_tiebreak",
        "note": "temporary_dino_embeddings_are_computed_locally",
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
- Runs DINOv2 first and keeps the top reference candidates.
  Bows use a slightly wider top-N because they are more device-sensitive.
- Reranks that DINO top 10 with patch-level DINO tokens.
- Applies color only as a narrow tie-breaker for candidates close to the best
  patch-level DINO score. It should separate recolored clones without
  overriding a confident shape match.
- Computes temporary DINO embeddings locally from cache/enka_ref_weapons for now.
- Saves visual debug images in debug/weapons:
  crops_shape/ contains prepared crop images on a gray background.
  crop_masks/ contains the binary crop alpha masks.
  crops_dino/ contains the crop images exactly as DINO sees them.
  refs_shape/<crop>/ contains the top prepared reference images.
  ref_masks/<crop>/ contains the top reference alpha masks.
  refs_dino/<crop>/ contains the top reference images exactly as DINO sees them.
  accepted/ and rejected/ contain side-by-side crop/ref previews.
- Copies the matched raw Enka ref into assets/hd/weapons for current UI
  compatibility.

Important temporary parts:
- _prepare_reference_candidates and _REF_SHAPE_CACHE are debug-stage helpers.
  They should disappear once prepared refs and DINO embeddings come from the
  SQLite/data-pack layer.
- _save_match_debug_images re-prepares top refs only for visual debugging. It
  should be disabled or guarded behind a debug flag in release builds.
- cache_weapons_dir currently points to raw Enka refs. Release code should use
  prepared refs from the data-pack/cache instead.
- min_score defaults to 0.0 so hybrid top-1 is accepted for investigation. Real
  acceptance thresholds must be calibrated after debug runs.
- Weapon rarity comes from weapon_crop_extractor info. Do not re-add the old
  legacy HSV corner detector here.

Production work still needed:
- Load prepared refs and precomputed DINO embeddings from the SQLite/data-pack
  layer instead of computing them here.
- Lazy-download prepared refs for color rerank/UI when they are missing locally.
- Decide whether patch-level DINO survives as a rerank stage. If not, move to a
  small trained Siamese/metric-learning model.
- Stop copying raw Enka refs to assets/hd/weapons once prepared refs become the
  canonical display/cache artifact.
- Legacy and experimental shape matchers were removed from the runtime tree.
"""
