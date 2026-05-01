from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from services.weapon_dino import (
    DEFAULT_ALPHA_THRESHOLD,
    DEFAULT_BG_BGR,
    DINO_MODEL_NAME,
    _get_dino,
    composite_bgra_on_background,
    l2_normalize,
    standardize_weapon_bgra_for_dino,
)


DEFAULT_QUERY_KEEP_FRACTION = 0.82
DEFAULT_REF_KEEP_FRACTION = 0.68
DEFAULT_MIN_PATCH_ALPHA = 0.08
POLEARM_MIN_HEAD_PATCHES = 18


@dataclass
class PatchDinoFeatures:
    tokens: np.ndarray
    patch_mask: np.ndarray
    grid_shape: tuple[int, int]
    token_count: int
    active_token_count: int
    mask_strategy: str = "alpha"


def _extract_patch_tokens(forward_output) -> np.ndarray:
    if isinstance(forward_output, dict):
        for key in ("x_norm_patchtokens", "patch_tokens", "tokens"):
            value = forward_output.get(key)
            if value is not None:
                return value
        value = forward_output.get("x")
        if value is not None:
            return value

    if isinstance(forward_output, (tuple, list)):
        for value in forward_output:
            if hasattr(value, "ndim") and int(value.ndim) == 3:
                return value

    return forward_output


def _drop_cls_token_if_present(tokens: np.ndarray, grid_area: int | None = None) -> np.ndarray:
    if tokens.ndim != 3:
        raise ValueError(f"Expected DINO patch tokens with shape BxNxC, got {tokens.shape}")
    if tokens.shape[0] != 1:
        raise ValueError(f"Expected batch size 1, got {tokens.shape[0]}")

    out = tokens[0]
    n_tokens = int(out.shape[0])
    if grid_area is not None:
        if n_tokens == grid_area:
            return out
        if n_tokens == grid_area + 1:
            return out[1:]

    side = int(round(np.sqrt(float(n_tokens))))
    if side * side == n_tokens:
        return out
    side_with_cls = int(round(np.sqrt(float(max(0, n_tokens - 1)))))
    if side_with_cls * side_with_cls == n_tokens - 1:
        return out[1:]
    return out


def _infer_grid_shape(token_count: int, image_hw: tuple[int, int], patch_size: int | None) -> tuple[int, int]:
    h, w = image_hw
    if patch_size:
        gh = max(1, int(round(float(h) / float(patch_size))))
        gw = max(1, int(round(float(w) / float(patch_size))))
        if gh * gw == int(token_count):
            return gh, gw

    side = int(round(np.sqrt(float(token_count))))
    if side * side == int(token_count):
        return side, side

    # Fallback for unusual model configs. DINOv2 in this project is square, but
    # keep the failure mode explicit instead of silently reshaping wrong.
    raise ValueError(f"Cannot infer patch grid for {token_count} tokens")


def _patch_alpha_mask(
    image_bgra: np.ndarray,
    grid_shape: tuple[int, int],
    alpha_threshold: int = DEFAULT_ALPHA_THRESHOLD,
    min_patch_alpha: float = DEFAULT_MIN_PATCH_ALPHA,
) -> np.ndarray:
    bgra = standardize_weapon_bgra_for_dino(image_bgra, alpha_threshold=alpha_threshold)
    alpha = (bgra[:, :, 3] >= int(alpha_threshold)).astype(np.float32)
    gh, gw = grid_shape
    small = cv2.resize(alpha, (gw, gh), interpolation=cv2.INTER_AREA)
    return (small >= float(min_patch_alpha)).reshape(-1)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return binary.astype(bool)

    best_label = 1
    best_area = int(stats[1, cv2.CC_STAT_AREA])
    for label in range(2, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area > best_area:
            best_area = area
            best_label = label
    return labels == best_label


def _polearm_head_patch_mask(
    image_bgra: np.ndarray,
    grid_shape: tuple[int, int],
    alpha_threshold: int = DEFAULT_ALPHA_THRESHOLD,
    min_patch_alpha: float = DEFAULT_MIN_PATCH_ALPHA,
) -> np.ndarray | None:
    bgra = standardize_weapon_bgra_for_dino(image_bgra, alpha_threshold=alpha_threshold)
    binary = np.where(bgra[:, :, 3] >= int(alpha_threshold), 255, 0).astype(np.uint8)
    if int(np.count_nonzero(binary)) == 0:
        return None

    # Polearm shafts are long and thin, so their distance-to-background stays
    # low. The head/guard usually has a thicker core. Keep that core and dilate
    # it back to a useful local region for patch-token matching.
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    fg_dist = dist[binary > 0]
    if fg_dist.size == 0:
        return None

    threshold = max(3.0, float(np.percentile(fg_dist, 72.0)))
    core = dist >= threshold
    if int(np.count_nonzero(core)) < 64:
        threshold = max(2.0, float(np.percentile(fg_dist, 58.0)))
        core = dist >= threshold

    if int(np.count_nonzero(core)) == 0:
        return None

    head_core = _largest_component(core)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    head_region = cv2.dilate(head_core.astype(np.uint8) * 255, kernel, iterations=1)
    head_region = (head_region > 0) & (binary > 0)

    gh, gw = grid_shape
    alpha_small = cv2.resize((binary > 0).astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA)
    head_small = cv2.resize(head_region.astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA)
    patch_mask = (alpha_small >= float(min_patch_alpha)) & (head_small >= 0.02)
    flat = patch_mask.reshape(-1)
    if int(np.count_nonzero(flat)) < int(POLEARM_MIN_HEAD_PATCHES):
        return None
    return flat.astype(bool)


def encode_weapon_bgra_patch_dino(
    image_bgra: np.ndarray,
    weapon_type: str | None = None,
    bg_bgr: tuple[int, int, int] = DEFAULT_BG_BGR,
    alpha_threshold: int = DEFAULT_ALPHA_THRESHOLD,
    model_name: str = DINO_MODEL_NAME,
    pretrained: bool = True,
    min_patch_alpha: float = DEFAULT_MIN_PATCH_ALPHA,
) -> PatchDinoFeatures:
    import torch
    from PIL import Image

    model, transform, device = _get_dino(
        model_name=model_name,
        pretrained=pretrained,
    )

    image_bgr = composite_bgra_on_background(
        image_bgra,
        bg_bgr=bg_bgr,
        alpha_threshold=alpha_threshold,
    )
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    x = transform(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        if hasattr(model, "forward_features"):
            raw = model.forward_features(x)
        else:
            raw = model.forward_intermediates(x)[0]

    raw_tokens = _extract_patch_tokens(raw)
    if hasattr(raw_tokens, "detach"):
        raw_tokens = raw_tokens.detach().cpu().numpy()
    raw_tokens = np.asarray(raw_tokens, dtype=np.float32)

    patch_size = getattr(getattr(model, "patch_embed", None), "patch_size", None)
    if isinstance(patch_size, tuple):
        patch_size = int(patch_size[0])
    elif patch_size is not None:
        patch_size = int(patch_size)

    # Try the common square-grid case before cls-token removal.
    token_count_raw = int(raw_tokens.shape[1]) if raw_tokens.ndim == 3 else 0
    side = int(round(np.sqrt(float(token_count_raw))))
    grid_area_guess = side * side if side * side == token_count_raw else None
    if grid_area_guess is None and token_count_raw > 1:
        side_no_cls = int(round(np.sqrt(float(token_count_raw - 1))))
        if side_no_cls * side_no_cls == token_count_raw - 1:
            grid_area_guess = side_no_cls * side_no_cls

    tokens = _drop_cls_token_if_present(raw_tokens, grid_area=grid_area_guess)
    grid_shape = _infer_grid_shape(
        token_count=int(tokens.shape[0]),
        image_hw=(int(x.shape[-2]), int(x.shape[-1])),
        patch_size=patch_size,
    )
    patch_mask = _patch_alpha_mask(
        image_bgra,
        grid_shape=grid_shape,
        alpha_threshold=alpha_threshold,
        min_patch_alpha=min_patch_alpha,
    )
    mask_strategy = "alpha"
    if str(weapon_type or "").lower() == "polearm":
        polearm_mask = _polearm_head_patch_mask(
            image_bgra,
            grid_shape=grid_shape,
            alpha_threshold=alpha_threshold,
            min_patch_alpha=min_patch_alpha,
        )
        if polearm_mask is not None:
            patch_mask = polearm_mask
            mask_strategy = "polearm_head"

    if int(np.count_nonzero(patch_mask)) == 0:
        patch_mask = np.ones((int(tokens.shape[0]),), dtype=bool)
        mask_strategy = "fallback_all"

    tokens = np.asarray([l2_normalize(row) for row in tokens], dtype=np.float32)
    active = tokens[patch_mask]
    return PatchDinoFeatures(
        tokens=active,
        patch_mask=patch_mask.astype(bool),
        grid_shape=grid_shape,
        token_count=int(tokens.shape[0]),
        active_token_count=int(active.shape[0]),
        mask_strategy=mask_strategy,
    )


def _trimmed_best_direction_score(
    source_tokens: np.ndarray,
    target_tokens: np.ndarray,
    keep_fraction: float,
) -> dict:
    if source_tokens.size == 0 or target_tokens.size == 0:
        return {
            "score": 0.0,
            "mean": 0.0,
            "trimmed_mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "source_count": int(source_tokens.shape[0]) if source_tokens.ndim == 2 else 0,
            "target_count": int(target_tokens.shape[0]) if target_tokens.ndim == 2 else 0,
        }

    sim = np.asarray(source_tokens, dtype=np.float32) @ np.asarray(target_tokens, dtype=np.float32).T
    best = np.max(sim, axis=1)
    best_sorted = np.sort(best)[::-1]
    keep = max(1, int(round(float(keep_fraction) * float(len(best_sorted)))))
    trimmed = best_sorted[:keep]
    return {
        "score": float(np.clip(np.mean(trimmed), -1.0, 1.0)),
        "mean": float(np.mean(best)),
        "trimmed_mean": float(np.mean(trimmed)),
        "median": float(np.median(best)),
        "p90": float(np.percentile(best, 90.0)),
        "source_count": int(source_tokens.shape[0]),
        "target_count": int(target_tokens.shape[0]),
    }


def compare_patch_dino_features(
    query: PatchDinoFeatures,
    ref: PatchDinoFeatures,
    query_keep_fraction: float = DEFAULT_QUERY_KEEP_FRACTION,
    ref_keep_fraction: float = DEFAULT_REF_KEEP_FRACTION,
) -> dict:
    query_to_ref = _trimmed_best_direction_score(
        query.tokens,
        ref.tokens,
        keep_fraction=query_keep_fraction,
    )
    ref_to_query = _trimmed_best_direction_score(
        ref.tokens,
        query.tokens,
        keep_fraction=ref_keep_fraction,
    )

    query_weight = 0.72
    ref_weight = 0.28
    if query.mask_strategy == "polearm_head" and ref.mask_strategy == "polearm_head":
        # For polearms, the crop head is the signal. Ref-only decoration/shaft
        # pieces should not outweigh whether the crop head finds matching local
        # details on the ref.
        query_weight = 0.85
        ref_weight = 0.15

    # Asymmetric on purpose: crop patches must be explained by the ref; ref-only
    # details such as bow strings and polearm shaft remnants should not dominate.
    score = float(
        np.clip(
            query_weight * float(query_to_ref["score"])
            + ref_weight * float(ref_to_query["score"]),
            -1.0,
            1.0,
        )
    )
    return {
        "score": score,
        "query_to_ref": query_to_ref,
        "ref_to_query": ref_to_query,
        "query_active_tokens": int(query.active_token_count),
        "ref_active_tokens": int(ref.active_token_count),
        "query_grid_shape": list(query.grid_shape),
        "ref_grid_shape": list(ref.grid_shape),
        "query_mask_strategy": str(query.mask_strategy),
        "ref_mask_strategy": str(ref.mask_strategy),
        "query_weight": float(query_weight),
        "ref_weight": float(ref_weight),
        "matcher": "patch_level_dino",
    }


def compare_weapon_bgra_patch_dino(
    query_bgra: np.ndarray,
    ref_bgra: np.ndarray,
    weapon_type: str | None = None,
) -> dict:
    query = encode_weapon_bgra_patch_dino(query_bgra, weapon_type=weapon_type)
    ref = encode_weapon_bgra_patch_dino(ref_bgra, weapon_type=weapon_type)
    return compare_patch_dino_features(query, ref)


"""
README / TODO

This module is an experimental patch-token DINO matcher.

Current behavior:
- Uses DINOv2 patch tokens from timm forward_features instead of the global
  image embedding.
- Filters patch tokens by the prepared weapon alpha mask.
- For polearms, can filter patch tokens down to the thicker head/guard region
  so the long shaft contributes much less to reranking.
- Scores local similarity by asking each crop patch for its nearest ref patch,
  and each ref patch for its nearest crop patch.
- Uses asymmetric scoring so missing ref-only details, like bow strings, are
  penalized less than crop details that cannot be found on the ref.
- Does not use color, contour maps, filesystem indexes, SQLite, metadata, or UI.

Production work still needed:
- Cache patch tokens for prepared refs in the data-pack/SQLite layer.
- Batch-encode refs/crops; this temporary implementation encodes one image at a
  time and is only for quality experiments.
- Tune keep fractions and token masking after comparing real debug runs.
- Decide whether this survives as rerank stage or gets replaced by a small
  trained Siamese/metric-learning model.
"""
