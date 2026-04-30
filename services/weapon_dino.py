from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


DINO_MODEL_NAME = "vit_base_patch14_dinov2.lvd142m"
DEFAULT_ALPHA_THRESHOLD = 32
DEFAULT_BG_BGR = (127, 127, 127)


@dataclass
class DinoTopKItem:
    ref_id: str
    score: float
    row_index: int


_DEVICE = None
_DINO = None
_DINO_TRANSFORM = None


def ensure_bgra_image(img: np.ndarray) -> np.ndarray:
    if img is None:
        raise ValueError("Expected image, got None")

    if img.ndim == 2:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        alpha = np.where(img > 0, 255, 0).astype(np.uint8)
        return np.dstack([bgr, alpha])

    if img.ndim != 3:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    if img.shape[2] == 4:
        return img

    if img.shape[2] == 3:
        bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = 255
        return bgra

    raise ValueError(f"Unsupported image shape: {img.shape}")


def standardize_weapon_bgra_for_dino(
    image_bgra: np.ndarray,
    alpha_threshold: int = DEFAULT_ALPHA_THRESHOLD,
) -> np.ndarray:
    bgra = ensure_bgra_image(image_bgra).copy()
    alpha_threshold = int(np.clip(alpha_threshold, 0, 255))

    alpha = bgra[:, :, 3]
    hard_alpha = np.where(alpha >= alpha_threshold, 255, 0).astype(np.uint8)
    bgra[:, :, 3] = hard_alpha
    bgra[hard_alpha == 0, :3] = 0
    return bgra


def composite_bgra_on_background(
    image_bgra: np.ndarray,
    bg_bgr: tuple[int, int, int] = DEFAULT_BG_BGR,
    alpha_threshold: int = DEFAULT_ALPHA_THRESHOLD,
) -> np.ndarray:
    bgra = standardize_weapon_bgra_for_dino(
        image_bgra,
        alpha_threshold=alpha_threshold,
    )

    bgr = bgra[:, :, :3].astype(np.float32)
    alpha = bgra[:, :, 3].astype(np.float32) / 255.0

    bg = np.empty_like(bgr, dtype=np.float32)
    bg[:, :, 0] = float(bg_bgr[0])
    bg[:, :, 1] = float(bg_bgr[1])
    bg[:, :, 2] = float(bg_bgr[2])

    out = bgr * alpha[:, :, None] + bg * (1.0 - alpha[:, :, None])
    return np.clip(out, 0, 255).astype(np.uint8)


def l2_normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    norm = float(np.linalg.norm(v))
    if norm <= 1e-12:
        return v
    return v / norm


def _get_dino(
    model_name: str = DINO_MODEL_NAME,
    pretrained: bool = True,
):
    global _DEVICE, _DINO, _DINO_TRANSFORM
    if _DINO is not None:
        return _DINO, _DINO_TRANSFORM, _DEVICE

    import torch
    import timm

    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    _DINO = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=0,
    )
    _DINO.eval()
    _DINO.to(_DEVICE)

    data_cfg = timm.data.resolve_model_data_config(_DINO)
    _DINO_TRANSFORM = timm.data.create_transform(**data_cfg, is_training=False)
    return _DINO, _DINO_TRANSFORM, _DEVICE


def encode_bgr_dino(
    image_bgr: np.ndarray,
    model_name: str = DINO_MODEL_NAME,
    pretrained: bool = True,
) -> np.ndarray:
    if image_bgr is None:
        raise ValueError("Expected BGR image, got None")

    import torch
    from PIL import Image

    model, transform, device = _get_dino(
        model_name=model_name,
        pretrained=pretrained,
    )

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    x = transform(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model(x)

    if isinstance(feat, (tuple, list)):
        feat = feat[0]

    emb = feat.squeeze(0).detach().cpu().numpy().astype(np.float32)
    return l2_normalize(emb)


def encode_weapon_bgra_dino(
    image_bgra: np.ndarray,
    bg_bgr: tuple[int, int, int] = DEFAULT_BG_BGR,
    alpha_threshold: int = DEFAULT_ALPHA_THRESHOLD,
    model_name: str = DINO_MODEL_NAME,
    pretrained: bool = True,
) -> np.ndarray:
    image_bgr = composite_bgra_on_background(
        image_bgra,
        bg_bgr=bg_bgr,
        alpha_threshold=alpha_threshold,
    )
    return encode_bgr_dino(
        image_bgr,
        model_name=model_name,
        pretrained=pretrained,
    )


def cosine_scores(
    query_embedding: np.ndarray,
    ref_embeddings: np.ndarray,
) -> np.ndarray:
    query = l2_normalize(query_embedding)
    refs = np.asarray(ref_embeddings, dtype=np.float32)
    if refs.ndim != 2:
        raise ValueError(f"Expected 2D ref embeddings, got shape {refs.shape}")
    return refs @ query


def cosine_topk(
    query_embedding: np.ndarray,
    ref_embeddings: np.ndarray,
    ref_ids: Iterable[str],
    top_k: int = 10,
) -> list[DinoTopKItem]:
    ids = [str(x) for x in ref_ids]
    refs = np.asarray(ref_embeddings, dtype=np.float32)
    if refs.shape[0] != len(ids):
        raise ValueError(
            f"ref_embeddings rows ({refs.shape[0]}) != ref_ids ({len(ids)})"
        )

    scores = cosine_scores(query_embedding, refs)
    if scores.size == 0:
        return []

    k = max(0, min(int(top_k), int(scores.size)))
    if k == 0:
        return []

    idx = np.argpartition(-scores, kth=k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [
        DinoTopKItem(ref_id=ids[int(i)], score=float(scores[int(i)]), row_index=int(i))
        for i in idx
    ]


"""
README / TODO

This module is intentionally only the DINO image/embedding layer.

Current temporary scope:
- Accepts already prepared BGRA weapon images from the crop extractor or ref
  preparer.
- Hardens alpha with DEFAULT_ALPHA_THRESHOLD before model input.
- Composites the object on a fixed gray background before DINO.
- Encodes one image at a time with timm/torch.
- Provides an in-memory cosine_topk helper for local debug runners.

Production work still needed:
- Move reference embeddings to SQLite and load/search them from there.
- Add a batch encoder for many runtime crops instead of one-by-one calls.
- Add local model weight loading so timm does not download weights at runtime.
- Decide whether the release build keeps timm/torch directly, exports DINO to
  another runtime, or ships separate CPU/CUDA packages.
- Remove or demote cosine_topk if the final SQLite/index layer owns search.
- Keep this module free of crop extraction, ref positioning, color rerank,
  UI, GitHub update, and final match decision logic.
"""
