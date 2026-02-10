import os
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np


@dataclass
class Rect:
    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    @property
    def side_min(self) -> int:
        return int(min(self.w, self.h))

    def as_dict(self) -> Dict[str, int]:
        return {"x": int(self.x), "y": int(self.y), "w": int(self.w), "h": int(self.h)}


class HoyolabParser:
    """
    HoYoLAB icons-first парсер (без карточек/костылей; два независимых стека).

    ВАЖНО: ref_width используется только для downscale (никогда не апскейлим),
    чтобы не плодить артефакты на маленьких скринах.
    """

    def __init__(
        self,
        image_path: str,
        out_dir: str = "assets",
        debug_dir: str = "debug",
        ref_width: int = 1500,
        debug: bool = True,
    ):
        self.image_path = image_path
        self.out_dir = out_dir
        self.debug_dir = debug_dir
        self.ref_width = int(ref_width)
        self.debug = bool(debug)

        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "characters"), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "weapons"), exist_ok=True)
        if self.debug:
            os.makedirs(self.debug_dir, exist_ok=True)

        with open(image_path, "rb") as f:
            data = f.read()
        self.image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if self.image is None:
            raise ValueError(f"Не удалось открыть изображение: {image_path}")

        self._mask_for_cnt_small: Optional[np.ndarray] = None
        self._scale_x: Optional[float] = None
        self._scale_y: Optional[float] = None
        self._dbg_params: Dict[str, Any] = {}

    # ---------------------------
    # Public API
    # ---------------------------
    def parse(self) -> Dict[str, Any]:
        img = self.image
        big_squares, small_squares = self.find_icon_squares(img)

        # порядок не важен, но сортировка помогает отладке
        big_squares = self.sort_rects_reading_order(big_squares)
        small_squares = self.sort_rects_reading_order(small_squares)

        chars_out: List[Dict[str, Any]] = []
        weaps_out: List[Dict[str, Any]] = []

        for i, r in enumerate(big_squares):
            crop = self.crop_rect(img, r, pad=0.02)
            cv2.imwrite(os.path.join(self.out_dir, "characters", f"char_{i:03d}.png"), crop)
            chars_out.append({"index": i, "rect": r.as_dict()})

        for j, r in enumerate(small_squares):
            crop = self.crop_rect(img, r, pad=0.02)
            cv2.imwrite(os.path.join(self.out_dir, "weapons", f"weapon_{j:03d}.png"), crop)
            weaps_out.append({"index": j, "rect": r.as_dict()})

        if self.debug:
            self._dbg_save_overlay(img, big_squares, "char_squares_overlay_original.png", color=(255, 0, 0))  # blue
            self._dbg_save_overlay(img, small_squares, "weapon_squares_overlay_original.png", color=(0, 0, 255))  # red
            self._dbg_overlay_two_sets(img, big_squares, small_squares, name="pairs_overlay_original.png")
            self.write_debug_summary(big_squares, small_squares, chars_out, weaps_out)

        return {"characters": chars_out, "weapons": weaps_out}

    # ---------------------------
    # Detect icon squares (big+small)
    # ---------------------------
    def find_icon_squares(self, img_bgr: np.ndarray) -> Tuple[List[Rect], List[Rect]]:
        h0, w0 = img_bgr.shape[:2]

        img_small, sx, sy = self._resize_to_ref_width(img_bgr, self.ref_width)  # downscale-only
        hs, ws = img_small.shape[:2]

        hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
        _, S, V = cv2.split(hsv)

        s_p60 = int(np.percentile(S, 60))
        s_thr = min(90, max(35, s_p60))

        v_p55 = int(np.percentile(V, 55))
        v_thr = min(140, max(70, v_p55))

        ui_mask = ((S < s_thr) & (V < v_thr)).astype(np.uint8) * 255
        icons_mask = 255 - ui_mask

        # Убрать тонкое (текст/шум), оставить толстые квадраты
        k_text = max(5, int(ws * 0.005) | 1)
        kernel_text = cv2.getStructuringElement(cv2.MORPH_RECT, (k_text, k_text))
        mask_no_text = cv2.morphologyEx(icons_mask, cv2.MORPH_OPEN, kernel_text, iterations=1)

        # Подштопать квадраты
        k_patch = max(3, int(ws * 0.003) | 1)
        kernel_patch = cv2.getStructuringElement(cv2.MORPH_RECT, (k_patch, k_patch))
        mask_for_cnt = cv2.morphologyEx(mask_no_text, cv2.MORPH_CLOSE, kernel_patch, iterations=1)

        _, mask_for_cnt = cv2.threshold(mask_for_cnt, 127, 255, cv2.THRESH_BINARY)

        # --- BREAK BRIDGES: режем тонкие горизонтальные перемычки между big и small ---
        k_bridge = max(3, int(ws * 0.010) | 1)  # для ws=724 это ~7..9
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_bridge))  # вертикальная палка
        mask_for_cnt = cv2.morphologyEx(mask_for_cnt, cv2.MORPH_OPEN, kernel_v, iterations=1)

        # (опционально) чуть подштопать края после разрыва
        mask_for_cnt = cv2.morphologyEx(mask_for_cnt, cv2.MORPH_CLOSE, kernel_patch, iterations=1)

        self._mask_for_cnt_small = mask_for_cnt
        self._scale_x = sx
        self._scale_y = sy
        self._dbg_params = {
            "s_thr": s_thr,
            "v_thr": v_thr,
            "k_text": k_text,
            "k_patch": k_patch,
            "scale_x": float(sx),
            "scale_y": float(sy),
            "small_shape": [int(hs), int(ws)],
            "note": "downscale-only (no upscaling)",
        }

        if self.debug:
            cv2.imwrite(os.path.join(self.debug_dir, "04_icons_mask_inverted.png"), icons_mask)
            cv2.imwrite(os.path.join(self.debug_dir, "06_icons_mask_no_text.png"), mask_for_cnt)
            vis = img_small.copy()
            cv2.putText(
                vis,
                f"s_thr={s_thr} v_thr={v_thr} k_text={k_text} k_patch={k_patch} sx={sx:.6f} sy={sy:.6f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imwrite(os.path.join(self.debug_dir, "00_small_with_thresholds.png"), vis)

        # --- connected components ---
        binm = (mask_for_cnt > 0).astype(np.uint8)  # 0/1
        num, labels, stats, _ = cv2.connectedComponentsWithStats(binm, connectivity=8)

        img_area = float(ws * hs)
        candidates: List[Rect] = []

        for i in range(1, num):
            x, y, w, h, area = stats[i].tolist()
            if w < 3 or h < 3:
                continue

            roi = binm[y:y + h, x:x + w]
            split_in_roi = self._split_merged_component_x(roi)

            parts = [Rect(x + r.x, y + r.y, r.w, r.h) for r in split_in_roi] if split_in_roi else [Rect(x, y, w, h)]

            for r in parts:
                ww, hh = r.w, r.h
                if ww < 3 or hh < 3:
                    continue

                # === IMPORTANT: ужимаем bbox до реальных белых пикселей ===
                rt = self._tighten_rect_to_mask(binm, r)
                if rt is None:
                    continue
                r = rt
                area_bbox = float(ww * hh)

                # слишком мелкое
                if area_bbox < img_area * 0.00002:
                    continue
                # слишком крупное
                if area_bbox > img_area * 0.25:
                    continue

                aspect = ww / float(hh)
                if not (0.70 <= aspect <= 1.45):
                    continue

                sub = binm[r.y:r.y + r.h, r.x:r.x + r.w]
                white = float(sub.sum())  # 0/1
                density = white / max(1.0, area_bbox)

                # было 0.12; для оружия часто нужно мягче
                if density < 0.08:
                    continue
                tight = self._tighten_rect_to_mask(binm, r)  # binm у тебя 0/1
                if tight is None:
                    continue
                candidates.append(r)

        if not candidates:
            if self.debug:
                self._dbg_save_overlay(img_small, [], "char_squares_overlay_small_none.png", color=(0, 255, 0))
            return [], []

        # --- split big/small по min(w,h) ДО squareize ---
        sizes = np.array([c.side_min for c in candidates], dtype=np.float32)

        # 1) выкидываем явные outlier'ы снизу, чтобы largest_gap не резался по мусору
        p10 = float(np.percentile(sizes, 10))
        floor = max(10.0, p10 * 0.70)  # у тебя p10=61 => floor~42, мусор 11 уйдёт
        keep_idx = [i for i, s in enumerate(sizes) if float(s) >= floor]

        candidates_kept = [candidates[i] for i in keep_idx]
        sizes_kept = [int(round(sizes[i])) for i in keep_idx]

        # дебаг в summary
        self._dbg_params["sizes_floor"] = {"p10": p10, "floor": floor, "kept": len(candidates_kept),
                                           "total": len(candidates)}

        big_raw, small_raw, split_dbg = self._split_big_small_by_sizes(sizes_kept, candidates_kept)

        self._dbg_params["split"] = split_dbg
        self._dbg_params["sizes_stats"] = {
            "count": int(len(sizes)),
            "min": int(np.min(sizes)),
            "p10": int(np.percentile(sizes, 10)),
            "p50": int(np.percentile(sizes, 50)),
            "p90": int(np.percentile(sizes, 90)),
            "max": int(np.max(sizes)),
        }

        # squareize отдельно
        big_sq = [self._squareize(r) for r in big_raw]
        small_sq = [self._squareize(r) for r in small_raw]

        if self.debug:
            self._dbg_save_overlay(img_small, big_sq, "char_squares_overlay_small.png", color=(0, 255, 0))
            self._dbg_save_overlay(img_small, small_sq, "weapon_squares_overlay_small.png", color=(0, 0, 255))

        big_orig = self._to_original_coords(big_sq, w0, h0, sx, sy)
        small_orig = self._to_original_coords(small_sq, w0, h0, sx, sy)

        return big_orig, small_orig

    # ---------------------------
    # Mizuki fix: split merged component by X-projection
    # ---------------------------
    def _split_merged_component_x(self, bin_roi: np.ndarray) -> Optional[List[Rect]]:
        h, w = bin_roi.shape[:2]
        if h < 12 or w < 12:
            return None
        if w < int(h * 1.15):
            return None

        col = bin_roi.sum(axis=0).astype(np.int32)  # 0..h
        sig = col > int(h * 0.18)

        runs = []
        i = 0
        while i < w:
            if not sig[i]:
                i += 1
                continue
            j = i
            while j < w and sig[j]:
                j += 1
            runs.append((i, j))
            i = j

        if len(runs) < 2:
            return None

        min_run = max(8, int(h * 0.22))
        runs = [(a, b) for (a, b) in runs if (b - a) >= min_run]
        if len(runs) < 2:
            return None

        (a1, b1), (a2, b2) = runs[0], runs[1]
        gap = a2 - b1
        if gap < max(2, int(h * 0.03)):
            return None

        split_x = (b1 + a2) // 2

        def tight_bbox(sub: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
            ys, xs = np.where(sub > 0)
            if xs.size == 0:
                return None
            x1, x2 = int(xs.min()), int(xs.max()) + 1
            y1, y2 = int(ys.min()), int(ys.max()) + 1
            return x1, y1, x2, y2

        left = bin_roi[:, :split_x]
        right = bin_roi[:, split_x:]

        bb1 = tight_bbox(left)
        bb2 = tight_bbox(right)
        if bb1 is None or bb2 is None:
            return None

        x1, y1, x2, y2 = bb1
        r1 = Rect(x=x1, y=y1, w=x2 - x1, h=y2 - y1)

        x1, y1, x2, y2 = bb2
        r2 = Rect(x=split_x + x1, y=y1, w=x2 - x1, h=y2 - y1)

        if r1.w < 8 or r1.h < 8 or r2.w < 8 or r2.h < 8:
            return None

        return [r1, r2]

    # ---------------------------
    # Split big/small robustly (sizes BEFORE squareize)
    # ---------------------------
    def _split_big_small_by_sizes(
        self,
        sizes: List[int],
        rects: List[Rect],
    ) -> Tuple[List[Rect], List[Rect], Dict[str, Any]]:
        s = np.array([float(x) for x in sizes], dtype=np.float32)
        if len(s) < 2:
            return rects, [], {"method": "degenerate", "thr": None, "count": int(len(s))}

        s_sorted = np.sort(s)
        ratios = s_sorted[1:] / np.maximum(1.0, s_sorted[:-1])
        k = int(np.argmax(ratios))
        best_ratio = float(ratios[k])

        # 1) если разрыв хороший — largest gap, но не даём ему резаться по мусору
        use_largest_gap = best_ratio >= 1.25
        if use_largest_gap:
            thr_gap = float((s_sorted[k] + s_sorted[k + 1]) / 2.0)
            p10 = float(np.percentile(s, 10))
            if thr_gap < p10 * 0.9:
                use_largest_gap = False  # это был разрыв "мусор -> всё остальное"

        if use_largest_gap:
            thr = thr_gap
            big = [r for r, ss in zip(rects, s) if float(ss) >= thr]
            small = [r for r, ss in zip(rects, s) if float(ss) < thr]
            dbg = {"method": "largest_gap", "thr": thr, "best_ratio": best_ratio, "guard_p10": p10}
        else:
            # 2) fallback: k-means 1D (k=2), small=меньший центр
            c1 = float(np.percentile(s, 25))
            c2 = float(np.percentile(s, 85))
            for _ in range(20):
                d1 = np.abs(s - c1)
                d2 = np.abs(s - c2)
                lab = (d2 < d1).astype(np.int32)
                c1n = float(np.mean(s[lab == 0])) if np.any(lab == 0) else c1
                c2n = float(np.mean(s[lab == 1])) if np.any(lab == 1) else c2
                if abs(c1n - c1) < 0.01 and abs(c2n - c2) < 0.01:
                    c1, c2 = c1n, c2n
                    break
                c1, c2 = c1n, c2n

            small_center = min(c1, c2)
            big_center = max(c1, c2)
            thr = float((small_center + big_center) / 2.0)
            big = [r for r, ss in zip(rects, s) if float(ss) >= thr]
            small = [r for r, ss in zip(rects, s) if float(ss) < thr]
            dbg = {"method": "kmeans2_1d", "thr": thr, "centers": [small_center, big_center], "best_ratio": best_ratio}

        if not small or not big:
            thr2 = float(np.median(s))
            big = [r for r, ss in zip(rects, s) if float(ss) >= thr2]
            small = [r for r, ss in zip(rects, s) if float(ss) < thr2]
            dbg["fallback"] = {"method": "median", "thr": thr2}

        # safety: ensure big really bigger
        if big and small:
            mb = float(np.median([r.side_min for r in big]))
            ms = float(np.median([r.side_min for r in small]))
            if ms > mb:
                big, small = small, big
                dbg["swapped"] = True

        return big, small, dbg

    # ---------------------------
    # Helpers: resize / coords / geometry
    # ---------------------------
    def _resize_to_ref_width(self, img_bgr: np.ndarray, ref_width: int) -> Tuple[np.ndarray, float, float]:
        """
        Downscale-only: если исходник уже меньше ref_width, НЕ увеличиваем.
        Это важно для маленьких скринов: апскейл плодит артефакты и ломает split big/small.
        """
        h0, w0 = img_bgr.shape[:2]
        ref_width = int(ref_width)

        if ref_width <= 0:
            return img_bgr.copy(), 1.0, 1.0

        if w0 <= ref_width:
            # no upscale
            return img_bgr.copy(), 1.0, 1.0

        target_w = int(ref_width)
        target_h = int(round(h0 * (target_w / float(w0))))

        interp = cv2.INTER_AREA  # downscale
        img_small = cv2.resize(img_bgr, (target_w, target_h), interpolation=interp)

        sx = target_w / float(w0)
        sy = target_h / float(h0)
        return img_small, float(sx), float(sy)

    def _to_original_coords(self, rects_small: List[Rect], w0: int, h0: int, sx: float, sy: float) -> List[Rect]:
        out: List[Rect] = []
        for r in rects_small:
            x = int(round(r.x / sx))
            y = int(round(r.y / sy))
            w = int(round(r.w / sx))
            h = int(round(r.h / sy))
            out.append(Rect(x, y, w, h))
        return self.clip_rects(out, w0, h0)

    def _tighten_rect_to_mask(self, binm01: np.ndarray, r: Rect) -> Optional[Rect]:
        """
        Ужать Rect до реальных белых пикселей (binm01 = 0/1) внутри него.
        Возвращает новый Rect или None (если пикселей нет).
        """
        h, w = binm01.shape[:2]
        x1 = max(0, min(w - 1, r.x))
        y1 = max(0, min(h - 1, r.y))
        x2 = max(x1 + 1, min(w, r.x2))
        y2 = max(y1 + 1, min(h, r.y2))

        roi = binm01[y1:y2, x1:x2]
        ys, xs = np.where(roi > 0)
        if xs.size == 0:
            return None

        xx1 = int(xs.min()) + x1
        xx2 = int(xs.max()) + 1 + x1
        yy1 = int(ys.min()) + y1
        yy2 = int(ys.max()) + 1 + y1
        return Rect(xx1, yy1, xx2 - xx1, yy2 - yy1)


    def _squareize(self, r: Rect) -> Rect:
        side = int(round(max(r.w, r.h)))
        cx = r.x + r.w / 2.0
        cy = r.y + r.h / 2.0
        x2 = int(round(cx - side / 2.0))
        y2 = int(round(cy - side / 2.0))
        return Rect(x2, y2, side, side)

    # ---------------------------
    # Crops / clip
    # ---------------------------
    def crop_rect(self, img_bgr: np.ndarray, r: Rect, pad: float = 0.0) -> np.ndarray:
        h, w = img_bgr.shape[:2]
        p = int(round(min(r.w, r.h) * pad))

        x1 = max(0, min(w - 1, r.x + p))
        y1 = max(0, min(h - 1, r.y + p))
        x2 = max(x1 + 1, min(w, r.x2 - p))
        y2 = max(y1 + 1, min(h, r.y2 - p))

        return img_bgr[y1:y2, x1:x2].copy()

    def clip_rects(self, rects: List[Rect], w: int, h: int) -> List[Rect]:
        out = []
        for r in rects:
            x1 = max(0, min(w - 1, r.x))
            y1 = max(0, min(h - 1, r.y))
            x2 = max(0, min(w, r.x2))
            y2 = max(0, min(h, r.y2))
            ww = max(1, x2 - x1)
            hh = max(1, y2 - y1)
            out.append(Rect(x1, y1, ww, hh))
        return out

    # ---------------------------
    # Sorting
    # ---------------------------
    def sort_rects_reading_order(self, rects: List[Rect]) -> List[Rect]:
        if not rects:
            return []

        rects = rects[:]
        rects.sort(key=lambda r: r.cy)

        h_med = float(np.median([r.h for r in rects]))
        row_tol = h_med * 0.45

        rows: List[List[Rect]] = []
        for r in rects:
            placed = False
            for row in rows:
                row_cy = float(np.mean([x.cy for x in row]))
                if abs(r.cy - row_cy) <= row_tol:
                    row.append(r)
                    placed = True
                    break
            if not placed:
                rows.append([r])

        for row in rows:
            row.sort(key=lambda r: r.cx)

        rows.sort(key=lambda row: float(np.mean([r.cy for r in row])))
        return [r for row in rows for r in row]

    # ---------------------------
    # Debug
    # ---------------------------
    def _dbg_save_overlay(self, img_bgr: np.ndarray, rects: List[Rect], name: str, color=(0, 255, 0)):
        if not self.debug:
            return
        vis = img_bgr.copy()
        for i, r in enumerate(rects):
            cv2.rectangle(vis, (r.x, r.y), (r.x2, r.y2), color, 2)
            cv2.putText(
                vis,
                str(i),
                (r.x + 4, r.y + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )
        cv2.imwrite(os.path.join(self.debug_dir, name), vis)

    def _dbg_overlay_two_sets(self, img_bgr: np.ndarray, big: List[Rect], small: List[Rect], name: str):
        if not self.debug:
            return
        vis = img_bgr.copy()
        for r in big:
            cv2.rectangle(vis, (r.x, r.y), (r.x2, r.y2), (255, 0, 0), 2)  # blue
        for r in small:
            cv2.rectangle(vis, (r.x, r.y), (r.x2, r.y2), (0, 0, 255), 2)  # red
        cv2.imwrite(os.path.join(self.debug_dir, name), vis)

    def write_debug_summary(
        self,
        big_squares: List[Rect],
        small_squares: List[Rect],
        chars: List[Dict[str, Any]],
        weapons: List[Dict[str, Any]],
    ):
        if not self.debug:
            return

        summary = {
            "image_path": self.image_path,
            "image_shape": list(self.image.shape[:2]),
            "ref_width": self.ref_width,
            "big_squares_found": len(big_squares),
            "small_squares_found": len(small_squares),
            "characters_saved": len(chars),
            "weapons_saved": len(weapons),
            "mask_params": self._dbg_params,
            "notes": [
                "00_small_with_thresholds.png: пороги сегментации и параметры морфологии + scale_x/scale_y",
                "04_icons_mask_inverted.png: белое = островки контента (иконки/текст) после инверта ui_mask",
                "06_icons_mask_no_text.png: после морфологии (OPEN->CLOSE)",
                "char_squares_overlay_small.png: big-квадраты на уменьшенной картинке",
                "weapon_squares_overlay_small.png: small-квадраты на уменьшенной картинке",
                "char_squares_overlay_original.png: big-квадраты в оригинальных координатах",
                "weapon_squares_overlay_original.png: small-квадраты в оригинальных координатах",
                "pairs_overlay_original.png: синий=персы, красный=оружие (независимые стеки)",
            ],
        }
        with open(os.path.join(self.debug_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)