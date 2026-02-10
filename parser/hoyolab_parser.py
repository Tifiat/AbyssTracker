import os
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

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


class HoyolabParser:
    """
    HoYoLAB icons-first парсер.

    Pipeline:
    1) Находим квадраты персонажей (портреты) через сегментацию UI + морфологию + контуры.
    2) Восстанавливаем "виртуальную" карточку по долям, используя квадрат персонажа как якорь.
    3) Автокалибруем dx оружия (на каждом скрине отдельно) по маске через IoU.
    4) Режем иконки персонажей и оружия.
    5) Пишем дебаг-артефакты (маски, оверлеи, json калибровки).
    """

    # ---- Нормализованные доли внутри карточки (из 332x167) ----
    # Персонаж (квадрат слева)
    CHAR_X = 0.0271
    CHAR_Y = 0.0659
    CHAR_W = 0.4458
    CHAR_H = 0.8862

    # Оружие (квадрат справа снизу)
    WEAP_X = 0.5090
    WEAP_Y = 0.5689
    WEAP_W = 0.1898
    WEAP_H = 0.3772

    # Эталон карточки
    CARD_W_REF = 332.0
    CARD_H_REF = 167.0
    CARD_ASPECT = CARD_H_REF / CARD_W_REF  # ~0.503

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
        self.ref_width = ref_width
        self.debug = debug

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

        # заполняются внутри find_character_squares()
        self._mask_for_cnt_small: Optional[np.ndarray] = None
        self._mask_scale: Optional[float] = None

    # ---------------------------
    # Public API
    # ---------------------------
    def parse(self) -> List[Dict[str, Any]]:
        img = self.image

        char_squares = self.find_character_squares(img)
        char_squares = self.sort_rects_reading_order(char_squares)

        dx_auto = self.calibrate_weapon_dx(char_squares)

        results: List[Dict[str, Any]] = []

        for i, char_sq in enumerate(char_squares):
            card = self.estimate_card_from_char_square(char_sq)

            char_icon = self.crop_rect(img, char_sq, pad=0.02)

            weapon_rect = self.rect_from_rel(
                card,
                self.WEAP_X + dx_auto,
                self.WEAP_Y,
                self.WEAP_W,
                self.WEAP_H,
            )
            weap_icon = self.crop_rect(img, weapon_rect, pad=0.02)

            cv2.imwrite(os.path.join(self.out_dir, "characters", f"char_{i:03d}.png"), char_icon)
            cv2.imwrite(os.path.join(self.out_dir, "weapons", f"weapon_{i:03d}.png"), weap_icon)

            results.append({
                "index": i,
                "char_square": {"x": char_sq.x, "y": char_sq.y, "w": char_sq.w, "h": char_sq.h},
                "estimated_card": {"x": card.x, "y": card.y, "w": card.w, "h": card.h},
                "weapon_rect": {"x": weapon_rect.x, "y": weapon_rect.y, "w": weapon_rect.w, "h": weapon_rect.h},
            })

        if self.debug:
            self._dbg_save_overlay(img, char_squares, "char_squares_overlay_original.png")
            self._dbg_overlay_pairs(img, results, name="pairs_overlay_original.png")
            self.write_debug_summary(char_squares, results)

        return results

    # ---------------------------
    # Detect character squares
    # ---------------------------
    def find_character_squares(self, img_bgr: np.ndarray) -> List[Rect]:
        h0, w0 = img_bgr.shape[:2]

        # normalize scale for stable morphology thresholds
        scale = self.ref_width / float(w0) if w0 > self.ref_width else 1.0
        img_small = (
            cv2.resize(img_bgr, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)
            if scale != 1.0 else img_bgr.copy()
        )
        hs, ws = img_small.shape[:2]

        hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
        _, S, V = cv2.split(hsv)

        # динамические пороги
        s_p60 = int(np.percentile(S, 60))
        s_thr = min(90, max(35, s_p60))

        v_p55 = int(np.percentile(V, 55))
        v_thr = min(140, max(70, v_p55))

        # ui_mask: white = dark+low sat (фон/панели)
        ui_mask = ((S < s_thr) & (V < v_thr)).astype(np.uint8) * 255

        # invert => content islands are white
        icons_mask = 255 - ui_mask

        # 1) удаляем тонкие структуры (текст/шумы), оставляем толстые квадраты
        k_text = max(5, int(ws * 0.005) | 1)  # 0.005..0.009
        kernel_text = cv2.getStructuringElement(cv2.MORPH_RECT, (k_text, k_text))
        mask_no_text = cv2.morphologyEx(icons_mask, cv2.MORPH_OPEN, kernel_text, iterations=1)

        # 2) чуть “подштопать” квадраты
        k_patch = max(3, int(ws * 0.003) | 1)
        kernel_patch = cv2.getStructuringElement(cv2.MORPH_RECT, (k_patch, k_patch))
        mask_for_cnt = cv2.morphologyEx(mask_no_text, cv2.MORPH_CLOSE, kernel_patch, iterations=1)

        # бинарность
        _, mask_for_cnt = cv2.threshold(mask_for_cnt, 127, 255, cv2.THRESH_BINARY)

        # сохраним маску и scale для автокалибровки оружия
        self._mask_for_cnt_small = mask_for_cnt
        self._mask_scale = scale

        if self.debug:
            cv2.imwrite(os.path.join(self.debug_dir, "04_icons_mask_inverted.png"), icons_mask)
            cv2.imwrite(os.path.join(self.debug_dir, "06_icons_mask_no_text.png"), mask_for_cnt)
            vis = img_small.copy()
            cv2.putText(
                vis,
                f"s_thr={s_thr} v_thr={v_thr} k_text={k_text} k_patch={k_patch}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imwrite(os.path.join(self.debug_dir, "00_small_with_thresholds.png"), vis)

        # 3) контуры
        contours, _ = cv2.findContours(mask_for_cnt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rects: List[Rect] = []
        img_area = ws * hs

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 2 or h < 2:
                continue

            area_bbox = w * h
            if area_bbox < img_area * 0.0005:  # мелочь
                continue
            if area_bbox > img_area * 0.08:  # слишком крупное (шапка/фон)
                continue

            aspect = w / float(h)
            if not (0.60 <= aspect <= 1.60):  # “пожёванные” квадраты
                continue

            rects.append(Rect(x, y, w, h))

        if not rects:
            if self.debug:
                self._dbg_save_overlay(img_small, [], "char_squares_overlay_small_none.png")
            return []

        # 4) выбираем кластер "самых больших" (портреты)
        sizes = np.array([min(r.w, r.h) for r in rects], dtype=np.float32)

        rects_sorted = sorted(rects, key=lambda r: min(r.w, r.h), reverse=True)
        s_sorted = np.array([min(r.w, r.h) for r in rects_sorted], dtype=np.float32)

        drops = s_sorted[:-1] / np.maximum(1.0, s_sorted[1:])
        if drops.size > 0 and np.any(drops > 1.25):
            cut = int(np.argmax(drops > 1.25))
            big = rects_sorted[:cut + 1]
        else:
            thr = float(np.percentile(sizes, 30))
            big = [r for r in rects if min(r.w, r.h) >= thr]

        # 5) “доквадрачиваем” bbox: квадрат по центру
        big_sq: List[Rect] = []
        for r in big:
            side = int(round(max(r.w, r.h)))
            cx = r.x + r.w / 2.0
            cy = r.y + r.h / 2.0
            x2 = int(round(cx - side / 2.0))
            y2 = int(round(cy - side / 2.0))
            big_sq.append(Rect(x2, y2, side, side))

        if self.debug:
            self._dbg_save_overlay(img_small, big_sq, "char_squares_overlay_small.png")

        # 6) convert to original coords
        inv = 1.0 / scale
        rects_orig = [
            Rect(
                x=int(round(r.x * inv)),
                y=int(round(r.y * inv)),
                w=int(round(r.w * inv)),
                h=int(round(r.h * inv)),
            )
            for r in big_sq
        ]
        rects_orig = self.clip_rects(rects_orig, w0, h0)

        return rects_orig

    # ---------------------------
    # Geometry / Crops
    # ---------------------------
    def estimate_card_from_char_square(self, char_sq: Rect) -> Rect:
        # card_w от квадрата персонажа
        card_w = float(char_sq.w) / float(self.CHAR_W)
        # card_h строго из аспекта карточки (стабильно!)
        card_h = card_w * float(self.CARD_ASPECT)

        card_x = float(char_sq.x) - self.CHAR_X * card_w
        card_y = float(char_sq.y) - self.CHAR_Y * card_h

        return Rect(
            int(round(card_x)),
            int(round(card_y)),
            int(round(card_w)),
            int(round(card_h)),
        )

    def rect_from_rel(self, card: Rect, rx: float, ry: float, rw: float, rh: float) -> Rect:
        x = int(round(card.x + rx * card.w))
        y = int(round(card.y + ry * card.h))
        w = int(round(rw * card.w))
        h = int(round(rh * card.h))
        return Rect(x, y, w, h)

    def crop_rect(self, img_bgr: np.ndarray, r: Rect, pad: float = 0.0) -> np.ndarray:
        """pad: доля от min(w,h), срезаем внутрь (чтобы не цеплять рамку/антиалиас)."""
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
    # Weapon dx calibration (IoU)
    # ---------------------------
    def _iou(self, a: Rect, b: Rect) -> float:
        x1 = max(a.x, b.x)
        y1 = max(a.y, b.y)
        x2 = min(a.x2, b.x2)
        y2 = min(a.y2, b.y2)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter <= 0:
            return 0.0
        union = a.w * a.h + b.w * b.h - inter
        return float(inter) / float(max(1, union))

    def _find_best_white_bbox_in_window(self, mask_small: np.ndarray, win: Rect) -> Optional[Rect]:
        """В окне win находим bbox самого большого белого объекта (обычно оружие)."""
        h, w = mask_small.shape[:2]
        x1 = max(0, min(w - 1, win.x))
        y1 = max(0, min(h - 1, win.y))
        x2 = max(x1 + 1, min(w, win.x2))
        y2 = max(y1 + 1, min(h, win.y2))

        roi = mask_small[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        best = None
        best_area = -1
        for c in contours:
            xx, yy, ww, hh = cv2.boundingRect(c)
            area = ww * hh
            if area > best_area:
                best_area = area
                best = Rect(x1 + xx, y1 + yy, ww, hh)

        return best

    def calibrate_weapon_dx(self, char_squares: List[Rect], samples: int = 20) -> float:
        """
        Автоподбор dx (в долях ширины карточки) для оружия.
        Находим dx, который даёт лучший IoU между ожидаемым weapon_rect и реальным белым bbox на маске.
        """
        mask_small = self._mask_for_cnt_small
        scale = self._mask_scale
        if mask_small is None or scale is None or not char_squares:
            return 0.0

        use = char_squares[: min(samples, len(char_squares))]

        dx_candidates = np.linspace(-0.04, 0.04, 33)  # шаг ~0.0025
        best_dx = 0.0
        best_score = -1.0
        scores_dump = []

        for dx in dx_candidates:
            ious = []

            for cs in use:
                card = self.estimate_card_from_char_square(cs)
                wr = self.rect_from_rel(card, self.WEAP_X + float(dx), self.WEAP_Y, self.WEAP_W, self.WEAP_H)

                # weapon rect -> small coords
                wr_s = Rect(
                    x=int(round(wr.x * scale)),
                    y=int(round(wr.y * scale)),
                    w=int(round(wr.w * scale)),
                    h=int(round(wr.h * scale)),
                )

                # окно поиска вокруг ожидаемого оружия
                pad_x = int(round(wr_s.w * 0.35))
                pad_y = int(round(wr_s.h * 0.35))
                win = Rect(wr_s.x - pad_x, wr_s.y - pad_y, wr_s.w + 2 * pad_x, wr_s.h + 2 * pad_y)

                true_bbox = self._find_best_white_bbox_in_window(mask_small, win)
                if true_bbox is None:
                    continue

                ious.append(self._iou(wr_s, true_bbox))

            avg = float(np.mean(ious)) if ious else 0.0
            scores_dump.append({"dx": float(dx), "score": avg})

            if avg > best_score:
                best_score = avg
                best_dx = float(dx)

        if self.debug:
            with open(os.path.join(self.debug_dir, "weapon_dx_calibration.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {"best_dx": best_dx, "best_score": best_score, "scores": scores_dump},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

        # если улучшение микроскопическое — dx не нужен
        score_at_0 = next((x["score"] for x in scores_dump if abs(x["dx"]) < 1e-9), None)
        if score_at_0 is not None and (best_score - float(score_at_0)) < 0.01:
            return 0.0

        return best_dx

    # ---------------------------
    # Sorting
    # ---------------------------
    def sort_rects_reading_order(self, rects: List[Rect]) -> List[Rect]:
        """Сортировка в порядке чтения: сверху вниз, слева направо (устойчиво по строкам)."""
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
                if abs(r.cy - float(np.mean([x.cy for x in row]))) <= row_tol:
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
    def _dbg_save_overlay(self, img_bgr: np.ndarray, rects: List[Rect], name: str):
        if not self.debug:
            return
        vis = img_bgr.copy()
        for i, r in enumerate(rects):
            cv2.rectangle(vis, (r.x, r.y), (r.x2, r.y2), (0, 255, 0), 2)
            cv2.putText(
                vis,
                str(i),
                (r.x + 4, r.y + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        cv2.imwrite(os.path.join(self.debug_dir, name), vis)

    def _dbg_overlay_pairs(self, img_bgr: np.ndarray, results: List[Dict[str, Any]], name: str):
        if not self.debug:
            return
        vis = img_bgr.copy()
        for item in results:
            cs = item["char_square"]
            cr = Rect(cs["x"], cs["y"], cs["w"], cs["h"])
            cv2.rectangle(vis, (cr.x, cr.y), (cr.x2, cr.y2), (255, 0, 0), 2)  # char (blue)

            card = item["estimated_card"]
            rr = Rect(card["x"], card["y"], card["w"], card["h"])
            cv2.rectangle(vis, (rr.x, rr.y), (rr.x2, rr.y2), (0, 255, 255), 2)  # card (yellow)

            wr = item["weapon_rect"]
            ww = Rect(wr["x"], wr["y"], wr["w"], wr["h"])
            cv2.rectangle(vis, (ww.x, ww.y), (ww.x2, ww.y2), (0, 0, 255), 2)  # weapon (red)

        cv2.imwrite(os.path.join(self.debug_dir, name), vis)

    def write_debug_summary(self, char_squares: List[Rect], results: List[Dict[str, Any]]):
        if not self.debug:
            return
        summary = {
            "image_path": self.image_path,
            "image_shape": list(self.image.shape[:2]),
            "ref_width": self.ref_width,
            "char_squares_found": len(char_squares),
            "results_count": len(results),
            "notes": [
                "00_small_with_thresholds.png: пороги сегментации и параметры морфологии",
                "04_icons_mask_inverted.png: белое = островки контента (иконки/текст) после инверта ui_mask",
                "06_icons_mask_no_text.png: после морфологии (текст/шумы убраны, квадраты стабильнее)",
                "char_squares_overlay_small.png: квадраты персонажей на уменьшенной картинке (debug)",
                "char_squares_overlay_original.png: квадраты персонажей в оригинальных координатах",
                "weapon_dx_calibration.json: подбор dx для оружия по IoU",
                "pairs_overlay_original.png: синий=перс, жёлтый=карточка(оценка), красный=оружие",
            ],
        }
        with open(os.path.join(self.debug_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)