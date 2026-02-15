import os
import json
import shutil
import cv2
import numpy as np
from urllib.request import urlopen, Request

ENKA_UI = "https://enka.network/ui"

# Абсолютный корень проекта: .../AbyssTracker
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _p(*parts: str) -> str:
    return os.path.join(ROOT, *parts)


def _load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _download_png(url: str, timeout: float = 20.0) -> bytes:
    req = Request(url, headers={"User-Agent": "AbyssTracker/1.0"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _copy_from_cache_if_exists(cache_dir: str, _id: str, dst_path: str) -> bool:
    src = os.path.join(cache_dir, f"{_id}.png")
    if os.path.exists(src):
        try:
            shutil.copyfile(src, dst_path)
            return True
        except Exception:
            return False
    return False


def _composite_alpha_to_gray_bg(img_rgba: np.ndarray, bg_value: int = 180) -> np.ndarray:
    """
    RGBA -> BGR на сером фоне.
    """
    b, g, r, a = cv2.split(img_rgba)
    alpha = a.astype(np.float32) / 255.0
    bg = np.full_like(b, bg_value, dtype=np.uint8)

    def comp(ch):
        return (ch.astype(np.float32) * alpha + bg.astype(np.float32) * (1 - alpha)).astype(np.uint8)

    return cv2.merge([comp(b), comp(g), comp(r)])


# -----------------------------
# Preprocessing: Characters
# -----------------------------
def _read_char_crop_for_orb(img_bgr: np.ndarray) -> np.ndarray:
    """
    Предобработка кропа персонажа (HoYoLAB):
    - центр-кроп
    - замазываем верхний левый угол (элемент)
    - grayscale
    """
    h, w = img_bgr.shape[:2]

    p = int(min(h, w) * 0.07)  # мягко
    if p > 0 and h - 2 * p >= 32 and w - 2 * p >= 32:
        img_bgr = img_bgr[p:h - p, p:w - p]

    h, w = img_bgr.shape[:2]
    corner = int(min(h, w) * 0.22)
    if corner >= 8:
        img_bgr[0:corner, 0:corner] = 0

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray


def _read_char_ref_for_orb(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray


# -----------------------------
# Preprocessing: Weapons
# -----------------------------
def _read_weapon_for_orb(img_bgr: np.ndarray) -> np.ndarray:
    """
    Оружие слишком маленькое и "плоское", поэтому:
    - upscale
    - grayscale
    - edges (Canny) по силуэту
    """

    h, w = img_bgr.shape[:2]
    p = int(min(h, w) * 0.18)  # было 0, делаем сильнее
    if p > 0 and h - 2 * p >= 16 and w - 2 * p >= 16:
        img_bgr = img_bgr[p:h - p, p:w - p]

    img = cv2.resize(img_bgr, (256, 256), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 40, 120)
    return edges


# -----------------------------
# ORB Index
# -----------------------------
class OrbIndex:
    """
    Кэширует (Enka PNG -> ORB descriptors) по id.
    Скачанные PNG складывает в cache_dir/<id>.png.
    allow_download=False => не делает сетевых запросов, только локальный кэш.
    """

    def __init__(self, items: dict, cache_dir: str, allow_download: bool):
        self.items = items
        self.cache_dir = cache_dir
        _ensure_dir(self.cache_dir)
        self.allow_download = bool(allow_download)

        # ORB для персонажей (нормально работает на портретах)
        self.orb = cv2.ORB_create(
            nfeatures=1200,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=15,
            fastThreshold=10,
        )

        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self._des_cache: dict[str, tuple] = {}  # id -> (kp, des)

    def _load_ref_image_bgr(self, _id: str) -> np.ndarray | None:
        meta = self.items.get(str(_id))
        if not meta:
            return None

        icon_name = meta.get("icon_name")
        if not icon_name:
            return None

        local_path = os.path.join(self.cache_dir, f"{_id}.png")

        # 1) локальный кэш
        if os.path.exists(local_path):
            img = cv2.imread(local_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                return None
            if len(img.shape) == 3 and img.shape[2] == 4:
                img = _composite_alpha_to_gray_bg(img)
            return img

        # 2) offline — не скачиваем
        if not self.allow_download:
            return None

        # 3) скачиваем
        url = f"{ENKA_UI}/{icon_name}.png"
        try:
            png = _download_png(url)
        except Exception:
            return None

        try:
            with open(local_path, "wb") as f:
                f.write(png)
        except Exception:
            pass

        img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if len(img.shape) == 3 and img.shape[2] == 4:
            img = _composite_alpha_to_gray_bg(img)
        return img

    def _ref_to_orb_input(self, ref_bgr: np.ndarray) -> np.ndarray:
        # по умолчанию — персонажи
        return _read_char_ref_for_orb(ref_bgr)

    def get_descriptors(self, _id: str):
        if _id in self._des_cache:
            return self._des_cache[_id]

        img_bgr = self._load_ref_image_bgr(_id)
        if img_bgr is None:
            self._des_cache[_id] = (None, None)
            return None, None

        inp = self._ref_to_orb_input(img_bgr)
        kp, des = self.orb.detectAndCompute(inp, None)
        self._des_cache[_id] = (kp, des)
        return kp, des

    def score_match(self, crop_inp: np.ndarray, ref_id: str) -> int:
        kp1, des1 = self.orb.detectAndCompute(crop_inp, None)
        if des1 is None or len(des1) < 12:
            return 0

        kp2, des2 = self.get_descriptors(ref_id)
        if des2 is None or len(des2) < 12:
            return 0

        matches = self.bf.knnMatch(des1, des2, k=2)

        good = []
        for m_n in matches:
            if len(m_n) != 2:
                continue
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good.append(m)

        # мало совпадений — даже не пытаемся RANSAC
        if len(good) < 10:
            return 0

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        if mask is None:
            return 0

        inliers = int(mask.ravel().sum())
        return inliers


class OrbIndexWeapons(OrbIndex):
    """
    Отдельный индекс для оружия: edges+upscale и другой ORB.
    """

    def __init__(self, items: dict, cache_dir: str, allow_download: bool):
        super().__init__(items=items, cache_dir=cache_dir, allow_download=allow_download)

        self.orb = cv2.ORB_create(
            nfeatures=2500,
            scaleFactor=1.2,
            nlevels=12,
            edgeThreshold=5,
            fastThreshold=5,
            patchSize=31,
        )

    def _ref_to_orb_input(self, ref_bgr: np.ndarray) -> np.ndarray:
        return _read_weapon_for_orb(ref_bgr)


# -----------------------------
# Enrich: Characters
# -----------------------------
def enrich_characters_orb(
    crops_char_dir: str,
    data_dir: str = "data",
    out_hd_dir: str = "assets/hd/characters",
    debug_dir: str = "debug/orb",
    score_threshold: int = 28,
    margin: int = 6,
    allow_download: bool = True,
) -> dict:
    characters = _load_json(_p(data_dir, "characters.json"))

    out_hd_dir = _p(out_hd_dir)
    debug_dir = _p(debug_dir)
    cache_dir = _p("cache", "enka_ref_characters")

    _ensure_dir(out_hd_dir)
    _ensure_dir(debug_dir)
    _ensure_dir(cache_dir)

    accepted_dir = os.path.join(debug_dir, "accepted")
    rejected_dir = os.path.join(debug_dir, "rejected")
    _ensure_dir(accepted_dir)
    _ensure_dir(rejected_dir)

    idx = OrbIndex(items=characters, cache_dir=cache_dir, allow_download=allow_download)

    crops_char_dir = _p(crops_char_dir)
    files = []
    if os.path.exists(crops_char_dir):
        files = [f for f in sorted(os.listdir(crops_char_dir)) if f.lower().endswith(".png")]

    report = {"accepted": [], "rejected": []}
    saved_hd = 0
    ids = list(characters.keys())

    for fname in files:
        crop_path = os.path.join(crops_char_dir, fname)
        crop_bgr = cv2.imread(crop_path, cv2.IMREAD_COLOR)
        if crop_bgr is None:
            continue

        crop_bgr_orig = crop_bgr.copy()
        crop_inp = _read_char_crop_for_orb(crop_bgr)

        best_id = None
        best_score = -1
        second_score = -1

        for _id in ids:
            s = idx.score_match(crop_inp, _id)
            if s > best_score:
                second_score = best_score
                best_score = s
                best_id = _id
            elif s > second_score:
                second_score = s

        accepted = (
            best_id is not None
            and best_score >= score_threshold
            and (best_score - second_score) >= margin
        )

        if not accepted:
            report["rejected"].append({
                "crop": fname,
                "best_id": best_id,
                "best_score": best_score,
                "second_score": second_score
            })
            try:
                dbg_name = f"{os.path.splitext(fname)[0]}__best_{best_id}__s_{best_score}__second_{second_score}.png"
                cv2.imwrite(os.path.join(rejected_dir, dbg_name), crop_bgr_orig)
            except Exception:
                pass
            continue

        report["accepted"].append({
            "crop": fname,
            "id": best_id,
            "best_score": best_score,
            "second_score": second_score
        })

        # debug: кроп + ref
        try:
            base = f"{os.path.splitext(fname)[0]}__id_{best_id}__score_{best_score}"
            cv2.imwrite(os.path.join(accepted_dir, base + ".png"), crop_bgr_orig)
            ref_bgr = idx._load_ref_image_bgr(best_id)
            if ref_bgr is not None:
                cv2.imwrite(os.path.join(accepted_dir, base + "__ref.png"), ref_bgr)
        except Exception:
            pass

        # HD: НЕ качаем заново — берём из cache/<id>.png
        hd_path = os.path.join(out_hd_dir, f"{best_id}.png")
        if os.path.exists(hd_path):
            continue

        if _copy_from_cache_if_exists(cache_dir, best_id, hd_path):
            saved_hd += 1

    try:
        with open(os.path.join(debug_dir, "report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return {"characters_saved_hd": saved_hd, "debug_report": os.path.join(debug_dir, "report.json")}


# -----------------------------
# Enrich: Weapons
# -----------------------------
def enrich_weapons_orb(
    crops_weap_dir: str,
    data_dir: str = "data",
    out_hd_dir: str = "assets/hd/weapons",
    debug_dir: str = "debug/orb_weapons",
    score_threshold: int = 10,
    margin: int = 3,
    allow_download: bool = True,
) -> dict:
    weapons = _load_json(_p(data_dir, "weapons.json"))

    out_hd_dir = _p(out_hd_dir)
    debug_dir = _p(debug_dir)
    cache_dir = _p("cache", "enka_ref_weapons")

    _ensure_dir(out_hd_dir)
    _ensure_dir(debug_dir)
    _ensure_dir(cache_dir)

    accepted_dir = os.path.join(debug_dir, "accepted")
    rejected_dir = os.path.join(debug_dir, "rejected")
    _ensure_dir(accepted_dir)
    _ensure_dir(rejected_dir)

    idx = OrbIndexWeapons(items=weapons, cache_dir=cache_dir, allow_download=allow_download)

    crops_weap_dir = _p(crops_weap_dir)
    files = []
    if os.path.exists(crops_weap_dir):
        files = [f for f in sorted(os.listdir(crops_weap_dir)) if f.lower().endswith(".png")]

    report = {"accepted": [], "rejected": []}
    saved_hd = 0
    ids = list(weapons.keys())

    for fname in files:
        crop_path = os.path.join(crops_weap_dir, fname)
        crop_bgr = cv2.imread(crop_path, cv2.IMREAD_COLOR)
        if crop_bgr is None:
            continue

        crop_bgr_orig = crop_bgr.copy()
        crop_inp = _read_weapon_for_orb(crop_bgr)

        best_id = None
        best_score = -1
        second_score = -1

        for _id in ids:
            s = idx.score_match(crop_inp, _id)
            if s > best_score:
                second_score = best_score
                best_score = s
                best_id = _id
            elif s > second_score:
                second_score = s

        accepted = (
            best_id is not None
            and best_score >= score_threshold
            and (best_score - second_score) >= margin
        )

        if not accepted:
            report["rejected"].append({
                "crop": fname,
                "best_id": best_id,
                "best_score": best_score,
                "second_score": second_score
            })
            try:
                dbg_name = f"{os.path.splitext(fname)[0]}__best_{best_id}__s_{best_score}__second_{second_score}.png"
                cv2.imwrite(os.path.join(rejected_dir, dbg_name), crop_bgr_orig)
            except Exception:
                pass
            continue

        report["accepted"].append({
            "crop": fname,
            "id": best_id,
            "best_score": best_score,
            "second_score": second_score
        })

        # debug: кроп + ref
        try:
            base = f"{os.path.splitext(fname)[0]}__id_{best_id}__score_{best_score}"
            cv2.imwrite(os.path.join(accepted_dir, base + ".png"), crop_bgr_orig)
            ref_bgr = idx._load_ref_image_bgr(best_id)
            if ref_bgr is not None:
                cv2.imwrite(os.path.join(accepted_dir, base + "__ref.png"), ref_bgr)
        except Exception:
            pass

        # HD: НЕ качаем заново — берём из cache/<id>.png
        hd_path = os.path.join(out_hd_dir, f"{best_id}.png")
        if os.path.exists(hd_path):
            continue

        if _copy_from_cache_if_exists(cache_dir, best_id, hd_path):
            saved_hd += 1

    try:
        with open(os.path.join(debug_dir, "report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return {"weapons_saved_hd": saved_hd, "debug_report": os.path.join(debug_dir, "report.json")}

