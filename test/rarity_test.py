from __future__ import annotations

from pathlib import Path

import cv2

import crop_extractor as extractor


TEST_DIR = Path(__file__).resolve().parent
CROP_RARITY_DIR = TEST_DIR / "crop_rarity"
DEBUG_DIR = TEST_DIR / "debug_step2self_5_2" / "rarity"
SUMMARY_PATH = DEBUG_DIR / "rarity_summary.tsv"


def run_rarity_debug() -> list[dict]:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for rarity_dir in sorted(CROP_RARITY_DIR.iterdir(), key=lambda p: p.name):
        if not rarity_dir.is_dir() or not rarity_dir.name.isdigit():
            continue

        expected = int(rarity_dir.name)
        for path in sorted(rarity_dir.glob("*.png")):
            crop_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if crop_bgr is None:
                rows.append({
                    "expected": expected,
                    "predicted": "",
                    "ok": "NO",
                    "file": str(path),
                    "confidence": "0.000",
                    "sample_pixels": "0",
                    "median_hue": "",
                    "median_saturation": "",
                    "median_value": "",
                    "color_fraction": "0.000",
                })
                continue

            info = extractor.detect_rarity_from_crop(crop_bgr)
            predicted = info["rarity"]
            rows.append({
                "expected": expected,
                "predicted": predicted if predicted is not None else "",
                "ok": "YES" if predicted == expected else "NO",
                "file": str(path),
                "confidence": f"{float(info['confidence']):.3f}",
                "sample_pixels": str(int(info["sample_pixels"])),
                "median_hue": "" if info["median_hue"] is None else f"{float(info['median_hue']):.1f}",
                "median_saturation": f"{float(info['median_saturation']):.1f}",
                "median_value": f"{float(info['median_value']):.1f}",
                "color_fraction": f"{float(info['color_fraction']):.3f}",
            })

    headers = [
        "ok",
        "expected",
        "predicted",
        "confidence",
        "sample_pixels",
        "median_hue",
        "median_saturation",
        "median_value",
        "color_fraction",
        "file",
    ]
    with SUMMARY_PATH.open("w", encoding="utf-8", newline="") as f:
        f.write("\t".join(headers) + "\n")
        for row in rows:
            f.write("\t".join(str(row[h]) for h in headers) + "\n")

    return rows


if __name__ == "__main__":
    rows = run_rarity_debug()
    total = len(rows)
    misses = [row for row in rows if row["ok"] != "YES"]
    print(f"[RARITY] total={total} ok={total - len(misses)} misses={len(misses)}")
    print(f"[OUT] {SUMMARY_PATH}")
    for row in misses:
        print(f"[MISS] expected={row['expected']} predicted={row['predicted']} file={row['file']}")
