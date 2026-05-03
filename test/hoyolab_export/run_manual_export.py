import asyncio
from pathlib import Path

from hoyolab_exporter import HoyolabExporter


BASE_DIR = Path(__file__).resolve().parent

exporter = HoyolabExporter(
    profile_dir=BASE_DIR / "profile",
    download_dir=BASE_DIR / "downloads",
    # Итоговая ширина ≈ fixed_container_width * scale
    # Например: 376 * 4 = 1504 px
    scale=4,
    fixed_container_width=500,
    browser_window_width=1280,
    browser_window_height=900,
    image_format="png",
)


if __name__ == "__main__":
    asyncio.run(exporter.export_manual())
