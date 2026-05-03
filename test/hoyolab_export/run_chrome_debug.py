import subprocess
import time
from pathlib import Path

CHROME_PATHS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]

BASE_DIR = Path(__file__).resolve().parent
PROFILE_DIR = BASE_DIR / "profile_chrome_debug"
DEBUG_PORT = "9222"

HOYOLAB_URL = "https://act.hoyolab.com/app/community-game-records-sea/index.html"


def find_chrome():
    for path in CHROME_PATHS:
        if Path(path).exists():
            return path
    raise FileNotFoundError("Google Chrome не найден")


if __name__ == "__main__":
    chrome_path = find_chrome()
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    subprocess.Popen([
        chrome_path,
        f"--remote-debugging-port={DEBUG_PORT}",
        f"--user-data-dir={PROFILE_DIR}",
        "--start-maximized",
        HOYOLAB_URL,
    ])

    print("Chrome открыт.")
    print("Войди в HoYoLAB вручную в открывшемся окне.")
    print("После входа закрой окно Chrome и запусти этот файл ещё раз для проверки.")

    time.sleep(3)