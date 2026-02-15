import os
import json
import urllib.request

DATA_URL = "https://raw.githubusercontent.com/Tifiat/AbyssTracker-data/main"
LOCAL_DATA_DIR = "data"


def _download_file(url: str, path: str):
    with urllib.request.urlopen(url, timeout=20) as response:
        data = response.read()
    with open(path, "wb") as f:
        f.write(data)


def check_and_update():
    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)

    manifest_url = f"{DATA_URL}/manifest.json"
    local_manifest_path = os.path.join(LOCAL_DATA_DIR, "manifest.json")

    try:
        with urllib.request.urlopen(manifest_url, timeout=20) as response:
            remote_manifest = json.loads(response.read().decode("utf-8"))
    except Exception as e:
        print("Не удалось проверить data-pack:", e)
        return False

    local_manifest = None
    if os.path.exists(local_manifest_path):
        try:
            with open(local_manifest_path, "r", encoding="utf-8") as f:
                local_manifest = json.load(f)
        except Exception:
            pass

    if local_manifest and local_manifest.get("version") == remote_manifest.get("version"):
        print("Data-pack актуален")
        return False

    print("Обновляем data-pack...")

    files = [
        "characters.json",
        "weapons.json",
        "hash_index_characters.json",
        "hash_index_weapons.json",
        "manifest.json",
    ]

    for fname in files:
        url = f"{DATA_URL}/{fname}"
        path = os.path.join(LOCAL_DATA_DIR, fname)
        _download_file(url, path)

    print("Data-pack обновлён")
    return True
