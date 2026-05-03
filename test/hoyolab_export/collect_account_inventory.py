import asyncio
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from playwright.async_api import BrowserContext, Page, Response, async_playwright

from hoyolab_exporter import HOYOLAB_URL, find_browser_exe, wait_for_devtools_port


BASE_DIR = Path(__file__).resolve().parent
PROFILE_DIR = BASE_DIR / "profile"
OUTPUT_DIR = BASE_DIR / "debug_data"

CHARACTERS_FILE = OUTPUT_DIR / "account_characters.json"
WEAPONS_FILE = OUTPUT_DIR / "account_weapons.json"

WEAPON_TYPE_NAMES = {
    1: "sword",
    10: "catalyst",
    11: "claymore",
    12: "bow",
    13: "polearm",
}


async def create_context() -> BrowserContext:
    playwright = await async_playwright().start()
    browser_exe = find_browser_exe()

    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    devtools_file = PROFILE_DIR / "DevToolsActivePort"
    if devtools_file.exists():
        try:
            devtools_file.unlink()
        except OSError:
            pass

    process = subprocess.Popen(
        [
            browser_exe,
            "--remote-debugging-port=0",
            f"--user-data-dir={PROFILE_DIR}",
            "--window-size=1280,900",
            "about:blank",
        ]
    )

    debug_port = wait_for_devtools_port(PROFILE_DIR, process)
    browser = await playwright.chromium.connect_over_cdp(
        f"http://127.0.0.1:{debug_port}"
    )

    context = browser.contexts[0]
    context._inventory_playwright = playwright  # type: ignore[attr-defined]
    context._inventory_process = process  # type: ignore[attr-defined]
    return context


async def is_login_open(page: Page) -> bool:
    if await page.locator("iframe#hyv-account-frame").count() > 0:
        return True

    return any(
        "account.hoyolab.com/login-platform" in frame.url for frame in page.frames
    )


async def wait_until_ready_or_login(page: Page, timeout_ms: int = 5 * 60_000) -> None:
    deadline = time.time() + timeout_ms / 1000

    while time.time() < deadline:
        if await is_login_open(page):
            print("[HoYoLAB Inventory] Login window is open. Please log in.")
            while time.time() < deadline and await is_login_open(page):
                await page.wait_for_timeout(1000)
            await page.wait_for_timeout(3000)

        if await page.locator(".block-title-right").count() > 0:
            try:
                if await page.locator(".block-title-right").first.is_visible(timeout=500):
                    return
            except Exception:
                pass

        await page.wait_for_timeout(500)

    raise RuntimeError("HoYoLAB page did not become ready: character button not found.")


async def open_character_list(page: Page) -> None:
    locator = page.locator(".block-title-right").first
    await locator.wait_for(state="visible", timeout=30_000)
    await locator.evaluate("(el) => el.click()")
    await page.wait_for_timeout(2500)


def normalize_character(item: dict[str, Any]) -> dict[str, Any]:
    weapon_type = item.get("weapon_type")
    return {
        "id": item.get("id"),
        "name": item.get("name"),
        "rarity": item.get("rarity"),
        "element": item.get("element"),
        "level": item.get("level"),
        "constellation": item.get("actived_constellation_num"),
        "weapon_type": weapon_type,
        "weapon_type_name": WEAPON_TYPE_NAMES.get(weapon_type, str(weapon_type)),
        "icon": item.get("icon"),
        "side_icon": item.get("side_icon"),
    }


def normalize_weapon(item: dict[str, Any]) -> dict[str, Any] | None:
    weapon = item.get("weapon")
    if not isinstance(weapon, dict):
        return None

    weapon_type = weapon.get("type", item.get("weapon_type"))
    return {
        "id": weapon.get("id"),
        "name": weapon.get("name"),
        "rarity": weapon.get("rarity"),
        "type": weapon_type,
        "type_name": WEAPON_TYPE_NAMES.get(weapon_type, str(weapon_type)),
        "level": weapon.get("level"),
        "refinement": weapon.get("affix_level"),
        "icon": weapon.get("icon"),
        "equipped_by": {
            "id": item.get("id"),
            "name": item.get("name"),
        },
    }


def build_inventory(character_list: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    characters = [normalize_character(item) for item in character_list]

    weapons = []
    for item in character_list:
        weapon = normalize_weapon(item)
        if weapon is not None:
            weapons.append(weapon)

    characters.sort(
        key=lambda c: (
            str(c.get("element") or ""),
            str(c.get("weapon_type_name") or ""),
            -(c.get("rarity") or 0),
            str(c.get("name") or ""),
        )
    )
    weapons.sort(
        key=lambda w: (
            str(w.get("type_name") or ""),
            -(w.get("rarity") or 0),
            str(w.get("name") or ""),
            str(w.get("equipped_by", {}).get("name") or ""),
        )
    )

    return characters, weapons


async def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    context = await create_context()
    page = context.pages[0] if context.pages else await context.new_page()
    character_list_future: asyncio.Future[list[dict[str, Any]]] = asyncio.Future()

    async def on_response(response: Response) -> None:
        if "/event/game_record/genshin/api/character/list" not in response.url:
            return

        try:
            payload = await response.json()
            items = payload.get("data", {}).get("list", [])
            if not isinstance(items, list):
                raise RuntimeError("character/list response has no data.list array")

            if not character_list_future.done():
                character_list_future.set_result(items)

            print(f"[HoYoLAB Inventory] Captured character/list: {len(items)} characters")
        except Exception as exc:
            if not character_list_future.done():
                character_list_future.set_exception(exc)

    page.on("response", lambda response: asyncio.create_task(on_response(response)))

    try:
        # Ask HoYoLAB for Russian names where possible. IDs are also saved, so
        # downstream matching does not have to depend on localized text.
        await page.set_extra_http_headers(
            {
                "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
                "x-rpc-language": "ru-ru",
            }
        )

        print("[HoYoLAB Inventory] Opening HoYoLAB...")
        await page.goto(HOYOLAB_URL, wait_until="domcontentloaded", timeout=60_000)
        await wait_until_ready_or_login(page)

        print("[HoYoLAB Inventory] Opening character list...")
        await open_character_list(page)

        character_list = await asyncio.wait_for(character_list_future, timeout=60)
        characters, weapons = build_inventory(character_list)

        CHARACTERS_FILE.write_text(
            json.dumps(characters, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        WEAPONS_FILE.write_text(
            json.dumps(weapons, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(f"[HoYoLAB Inventory] Characters saved: {CHARACTERS_FILE}")
        print(f"[HoYoLAB Inventory] Weapons saved: {WEAPONS_FILE}")
        print(f"[HoYoLAB Inventory] Character count: {len(characters)}")
        print(f"[HoYoLAB Inventory] Equipped weapon count: {len(weapons)}")

    finally:
        try:
            await page.close()
        except Exception:
            pass

        playwright = getattr(context, "_inventory_playwright", None)
        if playwright:
            await playwright.stop()

        process = getattr(context, "_inventory_process", None)
        if process and process.poll() is None:
            process.terminate()


if __name__ == "__main__":
    asyncio.run(main())
