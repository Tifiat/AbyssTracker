import asyncio
import json
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from playwright.async_api import BrowserContext, Page, Response, async_playwright

from hoyolab_exporter import HOYOLAB_URL, find_browser_exe, wait_for_devtools_port


BASE_DIR = Path(__file__).resolve().parent
PROFILE_DIR = BASE_DIR / "profile"
OUTPUT_DIR = BASE_DIR / "debug_data"

API_INTEREST_TERMS = (
    "avatar",
    "weapon",
    "equip",
    "reliquary",
    "level",
    "rarity",
    "constellation",
    "affix",
    "rank",
)


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except TypeError:
        return repr(value)


def _contains_interest_terms(value: Any) -> bool:
    text = json.dumps(_json_safe(value), ensure_ascii=False).lower()
    return any(term in text for term in API_INTEREST_TERMS)


def _is_game_record_api(url: str) -> bool:
    parsed = urlparse(url)
    return (
        "game_record" in parsed.path
        and "/api/" in parsed.path
        and parsed.path.endswith(tuple(["index", "char_master", "role_combat"]))
    ) or ("/game_record/" in parsed.path and "/api/" in parsed.path)


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
    context._debug_playwright = playwright  # type: ignore[attr-defined]
    context._debug_process = process  # type: ignore[attr-defined]
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
            print("[HoYoLAB Debug] Login window is open. Please log in in the browser.")
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


async def click_characters_block(page: Page) -> None:
    locator = page.locator(".block-title-right").first
    await locator.wait_for(state="visible", timeout=30_000)
    await locator.evaluate("(el) => el.click()")
    await page.wait_for_timeout(3500)


async def collect_dom_cards(page: Page) -> list[dict[str, Any]]:
    return await page.evaluate(
        """
        () => {
            const boxOf = (el) => {
                const r = el.getBoundingClientRect();
                return {
                    x: Math.round(r.x * 100) / 100,
                    y: Math.round(r.y * 100) / 100,
                    width: Math.round(r.width * 100) / 100,
                    height: Math.round(r.height * 100) / 100,
                    top: Math.round(r.top * 100) / 100,
                    right: Math.round(r.right * 100) / 100,
                    bottom: Math.round(r.bottom * 100) / 100,
                    left: Math.round(r.left * 100) / 100,
                };
            };

            const visible = (el) => {
                const r = el.getBoundingClientRect();
                const style = getComputedStyle(el);
                return r.width > 0 && r.height > 0
                    && style.display !== "none"
                    && style.visibility !== "hidden"
                    && Number(style.opacity || 1) > 0;
            };

            const textLooksUseful = (text) => {
                return /(Pair|Lv\\.|Ур\\.|等级|精炼|Rank|C\\d|命之座)/i.test(text);
            };

            const candidates = [];
            const selectors = [
                ".all-role *",
                "[class*='all-role'] *",
                "[class*='role']",
                "[class*='avatar']",
                "[class*='card']",
                "li",
                "section",
                "div",
            ];

            const seen = new Set();
            for (const selector of selectors) {
                for (const el of document.querySelectorAll(selector)) {
                    if (seen.has(el) || !visible(el)) continue;
                    seen.add(el);

                    const r = el.getBoundingClientRect();
                    const text = (el.innerText || "").replace(/\\s+/g, " ").trim();
                    const images = Array.from(el.querySelectorAll("img")).filter(visible);
                    if (images.length < 2) continue;
                    if (!textLooksUseful(text)) continue;
                    if (r.width < 80 || r.width > 520 || r.height < 35 || r.height > 220) continue;

                    const imgInfo = images.map((img, idx) => {
                        const b = boxOf(img);
                        return {
                            image_index: idx,
                            src: img.currentSrc || img.src || "",
                            alt: img.alt || "",
                            className: img.className || "",
                            box: b,
                        };
                    });

                    const likelyCharacter = imgInfo
                        .slice()
                        .sort((a, b) => (b.box.width * b.box.height) - (a.box.width * a.box.height))[0];

                    const possibleWeapons = imgInfo.filter((img) => {
                        if (!likelyCharacter) return false;
                        const area = img.box.width * img.box.height;
                        const charArea = likelyCharacter.box.width * likelyCharacter.box.height;
                        const rightOfCharacter = img.box.left > likelyCharacter.box.left + likelyCharacter.box.width * 0.45;
                        const smallEnough = area < charArea * 0.8;
                        return rightOfCharacter && smallEnough && img.src !== likelyCharacter.src;
                    });

                    candidates.push({
                        text,
                        className: el.className || "",
                        tagName: el.tagName,
                        box: boxOf(el),
                        images: imgInfo,
                        possible_weapon_images: possibleWeapons,
                    });
                }
            }

            candidates.sort((a, b) => {
                const dy = a.box.top - b.box.top;
                if (Math.abs(dy) > 8) return dy;
                return a.box.left - b.box.left;
            });

            const deduped = [];
            for (const item of candidates) {
                const duplicate = deduped.some((prev) => {
                    return Math.abs(prev.box.left - item.box.left) < 3
                        && Math.abs(prev.box.top - item.box.top) < 3
                        && Math.abs(prev.box.width - item.box.width) < 3
                        && Math.abs(prev.box.height - item.box.height) < 3;
                });
                if (!duplicate) {
                    item.card_index = deduped.length;
                    deduped.push(item);
                }
            }

            return deduped;
        }
        """
    )


async def collect_page_images(page: Page) -> list[dict[str, Any]]:
    return await page.evaluate(
        """
        () => Array.from(document.images).map((img, index) => {
            const r = img.getBoundingClientRect();
            return {
                image_index: index,
                src: img.currentSrc || img.src || "",
                alt: img.alt || "",
                className: img.className || "",
                box: {
                    x: Math.round(r.x * 100) / 100,
                    y: Math.round(r.y * 100) / 100,
                    width: Math.round(r.width * 100) / 100,
                    height: Math.round(r.height * 100) / 100,
                    top: Math.round(r.top * 100) / 100,
                    right: Math.round(r.right * 100) / 100,
                    bottom: Math.round(r.bottom * 100) / 100,
                    left: Math.round(r.left * 100) / 100,
                },
            };
        }).filter((item) => item.box.width > 0 && item.box.height > 0)
        """
    )


async def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    network_responses: list[dict[str, Any]] = []

    context = await create_context()
    page = context.pages[0] if context.pages else await context.new_page()

    async def on_response(response: Response) -> None:
        url = response.url
        if not _is_game_record_api(url):
            return

        record: dict[str, Any] = {
            "url": url,
            "status": response.status,
            "headers": dict(response.headers),
            "captured_at": time.time(),
        }

        try:
            content_type = response.headers.get("content-type", "")
            if "json" in content_type or "/api/" in url:
                data = await response.json()
                record["json"] = data
                record["contains_interest_terms"] = _contains_interest_terms(data)
            else:
                text = await response.text()
                record["text"] = text[:200_000]
                record["contains_interest_terms"] = any(
                    term in text.lower() for term in API_INTEREST_TERMS
                )
        except Exception as exc:
            record["error"] = repr(exc)

        network_responses.append(record)
        print(f"[HoYoLAB Debug] Captured API: {url}")

    page.on("response", lambda response: asyncio.create_task(on_response(response)))

    try:
        await page.set_extra_http_headers(
            {
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                "x-rpc-language": "zh-cn",
            }
        )

        print("[HoYoLAB Debug] Opening HoYoLAB...")
        await page.goto(HOYOLAB_URL, wait_until="domcontentloaded", timeout=60_000)
        await wait_until_ready_or_login(page)

        print("[HoYoLAB Debug] Opening character list...")
        await click_characters_block(page)
        try:
            await page.wait_for_load_state("networkidle", timeout=30_000)
        except Exception:
            print("[HoYoLAB Debug] networkidle timeout; continuing with current page state.")

        # Give lazy images and delayed API calls a little time to settle.
        await page.wait_for_timeout(3000)

        dom_cards = await collect_dom_cards(page)
        page_images = await collect_page_images(page)
        html = await page.content()

        (OUTPUT_DIR / "network_responses.json").write_text(
            json.dumps(network_responses, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (OUTPUT_DIR / "dom_cards.json").write_text(
            json.dumps(
                {
                    "source_url": page.url,
                    "captured_at": time.time(),
                    "card_count": len(dom_cards),
                    "cards": dom_cards,
                    "page_image_count": len(page_images),
                    "page_images": page_images,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (OUTPUT_DIR / "page_snapshot.html").write_text(html, encoding="utf-8")
        await page.screenshot(path=str(OUTPUT_DIR / "screenshot_debug.png"), full_page=True)

        print(f"[HoYoLAB Debug] Saved debug data to: {OUTPUT_DIR}")
        print(f"[HoYoLAB Debug] API responses: {len(network_responses)}")
        print(f"[HoYoLAB Debug] DOM cards: {len(dom_cards)}")
        print(f"[HoYoLAB Debug] Page images: {len(page_images)}")

    finally:
        try:
            await page.close()
        except Exception:
            pass

        playwright = getattr(context, "_debug_playwright", None)
        if playwright:
            await playwright.stop()

        process = getattr(context, "_debug_process", None)
        if process and process.poll() is None:
            process.terminate()


if __name__ == "__main__":
    asyncio.run(main())
