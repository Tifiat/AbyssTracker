import asyncio
import re
import subprocess
import time
from pathlib import Path
from typing import Optional

from PIL import Image
from playwright.async_api import async_playwright, Route, Request, BrowserContext, Page


HOYOLAB_URL = "https://act.hoyolab.com/app/community-game-records-sea/index.html"
CHROME_PATHS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]

EDGE_PATHS = [
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
]

def wait_for_devtools_port(profile_dir: Path, process: subprocess.Popen, timeout_sec: int = 15) -> int:
    devtools_file = profile_dir / "DevToolsActivePort"
    deadline = time.time() + timeout_sec

    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Р‘СЂР°СѓР·РµСЂ Р·Р°РєСЂС‹Р»СЃСЏ РґРѕ Р·Р°РїСѓСЃРєР° CDP.")

        if devtools_file.exists():
            lines = devtools_file.read_text(encoding="utf-8", errors="ignore").splitlines()
            if lines:
                return int(lines[0])

        time.sleep(0.2)

    raise RuntimeError("РќРµ РґРѕР¶РґР°Р»РёСЃСЊ CDP-РїРѕСЂС‚Р° РѕС‚ Р±СЂР°СѓР·РµСЂР°.")


def find_browser_exe() -> str:
    for path in CHROME_PATHS:
        if Path(path).exists():
            return path

    for path in EDGE_PATHS:
        if Path(path).exists():
            return path

    raise FileNotFoundError(
        "Не найден Google Chrome или Microsoft Edge. "
        "Установите один из этих браузеров."
    )


async def close_export_context(context: BrowserContext) -> None:
    try:
        pages = list(context.pages)
        for page in pages:
            try:
                if not page.is_closed():
                    await page.close()
            except Exception:
                pass
    finally:
        playwright = getattr(context, "_playwright_instance", None)
        if playwright:
            await playwright.stop()

        process = getattr(context, "_browser_process", None)
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

        await asyncio.sleep(0.2)


class HoyolabExporter:
    def __init__(
        self,
        profile_dir: str | Path,
        download_dir: str | Path,
        scale: int = 4,
        fixed_container_width: int = 376,
        browser_window_width: int = 1280,
        browser_window_height: int = 900,
        image_format: str = "png",
    ):
        self.profile_dir = Path(profile_dir)
        self.download_dir = Path(download_dir)
        self.scale = scale
        self.fixed_container_width = fixed_container_width
        self.browser_window_width = browser_window_width
        self.browser_window_height = browser_window_height
        self.image_format = image_format.lower()

        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self.download_dir.mkdir(parents=True, exist_ok=True)

        if self.image_format not in {"png", "jpeg", "jpg"}:
            raise ValueError("image_format должен быть png или jpeg")

    async def _patch_api_language_route(self, route: Route, request: Request):
        url = request.url

        # JS-файлы не трогаем здесь, иначе можно сломать scale-патч
        if url.endswith(".js"):
            await route.continue_()
            return

        headers = dict(request.headers)
        headers["accept-language"] = "zh-CN,zh;q=0.9,en;q=0.8"
        headers["x-rpc-language"] = "zh-cn"

        await route.continue_(headers=headers)

    async def _is_login_open(self, page: Page) -> bool:
        if await page.locator("iframe#hyv-account-frame").count() > 0:
            return True

        for frame in page.frames:
            if "account.hoyolab.com/login-platform" in frame.url:
                return True

        return False

    async def _wait_for_login_if_needed(self, page: Page, timeout_ms: int = 5 * 60_000):
        if not await self._is_login_open(page):
            return

        print("[HoYoLAB Exporter] Открыто окно входа. Войдите в аккаунт в браузере.")
        print("[HoYoLAB Exporter] После успешного входа экспорт продолжится автоматически.")

        deadline = time.time() + timeout_ms / 1000

        while time.time() < deadline:
            if not await self._is_login_open(page):
                print("[HoYoLAB Exporter] Вход выполнен. Продолжаю экспорт...")
                await page.wait_for_timeout(3000)
                return

            await page.wait_for_timeout(1000)

        raise RuntimeError("Истекло время ожидания входа в HoYoLAB.")
    async def _block_user_input(self, page: Page):
        await page.evaluate("""
                            () => {
                                if (document.getElementById('__abyss_tracker_blocker__')) return;

                                const blocker = document.createElement('div');
                                blocker.id = '__abyss_tracker_blocker__';
                                blocker.style.position = 'fixed';
                                blocker.style.inset = '0';
                                blocker.style.zIndex = '2147483647';
                                blocker.style.background = 'rgba(0,0,0,0)';
                                blocker.style.cursor = 'wait';
                                blocker.style.pointerEvents = 'auto';

                                document.body.appendChild(blocker);
                                document.body.style.overflow = 'hidden';
                            }
                            """)

    async def _unblock_user_input(self, page: Page):
        await page.evaluate("""
                            () => {
                                const blocker = document.getElementById('__abyss_tracker_blocker__');
                                if (blocker) blocker.remove();
                                document.body.style.overflow = '';
                            }
                            """)

    async def _patch_js_route(self, route: Route, request: Request):
        url = request.url

        try:
            response = await route.fetch()
            body = await response.text()
        except Exception as exc:
            print(f"[HoYoLAB Exporter] Не удалось прочитать JS для подмены: {exc}")
            await route.continue_()
            return

        original_body = body

        body = re.sub(r"scale\s*:\s*2", f"scale:{self.scale}", body)
        body = re.sub(
            r"r=\{useCORS:!0,backgroundColor:null,scale:(\d+)\}",
            f"r={{useCORS:!0,backgroundColor:null,scale:\\1,width:{self.fixed_container_width},windowWidth:{self.fixed_container_width}}}",
            body,
        )
        body = body.replace(
            ",n.next=5,f()(t,r);case 5:",
            (
                ",t&&t.style&&(t.style.setProperty('width','"
                f"{self.fixed_container_width}px','important'),"
                "t.style.setProperty('min-width','"
                f"{self.fixed_container_width}px','important'),"
                "t.style.setProperty('max-width','"
                f"{self.fixed_container_width}px','important')),"
                "n.next=5,f()(t,r);case 5:"
            ),
        )

        if self.image_format == "png":
            body = body.replace('toDataURL("image/jpeg")', 'toDataURL("image/png")')
            body = body.replace("toDataURL('image/jpeg')", "toDataURL('image/png')")

        if original_body != body:
            print(f"[HoYoLAB Exporter] JS patched: {url}")
        else:
            print(f"[HoYoLAB Exporter] JS найден, но нужные строки не заменены: {url}")

        headers = dict(response.headers)
        headers["content-type"] = "application/javascript"

        await route.fulfill(
            status=response.status,
            headers=headers,
            body=body,
        )

    async def _wait_until_ready_or_login(self, page: Page, timeout_ms: int = 5 * 60_000):
        deadline = time.time() + timeout_ms / 1000

        while time.time() < deadline:
            if await self._is_login_open(page):
                await self._wait_for_login_if_needed(page, timeout_ms=timeout_ms)

            if await page.locator(".block-title-right").count() > 0:
                try:
                    if await page.locator(".block-title-right").first.is_visible(timeout=500):
                        return
                except Exception:
                    pass

            await page.wait_for_timeout(500)

        raise RuntimeError("Страница HoYoLAB не загрузилась: не найден вход и не найдена кнопка персонажей.")

    async def _js_click(self, page: Page, selector: str, timeout: int = 30_000):
        locator = page.locator(selector).first
        await locator.wait_for(state="visible", timeout=timeout)
        await locator.evaluate("(el) => el.click()")

    async def _run_export_flow(self, page: Page):
        await page.wait_for_load_state("domcontentloaded")
        await self._wait_until_ready_or_login(page)

        await self._block_user_input(page)

        try:
            await self._js_click(page, ".block-title-right")
            await page.wait_for_timeout(2500)

            await self._js_click(page, ".me-share__btn")
            await page.wait_for_timeout(2500)

            async with page.expect_download(timeout=90_000) as download_info:
                await self._js_click(
                    page,
                    '.me-share-popover__item:has(img[src*="35b0742f6ed3b58d65f1491ca1bf94e2"])',
                    timeout=30_000,
                )

            return await download_info.value

        finally:
            await self._unblock_user_input(page)

    async def _prepare_export_page(self, page: Page):
        # Только для экспортной вкладки: подменяем scale и ширину html2canvas.
        await page.route("**/*role_combat_tarot*.js", self._patch_js_route)

        # Только для экспортной вкладки: подменяем язык API
        await page.route("**/game_record/**", self._patch_api_language_route)
        await page.route("**/event/game_record/**", self._patch_api_language_route)

        # Заголовки только для этой вкладки
        await page.set_extra_http_headers({
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "x-rpc-language": "zh-cn",
        })



    async def _create_context(self) -> BrowserContext:
        playwright = await async_playwright().start()

        browser_exe = find_browser_exe()

        self.profile_dir.mkdir(parents=True, exist_ok=True)
        devtools_file = self.profile_dir / "DevToolsActivePort"
        if devtools_file.exists():
            try:
                devtools_file.unlink()
            except OSError:
                pass

        process = subprocess.Popen([
            browser_exe,
            "--remote-debugging-port=0",
            f"--user-data-dir={self.profile_dir}",
            f"--window-size={self.browser_window_width},{self.browser_window_height}",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-session-crashed-bubble",
            "about:blank",
        ])

        debug_port = wait_for_devtools_port(self.profile_dir, process)

        browser = await playwright.chromium.connect_over_cdp(
            f"http://127.0.0.1:{debug_port}"
        )

        context = browser.contexts[0]

        context._playwright_instance = playwright  # type: ignore[attr-defined]
        context._browser_process = process  # type: ignore[attr-defined]
        return context

    async def open_login_page(self):
        """
        Просто открывает HoYoLAB.
        Используется для первого входа в аккаунт.
        """
        context = await self._create_context()
        page = context.pages[0] if context.pages else await context.new_page()

        print("[HoYoLAB Exporter] Открываю HoYoLAB. Войдите в аккаунт, если нужно.")
        await page.goto(HOYOLAB_URL, wait_until="domcontentloaded", timeout=60_000)

        print("[HoYoLAB Exporter] Браузер оставлен открытым. После входа можно закрыть окно вручную.")

        # Не закрываем сразу, чтобы пользователь мог логиниться.
        while True:
            await asyncio.sleep(1)

    async def export_manual(self) -> Optional[Path]:
        context = await self._create_context()

        try:
            export_page = context.pages[0] if context.pages else await context.new_page()

            await self._prepare_export_page(export_page)
            await export_page.goto(HOYOLAB_URL, wait_until="domcontentloaded", timeout=60_000)

            print()
            print("Страница открыта. Запускаю автоматический экспорт...")
            print()

            download = await self._run_export_flow(export_page)

            suggested_name = download.suggested_filename or "hoyolab_export"

            if self.image_format == "png" and not suggested_name.lower().endswith(".png"):
                suggested_name = Path(suggested_name).stem + ".png"

            save_path = self.download_dir / suggested_name
            await download.save_as(str(save_path))

            print(f"[HoYoLAB Exporter] Файл сохранён: {save_path}")

            self._validate_image(save_path)

            return save_path

        except Exception as exc:
            print(f"[HoYoLAB Exporter] Ошибка экспорта: {exc}")
            raise



        finally:

            await close_export_context(context)

    def _validate_image(self, path: Path):
        try:
            with Image.open(path) as img:
                width, height = img.size
                print(f"[HoYoLAB Exporter] Размер изображения: {width} × {height}")

                expected_width = self.fixed_container_width * self.scale
                if width < expected_width * 0.9:
                    print(
                        "[HoYoLAB Exporter] Предупреждение: "
                        f"ширина {width}px ниже ожидаемой {expected_width}px. "
                        "Возможно, scale или fixed_container_width не применились."
                    )

        except Exception as exc:
            print(f"[HoYoLAB Exporter] Не удалось проверить изображение: {exc}")
