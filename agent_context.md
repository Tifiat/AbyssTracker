# Agent Context: AbyssTracker Screenshot Auto-Download Feature

This file is a technical handoff for a new Codex chat. Read this before planning or implementing a new button/function for automatic screenshot download/import.

## Project Summary

AbyssTracker is a desktop Python/PySide6 application for Genshin Impact / HoYoLAB screenshots.

The current app flow:

1. User manually selects a HoYoLAB screenshot from disk.
2. `parser.hoyolab_parser.HoyolabParser` detects character and weapon icon squares.
3. Character crops are matched/enriched through `services.icon_enricher_orb`.
4. Weapon crops are prepared and matched through `services.weapon_matcher`.
5. Recognized HD character/weapon icons are copied into `assets/hd/...`.
6. The UI grids refresh and show draggable icons.

Main entry point:

- `main.py`

Main window:

- `ui/main_window.py`

Current active weapon pipeline:

- `services/weapon_crop_extractor.py`
- `services/weapon_ref_preparer.py`
- `services/weapon_dino.py`
- `services/weapon_patch_dino_matcher.py`
- `services/weapon_matcher.py`

Legacy/experimental weapon matchers were removed from runtime. Do not reintroduce old phash/shape/chamfer/legacy matcher code.

## Current Screenshot Import Flow

The manual screenshot button is created in:

- `ui/main_window.py`
- method: `App.build_left_panel`

Current button:

```python
btn_load = QPushButton("Загрузить скриншот HoYoLAB")
btn_load.clicked.connect(self.load_screenshot)
```

Current import method:

- `App.load_screenshot`

It currently does all processing inline:

1. Opens `QFileDialog.getOpenFileName(...)`.
2. Starts timing with `time.perf_counter()`.
3. Calls `check_and_update()`.
4. Creates `HoyolabParser(path)` and calls `parse()`.
5. Calls `enrich_characters_orb(...)`.
6. Reads `debug/orb/report.json` into `char_map`.
7. Loads `data/characters.json` and `data/weapons.json`.
8. Calls `warmup_weapon_cache(...)`.
9. Calls `match_weapons(...)`.
10. Refreshes the UI via `_refresh_ui_after_parse()`.
11. Prints total processing time via `_finish_screenshot_timing(...)`.

Important current constants in `ui/main_window.py`:

```python
CROPS_CHAR = "assets/characters"
CROPS_WEAP = "assets/weapons"
ASSETS_CHAR = "assets/hd/characters"
ASSETS_WEAP = "assets/hd/weapons"
STATE_FILE = "state.json"
RUNS_FILE = "runs_history.json"
```

## Recommended First Refactor

Before adding automatic download, avoid duplicating the whole `load_screenshot` method.

Recommended split:

```python
def load_screenshot(self):
    path, _ = QFileDialog.getOpenFileName(...)
    if not path:
        return
    self.process_screenshot_path(path)

def process_screenshot_path(self, path: str):
    # all existing parser/enrich/match/update logic goes here
```

Then the new auto-download button can do:

```python
downloaded_path = ...
self.process_screenshot_path(downloaded_path)
```

This keeps manual import and automatic import using the exact same pipeline.

## Existing Data Update / Download Patterns

Data-pack update code:

- `services/data_updater.py`
- function: `check_and_update()`

It uses:

```python
urllib.request.urlopen(url, timeout=20)
```

Character/ref icon downloading exists in:

- `services/icon_enricher_orb.py`

There is no existing module for downloading a HoYoLAB screenshot itself.

## Important Runtime Folders

Parser output:

- `assets/characters`
- `assets/weapons`

Final UI icons:

- `assets/hd/characters`
- `assets/hd/weapons`

Debug:

- `debug`
- `debug/orb/report.json`
- `debug/weapons/report_v2_hybrid.json`

Raw/reference weapon cache:

- `cache/enka_ref_weapons`

Local data:

- `data/characters.json`
- `data/weapons.json`
- `data/manifest.json`

## UI Notes

`App` is a `QWidget` in `ui/main_window.py`.

Left panel currently contains:

- weapon scroll area
- character scroll area
- manual screenshot load button
- clear assets button

Right panel contains:

- team slots
- abyss timers
- reset/save/history buttons

New button should probably live near the existing manual screenshot button in `build_left_panel`.

Suggested button text:

- `Скачать скриншот автоматически`
- or `Автозагрузка скриншота`

The current UI code has mojibake-looking text in some places due older encoding issues, but many strings render correctly in the source when opened as UTF-8. Prefer adding new user-facing Russian text as UTF-8 and avoid touching unrelated existing strings.

## Performance / Threading Warning

Current `load_screenshot` runs heavy work synchronously on the UI thread:

- parser
- ORB character matching
- DINO weapon matching
- patch-DINO rerank

This can freeze the UI during processing.

For a minimal first version, it is acceptable to follow the current pattern and keep it synchronous.

For a better version, the new chat should consider moving the whole screenshot workflow to a `QThread`/worker later. Do not mix a large threading refactor with the first auto-download implementation unless explicitly requested.

## Feature To Plan

The requested feature is a button for automatic screenshot download/import.

The source of the screenshot is not defined yet. The new chat must clarify or choose one implementation target before coding. Possible meanings:

1. Download the newest screenshot from a known URL.
2. Watch/read a local Downloads folder and import the newest image.
3. Pull an image from clipboard.
4. Download from HoYoLAB through browser/session/API.
5. Open a local browser automation flow and save a screenshot.

Important: if HoYoLAB authentication/session is required, do not hardcode credentials or tokens. Store user settings locally if needed and keep secrets out of git.

## Suggested Minimal Implementation Path

1. Refactor `App.load_screenshot` into `load_screenshot` + `process_screenshot_path`.
2. Add a small service module only if needed, for example:

   - `services/screenshot_downloader.py`

3. Add a new button in `App.build_left_panel`.
4. The new button handler should:

   - obtain/download/find a screenshot path;
   - validate that it exists and is an image;
   - call `self.process_screenshot_path(path)`;
   - show `QMessageBox.warning(...)` on recoverable errors.

5. Keep manual screenshot loading working exactly as before.
6. Verify with:

   ```powershell
   .\.venv311\Scripts\python.exe -m py_compile .\ui\main_window.py
   ```

7. If a new service module is added, compile it too.

## Things Not To Break

- Do not change weapon matching logic while adding the button.
- Do not change parser behavior.
- Do not clear assets automatically unless the existing workflow already does so or user asks.
- Do not remove debug report generation.
- Do not reintroduce deleted legacy matchers.
- Do not require CUDA-specific behavior for this UI feature.
- Do not add large dependencies unless the screenshot source truly requires them.

## Useful Current Commands

Run the app:

```powershell
.\.venv311\Scripts\python.exe .\main.py
```

Compile key files:

```powershell
.\.venv311\Scripts\python.exe -m py_compile .\ui\main_window.py .\parser\hoyolab_parser.py .\services\weapon_matcher.py
```

Search code:

```powershell
rg -n "load_screenshot|QPushButton|QFileDialog|match_weapons|HoyolabParser" .\ui .\services .\parser -g "*.py"
```

## Recommended Prompt For New Chat

Start the next chat with something like:

```text
Read C:\Users\user\Desktop\AbyssTracker\agent_context.md.
We need to add a new button for automatic screenshot download/import.
First, inspect the current screenshot loading flow and propose a minimal implementation plan.
Do not change weapon matching or parser logic.
```

