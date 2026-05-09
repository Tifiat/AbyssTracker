# AbyssTracker

> Legacy project. Development moved to `GenshinTeamsTracker`.

`AbyssTracker` was an experimental desktop tool for building and saving Genshin Impact Abyss teams from HoYoLAB screenshots. The project reached a useful MVP state, but its main parsing pipeline became too heavy and fragile, so it was archived as legacy and replaced by a new HoYoLAB-first project.

## What This Project Did

The app was built around this workflow:

1. The user loads a HoYoLAB character screenshot from disk.
2. The parser detects character and weapon icon regions in the image.
3. Character crops are matched/enriched against local data.
4. Weapon crops are matched through the later DINO-based weapon pipeline.
5. Recognized icons are copied into `assets/hd/characters` and `assets/hd/weapons`.
6. The PySide6 UI shows draggable character/weapon icons.
7. The user builds two Abyss teams, tracks chamber timers, and can save/view run history.

The main entry point is:

```powershell
python main.py
```

## Current Status

This repository is preserved for reference, not active feature work.

Useful parts:

- PySide6 main window and widget behavior.
- Drag-and-drop team builder.
- Timer and run-history UI.
- Early HoYoLAB screenshot parsing experiments.
- Weapon/character crop and matching experiments.
- HoYoLAB browser-export sandbox under `test/hoyolab_export/`.

Legacy parts:

- Binary-mask screenshot parsing as the main source of truth.
- OpenCV-heavy icon detection.
- ORB/DINO matching experiments mixed into app flow.
- Debug folders and generated experiment outputs.
- Mojibake text in older UI strings.

The successor project should prefer HoYoLAB-generated exports, account API data, and DOM/layout coordinates instead of trying to infer everything from pixels.

## Project Structure

```text
AbyssTracker/
  main.py                    # PySide6 app entry point
  ui/                        # Main window, history window, draggable widgets, timers
  parser/                    # HoYoLAB screenshot parser
  services/                  # Data updater, character/weapon enrichment and matching
  data/                      # Local Genshin metadata cache
  assets/                    # Generated crops and final UI icons
  cache/                     # Downloaded/reference cache
  debug/                     # Debug reports and intermediate outputs
  test/                      # Experiments, old probes, HoYoLAB exporter sandbox
  agent_context.md           # Historical technical handoff notes
```

Important runtime paths:

- `assets/characters` - raw detected character crops.
- `assets/weapons` - raw detected weapon crops.
- `assets/hd/characters` - final character icons used by the UI.
- `assets/hd/weapons` - final weapon icons used by the UI.
- `runs_history.json` - saved run history.
- `state.json` - current UI state.

## Main Runtime Flow

The original screenshot import button lives in:

```text
ui/main_window.py
```

The app flow is roughly:

```text
QFileDialog screenshot pick
  -> services.data_updater.check_and_update()
  -> parser.hoyolab_parser.HoyolabParser.parse()
  -> services.icon_enricher_orb.enrich_characters_orb()
  -> services.weapon_matcher.match_weapons()
  -> refresh PySide6 icon grids
```

The UI itself is still valuable as a prototype for:

- left-side character/weapon asset panels;
- right-side team slots;
- Abyss timers;
- saved run/history window;
- draggable icons.

## HoYoLAB Export Sandbox

The later and more promising export work lived in:

```text
test/hoyolab_export/
```

That sandbox explored a better direction:

- launching Chrome/Edge with a persistent profile;
- letting the user log in to HoYoLAB;
- automatically clicking through HoYoLAB controls;
- downloading a HoYoLAB-generated image;
- collecting account character/weapon data;
- debugging DOM/card positions.

That direction became the basis for the new `GenshinTeamsTracker` project.

## Installation

This was developed on Windows with Python and PySide6. The repo contains pinned/experimental dependencies in `requirements.txt`.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Some matching experiments require heavier packages:

- `opencv-python`
- `pillow`
- `torch`
- `torchvision`
- `torchaudio`
- `timm`

Depending on machine/GPU setup, the weapon matching experiments may need additional environment work.

## Building an EXE

The original MVP note used PyInstaller:

```powershell
pyinstaller --onefile --windowed --name AbyssTracker main.py
```

This is preserved as a historical note. The current repository was not cleaned for a polished distributable build.

## Known Limitations

- The pixel/CV pipeline is fragile and can detect false positives.
- Weapon detection was especially experimental and accumulated multiple matching approaches.
- Heavy processing runs synchronously on the UI thread and may freeze the window.
- Some old UI strings contain mojibake from earlier encoding issues.
- Generated/debug files may contain personal gameplay/account visuals.
- The app was not designed as a clean public package.

## Why It Was Replaced

The project proved the UI idea, but the parsing strategy became the wrong foundation. HoYoLAB already knows the structured data and renders the export card itself, so the next project should use:

- HoYoLAB export image as the visual source;
- HoYoLAB account data as structured truth;
- DOM/layout coordinates for precise crops;
- a cleaner bundle pipeline instead of binary-mask icon detection.

That successor is `GenshinTeamsTracker`.

## Archive Notes

Keep this repo as a reference for:

- UI behavior worth porting;
- history/timer UX ideas;
- old CV/matching experiments;
- HoYoLAB exporter research.

Avoid using it as the architecture for new work. The useful lesson from `AbyssTracker` is not the final parser code; it is the path away from parser-heavy screenshot inference.
