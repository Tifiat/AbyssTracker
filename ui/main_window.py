import os
import shutil
import json
import cv2
import time

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
	QWidget,
	QLabel,
	QPushButton,
	QVBoxLayout,
	QHBoxLayout,
	QGridLayout,
	QScrollArea,
	QFileDialog,
	QMessageBox,
)

from services.weapon_matcher import match_weapons
from services.icon_enricher_orb import enrich_characters_orb, warmup_weapon_cache
from services.data_updater import check_and_update
from parser.hoyolab_parser import HoyolabParser
from ui.run_history_window import RunHistoryWindow
from ui.widgets.drag import DraggableIcon
from ui.widgets.team import TeamSlot
from ui.widgets.timers import AbyssFloorRow

CROPS_CHAR = "assets/characters"
CROPS_WEAP = "assets/weapons"

ASSETS_CHAR = "assets/hd/characters"
ASSETS_WEAP = "assets/hd/weapons"

STATE_FILE = "state.json"
RUNS_FILE = "runs_history.json"


class App(QWidget):
	def __init__(self):
		super().__init__()
		self.setWindowTitle("Abyss Team Builder")
		self.resize(1400, 800)

		self.main = QHBoxLayout(self)
		self.floors = []
		self.teams = []

		self.ctrl_pressed = False
		self.pending_grid_updates = False
		self._resize_timer = None
		self._ui_ready = False
		self._initial_grid_built = False

		self.build_left_panel()
		self.build_right_panel()
		self.load_state()

	def showEvent(self, event):
		super().showEvent(event)

		if not self._initial_grid_built:
			self._initial_grid_built = True
			QTimer.singleShot(0, self._finish_initial_ui)

	def _finish_initial_ui(self):
		self.update_grids()
		self._ui_ready = True

	def _refresh_ui_after_parse(self):
		QTimer.singleShot(0, self.safe_update_grids)

	def keyPressEvent(self, event):
		if event.key() == Qt.Key_Control:
			self.ctrl_pressed = True
		super().keyPressEvent(event)

	def keyReleaseEvent(self, event):
		if event.key() == Qt.Key_Control:
			self.ctrl_pressed = False
			if self.pending_grid_updates:
				self.update_grids()
				self.pending_grid_updates = False
		super().keyReleaseEvent(event)

	def safe_update_grids(self):
		if self.ctrl_pressed:
			self.pending_grid_updates = True
		else:
			self.update_grids()

	# ---------- HISTORY ----------
	def open_run_history(self):
		if not hasattr(self, "_run_history_window") or self._run_history_window is None:
			self._run_history_window = RunHistoryWindow()
		else:
			self._run_history_window._first_show = True
			self._run_history_window._min_width = None
			self._run_history_window.reload()

		self._run_history_window.show()
		self._run_history_window.raise_()
		self._run_history_window.activateWindow()

	# ---------- LEFT PANEL ----------
	def build_left_panel(self):
		left = QVBoxLayout()

		btn_load = QPushButton("Загрузить скриншот HoYoLAB")
		btn_load.clicked.connect(self.load_screenshot)

		btn_clear = QPushButton("Очистить персонажей и оружие")
		btn_clear.clicked.connect(self.clear_assets)

		left.addWidget(QLabel("Оружие"))

		self.weapon_area = QScrollArea()
		self.weapon_area.setWidgetResizable(True)
		self.weapon_widget = QWidget()
		self.weapon_grid = QGridLayout(self.weapon_widget)
		self.weapon_area.setWidget(self.weapon_widget)
		left.addWidget(self.weapon_area, 1)

		left.addWidget(QLabel("Персонажи"))

		self.char_area = QScrollArea()
		self.char_area.setWidgetResizable(True)
		self.char_widget = QWidget()
		self.char_grid = QGridLayout(self.char_widget)
		self.char_area.setWidget(self.char_widget)
		left.addWidget(self.char_area, 3)

		left.addWidget(btn_load)
		left.addWidget(btn_clear)

		self.main.addLayout(left, 2)

	# ---------- RIGHT PANEL ----------
	def build_right_panel(self):
		right = QVBoxLayout()

		for i in range(2):
			team = []
			right.addWidget(QLabel(f"Команда {i + 1}"))
			row = QHBoxLayout()
			for _ in range(4):
				slot = TeamSlot()
				team.append(slot)
				row.addWidget(slot)
			self.teams.append(team)
			right.addLayout(row)

		right.addSpacing(20)
		right.addWidget(QLabel("Таймеры бездны"))

		for i in range(1, 4):
			floor = AbyssFloorRow(i, self.calculate_abyss)
			self.floors.append(floor)
			right.addWidget(floor)

		self.total_label = QLabel("Итого: 0 сек")
		right.addWidget(self.total_label)

		btn_reset = QPushButton("Сбросить забег")
		btn_reset.clicked.connect(self.reset_run)
		right.addWidget(btn_reset)

		btn_save = QPushButton("Сохранить забег")
		btn_save.clicked.connect(self.save_run)
		right.addWidget(btn_save)

		btn_history = QPushButton("Открыть историю забегов")
		btn_history.clicked.connect(self.open_run_history)
		right.addWidget(btn_history)

		right.addStretch()
		self.main.addLayout(right, 1)

	# ---------- GRID METHODS ----------
	def _clear_grid(self, grid):
		while grid.count():
			item = grid.takeAt(0)
			w = item.widget()
			if w is not None:
				w.setParent(None)
				w.deleteLater()

	def reload_characters(self):
		self._clear_grid(self.char_grid)

		if not os.path.exists(ASSETS_CHAR):
			self.char_widget.adjustSize()
			return

		files = [f for f in sorted(os.listdir(ASSETS_CHAR)) if f.lower().endswith(".png")]
		if not files:
			self.char_widget.adjustSize()
			return

		available_width = self.char_area.viewport().width()
		if available_width <= 20:
			available_width = self.char_area.width()
		if available_width <= 20:
			available_width = 300

		icon_size = 72
		fixed_spacing = 3
		cell_width = icon_size + fixed_spacing

		cols = max(1, (available_width + fixed_spacing) // cell_width)
		total_grid_width = cols * icon_size + max(0, cols - 1) * fixed_spacing

		left_margin = max(0, (available_width - total_grid_width) // 2)
		right_margin = max(0, available_width - total_grid_width - left_margin)

		self.char_grid.setContentsMargins(left_margin, 0, right_margin, 0)
		self.char_grid.setHorizontalSpacing(fixed_spacing)
		self.char_grid.setVerticalSpacing(fixed_spacing)

		for c in range(cols):
			self.char_grid.setColumnMinimumWidth(c, icon_size)
			self.char_grid.setColumnStretch(c, 0)

		for i, f in enumerate(files):
			try:
				icon = DraggableIcon(os.path.join(ASSETS_CHAR, f), icon_size)
				row = i // cols
				col = i % cols
				self.char_grid.addWidget(icon, row, col)
			except Exception as e:
				print(f"Ошибка загрузки {f}: {e}")

		self.char_widget.adjustSize()
		self.char_widget.updateGeometry()
		self.char_area.viewport().update()

	def reload_weapons(self):
		self._clear_grid(self.weapon_grid)

		if not os.path.exists(ASSETS_WEAP):
			self.weapon_widget.adjustSize()
			return

		files = [f for f in sorted(os.listdir(ASSETS_WEAP)) if f.lower().endswith(".png")]
		if not files:
			self.weapon_widget.adjustSize()
			return

		available_width = self.weapon_area.viewport().width()
		if available_width <= 20:
			available_width = self.weapon_area.width()
		if available_width <= 20:
			available_width = 300

		icon_size = 48
		fixed_spacing = 6
		cell_width = icon_size + fixed_spacing

		cols = max(1, (available_width + fixed_spacing) // cell_width)
		total_grid_width = cols * icon_size + max(0, cols - 1) * fixed_spacing

		left_margin = max(0, (available_width - total_grid_width) // 2)
		right_margin = max(0, available_width - total_grid_width - left_margin)

		self.weapon_grid.setContentsMargins(left_margin, 0, right_margin, 0)
		self.weapon_grid.setHorizontalSpacing(fixed_spacing)
		self.weapon_grid.setVerticalSpacing(fixed_spacing)

		for c in range(cols):
			self.weapon_grid.setColumnMinimumWidth(c, icon_size)
			self.weapon_grid.setColumnStretch(c, 0)

		for i, f in enumerate(files):
			try:
				icon = DraggableIcon(os.path.join(ASSETS_WEAP, f), icon_size)
				row = i // cols
				col = i % cols
				self.weapon_grid.addWidget(icon, row, col)
			except Exception as e:
				print(f"Ошибка загрузки {f}: {e}")

		self.weapon_widget.adjustSize()
		self.weapon_widget.updateGeometry()
		self.weapon_area.viewport().update()

	# ---------- RESIZE ----------
	def resizeEvent(self, event):
		super().resizeEvent(event)
		if self._ui_ready:
			self.update_grids_delayed()

	def update_grids_delayed(self):
		if self._resize_timer is None:
			self._resize_timer = QTimer(self)
			self._resize_timer.setSingleShot(True)
			self._resize_timer.timeout.connect(self.update_grids)

		self._resize_timer.start(75)

	def update_grids(self):
		self.reload_characters()
		self.reload_weapons()

	# ---------- LOGIC ----------
	def _finish_screenshot_timing(self, started_at: float, label: str = "SCREENSHOT WORKFLOW TOTAL"):
		elapsed = time.perf_counter() - started_at
		print(f"{label}: {elapsed:.2f}s ({elapsed / 60.0:.2f} min)")

	def calculate_abyss(self):
		total = sum(f.calculate() for f in self.floors)
		self.total_label.setText(f"Итого: {total} сек")
		self.save_state()

	def save_state(self):
		data = {
			"floors": [{"t1": f.t1.seconds_left, "t2": f.t2.seconds_left} for f in self.floors],
			"teams": [[slot.to_dict() for slot in team] for team in self.teams],
		}
		with open(STATE_FILE, "w", encoding="utf-8") as f:
			json.dump(data, f, indent=2)

	def load_state(self):
		if not os.path.exists(STATE_FILE):
			return

		try:
			with open(STATE_FILE, "r", encoding="utf-8") as f:
				data = json.load(f)
		except Exception as e:
			print(f"Ошибка загрузки состояния: {e}")
			return

		for team_slots, saved_team in zip(self.teams, data.get("teams", [])):
			for slot, saved in zip(team_slots, saved_team):
				slot.from_dict(saved)

		for floor, saved in zip(self.floors, data.get("floors", [])):
			floor.t1.seconds_left = saved.get("t1", 600)
			floor.t2.seconds_left = saved.get("t2", 600)
			floor.t1.min_spin.setValue(floor.t1.seconds_left // 60)
			floor.t1.sec_spin.setValue(floor.t1.seconds_left % 60)
			floor.t2.min_spin.setValue(floor.t2.seconds_left // 60)
			floor.t2.sec_spin.setValue(floor.t2.seconds_left % 60)

	def load_screenshot(self):
		path, _ = QFileDialog.getOpenFileName(
			self, "Выбрать скриншот", "", "Images (*.png *.jpg *.jpeg)"
		)
		if not path:
			return

		workflow_started_at = time.perf_counter()
		print(f"SCREENSHOT WORKFLOW START: {os.path.basename(path)}")

		try:
			check_and_update()
		except Exception as e:
			print("check_and_update failed:", e)

		parser = HoyolabParser(path)
		parsed = parser.parse()

		try:
			res_chars = enrich_characters_orb(
				crops_char_dir=CROPS_CHAR,
				data_dir="data",
				out_hd_dir=ASSETS_CHAR,
				debug_dir="debug/orb",
				score_threshold=28,
				margin=6,
			)
			print("ORB chars:", res_chars)
		except Exception as e:
			print("enrich_characters_orb failed:", e)

		char_map = {}
		try:
			with open("debug/orb/report.json", "r", encoding="utf-8") as f:
				rep = json.load(f)
			for it in rep.get("accepted", []):
				if "crop" in it and "id" in it:
					char_map[str(it["crop"])] = str(it["id"])
		except Exception as e:
			print("Не удалось прочитать debug/orb/report.json:", e)

		if not char_map:
			print("char_map пустой — оружие пропущено (нет распознанных персонажей).")
			self._refresh_ui_after_parse()
			self._finish_screenshot_timing(workflow_started_at)
			return

		try:
			with open("data/characters.json", "r", encoding="utf-8") as f:
				chars_db = json.load(f)
			with open("data/weapons.json", "r", encoding="utf-8") as f:
				weaps_db = json.load(f)
		except Exception as e:
			print("Не удалось загрузить data/*.json:", e)
			self._refresh_ui_after_parse()
			self._finish_screenshot_timing(workflow_started_at)
			return

		try:
			res_cache = warmup_weapon_cache(
				data_dir="data",
				cache_dir="cache/enka_ref_weapons",
				allow_download=True,
			)
			print("WEAPON CACHE WARMUP:", res_cache)
		except Exception as e:
			print("warmup_weapon_cache failed:", e)

		try:
			res_weapons = match_weapons(
				parsed=parsed,
				char_map=char_map,
				chars_db=chars_db,
				weaps_db=weaps_db,
				crops_weap_dir=CROPS_WEAP,
				out_hd_weap_dir=ASSETS_WEAP,
				cache_weapons_dir="cache/enka_ref_weapons",
				debug_dir="debug/weapons",
			)
			print("WEAPONS MATCH RESULT:", res_weapons)
		except Exception as e:
			print("match_weapons failed:", e)

		self._refresh_ui_after_parse()
		self._finish_screenshot_timing(workflow_started_at)

	def clear_assets(self):
		folders_to_clear = [
			"assets/characters",
			"assets/weapons",
			"assets/hd/characters",
			"assets/hd/weapons",
			"debug",
		]

		for folder in folders_to_clear:
			shutil.rmtree(folder, ignore_errors=True)
			os.makedirs(folder, exist_ok=True)

		self._refresh_ui_after_parse()
		QMessageBox.information(self, "Готово", "Кропы/HD/дебаг очищены")

	def reset_run(self):
		for floor in self.floors:
			floor.t1.min_spin.setValue(10)
			floor.t1.sec_spin.setValue(0)
			floor.t1.seconds_left = 600
			floor.t1.result.setText("0")

			floor.t2.min_spin.setValue(10)
			floor.t2.sec_spin.setValue(0)
			floor.t2.seconds_left = 600
			floor.t2.result.setText("0")

			floor.total.setText("0")

		for team in self.teams:
			for slot in team:
				slot.char.clear()
				slot.weapon.clear()
				slot.artifact.clear()

		self.total_label.setText("Итого: 0 сек")
		self.save_state()

	def save_run(self):
		team1_floors = []
		team2_floors = []

		for f in self.floors:
			t1_left = f.t1.seconds_left
			t2_left = f.t2.seconds_left

			t1_spent = 600 - t1_left
			t2_spent = t1_left - t2_left

			team1_floors.append(t1_spent)
			team2_floors.append(t2_spent)

		run = {
			"teams": {
				"team1": [slot.to_dict() for slot in self.teams[0]],
				"team2": [slot.to_dict() for slot in self.teams[1]],
			},
			"floors": {
				"team1": team1_floors,
				"team2": team2_floors,
			},
		}

		runs = []
		if os.path.exists(RUNS_FILE):
			try:
				with open(RUNS_FILE, "r", encoding="utf-8") as f:
					runs = json.load(f)
			except Exception as e:
				print(f"Ошибка загрузки истории: {e}")

		runs.append(run)

		with open(RUNS_FILE, "w", encoding="utf-8") as f:
			json.dump(runs, f, indent=2, ensure_ascii=False)

		if hasattr(self, "_run_history_window"):
			self._run_history_window.reload()
