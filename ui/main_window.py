import os
import shutil
import json

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
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
    QSpinBox,
)

from parser.hoyolab_parser import HoyolabParser
from ui.run_history_window import RunHistoryWindow
from ui.widgets.drag import DraggableIcon
from ui.widgets.team import TeamSlot
from ui.widgets.timers import AbyssFloorRow

ASSETS_CHAR = "assets/characters"
ASSETS_WEAP = "assets/weapons"
STATE_FILE = "state.json"
RUNS_FILE = "runs_history.json"

# ===============================
# MAIN WINDOW
# ===============================
class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Abyss Team Builder")
        self.resize(1400, 800)

        self.main = QHBoxLayout(self)
        self.floors = []
        self.teams = []

        self.build_left_panel()
        self.build_right_panel()
        self.load_state()

    # ---------- HISTORY ----------
    def open_run_history(self):
        # если окна ещё нет, создаём его
        if not hasattr(self, "_run_history_window") or self._run_history_window is None:
            self._run_history_window = RunHistoryWindow()
        else:
            # при повторном открытии можно сбросить флаг, чтобы пересчитать ширину
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

        self.reload_characters()
        self.reload_weapons()

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

    # ---------- LOGIC ----------
    def calculate_abyss(self):
        total = sum(f.calculate() for f in self.floors)
        self.total_label.setText(f"Итого: {total} сек")
        self.save_state()

    def save_state(self):
        data = {
            "floors": [{"t1": f.t1.seconds_left, "t2": f.t2.seconds_left} for f in self.floors],
            "teams": [[slot.to_dict() for slot in team] for team in self.teams]
        }
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_state(self):
        if not os.path.exists(STATE_FILE):
            return
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
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

    def reload_characters(self):
        while self.char_grid.count():
            item = self.char_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        if not os.path.exists(ASSETS_CHAR):
            return
        icons = [DraggableIcon(os.path.join(ASSETS_CHAR, f), 72)
                 for f in sorted(os.listdir(ASSETS_CHAR)) if f.lower().endswith(".png")]
        cols = max(1, self.char_area.width() // 90)
        for i, icon in enumerate(icons):
            self.char_grid.addWidget(icon, i // cols, i % cols)

    def reload_weapons(self):
        while self.weapon_grid.count():
            item = self.weapon_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        if not os.path.exists(ASSETS_WEAP):
            return
        icons = [DraggableIcon(os.path.join(ASSETS_WEAP, f), 48)
                 for f in sorted(os.listdir(ASSETS_WEAP)) if f.lower().endswith(".png")]
        cols = max(1, self.weapon_area.width() // 70)
        for i, icon in enumerate(icons):
            self.weapon_grid.addWidget(icon, i // cols, i % cols)

    def load_screenshot(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выбрать скриншот", "", "Images (*.png *.jpg *.jpeg)"
        )
        if not path:
            return
        parser = HoyolabParser(path)
        parser.parse()
        self.reload_characters()
        self.reload_weapons()

    def clear_assets(self):
        for folder in [ASSETS_CHAR, ASSETS_WEAP]:
            shutil.rmtree(folder, ignore_errors=True)
            os.makedirs(folder, exist_ok=True)
        self.reload_characters()
        self.reload_weapons()
        QMessageBox.information(self, "Готово", "Ассеты очищены")

    def reset_run(self):
        # Сбрасываем таймеры до 10:00
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

        # Очищаем слоты команд
        for team in self.teams:
            for slot in team:
                slot.char.clear()
                slot.weapon.clear()

        # Обновляем общий лейбл
        self.total_label.setText("Итого: 0 сек")

        # Сохраняем состояние
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
                "team2": team2_floors
            }
        }

        runs = []
        if os.path.exists(RUNS_FILE):
            try:
                with open(RUNS_FILE, "r", encoding="utf-8") as f:
                    runs = json.load(f)
            except Exception:
                pass

        runs.append(run)

        with open(RUNS_FILE, "w", encoding="utf-8") as f:
            json.dump(runs, f, indent=2, ensure_ascii=False)

        if hasattr(self, "_run_history_window"):
            self._run_history_window.reload()

