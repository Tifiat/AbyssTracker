from PySide6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy, QScrollArea
from PySide6.QtCore import Qt, QTimer
import os, json

from .widgets.flow_layout import FlowLayout
from .widgets.run_card import RunCard
from .widgets.run_history_container import RunHistoryContainer

RUNS_FILE = "runs_history.json"


class ZoomScrollArea(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            event.ignore()  # Ctrl+Wheel обработается в RunHistoryWindow
        else:
            super().wheelEvent(event)


class RunHistoryWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("История забегов")
        self._first_show = True
        self._min_width = None
        self.scale_factor = 1.0  # 1.0 – 2.5

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        self.scroll = ZoomScrollArea(self)
        root.addWidget(self.scroll)

        # ---- Контейнер и layout ----
        self.container = RunHistoryContainer()
        self.flow = FlowLayout(self.container, spacing=8)
        self.container.setLayout(self.flow)
        self.scroll.setWidget(self.container)

        # ---- Масштабируем сразу для первого рендера ----
        self.apply_scale()
        QTimer.singleShot(0, self.apply_initial_scale)

        self.reload()
        self.adjust_to_content()

    def apply_initial_scale(self):
        for i in range(self.flow.count()):
            item = self.flow.itemAt(i)
            if item and item.widget():
                item.widget().set_scale(1.0)

    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()
            step = 1.1

            if delta > 0:
                new_scale = self.scale_factor * step
            elif delta < 0:
                new_scale = self.scale_factor / step
            else:
                return

            # ограничение масштаба
            new_scale = max(1.0, min(2.5, new_scale))

            if abs(new_scale - self.scale_factor) > 1e-5:
                self.scale_factor = new_scale
                self.apply_scale()

            event.accept()
        else:
            super().wheelEvent(event)

    def apply_scale(self):
        for i in range(self.flow.count()):
            item = self.flow.itemAt(i)
            if item and item.widget():
                item.widget().set_scale(self.scale_factor)

        # пересчёт layout и vertical lines
        self.container.adjustSize()
        self.container.updateGeometry()
        self.flow.update()
        self.container.refresh_lines()

    def adjust_to_content(self):
        self.container.adjustSize()
        content_size = self.container.sizeHint()
        screen = self.screen().availableGeometry()
        width = min(content_size.width() + 40, screen.width())
        height = min(content_size.height() + 20, screen.height())

        if self._first_show:
            self._min_width = width
            self.setMinimumWidth(self._min_width)
            self.resize(width, height)
            self.setMinimumHeight(height)
            self._first_show = False
        else:
            self.resize(max(self.width(), self._min_width), height)
            self.setMinimumHeight(height)

    def reload(self):
        while self.flow.count():
            item = self.flow.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not os.path.exists(RUNS_FILE):
            return

        try:
            with open(RUNS_FILE, "r", encoding="utf-8") as f:
                runs = json.load(f)
        except Exception:
            return

        for run in reversed(runs):
            card = RunCard(run, self.delete_run)
            card.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            self.flow.addWidget(card)

        QTimer.singleShot(0, self.adjust_to_content)

    def delete_run(self, card):
        if not os.path.exists(RUNS_FILE):
            return

        with open(RUNS_FILE, "r", encoding="utf-8") as f:
            runs = json.load(f)

        runs.remove(card.run_data)

        with open(RUNS_FILE, "w", encoding="utf-8") as f:
            json.dump(runs, f, indent=2, ensure_ascii=False)

        self.reload()


