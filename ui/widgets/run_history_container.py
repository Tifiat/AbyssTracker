from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QPainter, QColor
from .run_card import RunCard

class RunHistoryContainer(QWidget):
    def __init__(self):
        super().__init__()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#111"))

        painter.setPen(QColor("#333"))

        cards = [w for w in self.children() if isinstance(w, RunCard)]
        if not cards:
            return

        # собираем левый x каждой колонки
        columns = sorted(set(card.x() for card in cards))
        for x in columns[1:]:  # первая колонка слева не рисуется
            painter.drawLine(x, 0, x, self.height())

    def refresh_lines(self):
        self.update()



