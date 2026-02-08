from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy
from PySide6.QtCore import Qt
from .team_row import TeamRow

class RunCard(QWidget):
    def __init__(self, run_data, delete_callback):
        super().__init__()
        self.run_data = run_data
        self.delete_callback = delete_callback

        outer = QVBoxLayout(self)
        outer.setContentsMargins(2, 2, 2, 2)
        outer.setSpacing(2)

        main = QHBoxLayout()
        main.setSpacing(4)

        # ---- команды ----
        left = QVBoxLayout()
        left.setSpacing(1)

        floors = run_data.get("floors", {})
        teams = run_data.get("teams", {})

        t1 = floors.get("team1", [])
        t2 = floors.get("team2", [])

        t1_sum = sum(t1)
        t2_sum = sum(t2)
        total = t1_sum + t2_sum

        self.team_rows = []
        row1 = TeamRow(teams.get("team1", []), t1, t1_sum)
        row2 = TeamRow(teams.get("team2", []), t2, t2_sum)
        self.team_rows.extend([row1, row2])

        left.addWidget(row1)
        left.addWidget(row2)
        main.addLayout(left)

        # ---- таймер и крестик ----
        right = QVBoxLayout()
        right.setSpacing(2)

        close = QLabel("✕")
        close.setFixedSize(16, 16)
        close.setAlignment(Qt.AlignCenter)
        close.setStyleSheet("""
            QLabel { color:#a66; font-weight:bold; }
            QLabel:hover { color:red; }
        """)
        close.mousePressEvent = lambda e: self.delete_callback(self)

        total_lbl = QLabel(str(total))
        total_lbl.setAlignment(Qt.AlignCenter)
        total_lbl.setFixedWidth(56)
        total_lbl.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        total_lbl.setStyleSheet("""
            QLabel {
                border:1px solid #777;
                background:#111;
                font-size:15px;
                font-weight:bold;
            }
        """)

        self.total_label = total_lbl

        right.addWidget(close, alignment=Qt.AlignRight)
        right.addWidget(total_lbl)
        main.addLayout(right)

        outer.addLayout(main)

        # ---- горизонтальная линия ----
        line = QWidget()
        line.setFixedHeight(1)
        line.setStyleSheet("background:#444;")
        outer.addWidget(line)

    # -------------------
    # масштабируем карточку
    # -------------------
    def set_scale(self, factor):
        for row in self.team_rows:
            row.set_scale(factor)

        # масштабируем финальный таймер
        lbl = self.total_label
        lbl.setFixedWidth(int(56 * factor))
        font = lbl.font()
        font.setPointSizeF(15 * factor)
        lbl.setFont(font)
