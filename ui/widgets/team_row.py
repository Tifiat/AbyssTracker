import os
from PySide6.QtWidgets import QWidget, QLabel, QHBoxLayout
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap

class TeamRow(QWidget):
    ICON_SIZE = 56

    def __init__(self, team_slots, floor_times, total_time):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setSpacing(2)
        layout.setContentsMargins(0, 0, 0, 0)

        self.char_icons = []
        self.weapon_icons = []
        self.artifact_icons = []
        self.floor_labels = []

        self.base_icon_size = self.ICON_SIZE
        self.base_weapon_size = 22
        self.base_artifact_size = 20

        for slot in team_slots:
            icon_container = QWidget()
            icon_layout = QHBoxLayout(icon_container)
            icon_layout.setContentsMargins(0, 0, 0, 0)

            # ---- персонаж ----
            char_label = QLabel()
            char_label.setFixedSize(self.ICON_SIZE, self.ICON_SIZE)
            char_path = slot.get("char")
            if isinstance(char_path, str) and os.path.exists(char_path):
                pix = QPixmap(char_path)
                char_label.base_pixmap = pix
                char_label.setPixmap(
                    pix.scaled(self.ICON_SIZE, self.ICON_SIZE, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
            else:
                char_label.base_pixmap = None
            icon_layout.addWidget(char_label)
            self.char_icons.append(char_label)

            # ---- оружие ----
            weapon_label = QLabel(char_label)
            weapon_path = slot.get("weapon")
            if isinstance(weapon_path, str) and os.path.exists(weapon_path):
                pix = QPixmap(weapon_path)
                weapon_label.base_pixmap = pix
                weapon_label.setPixmap(
                    pix.scaled(self.base_weapon_size, self.base_weapon_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
            else:
                weapon_label.base_pixmap = None
            self.weapon_icons.append(weapon_label)

            # ---- артефакт ----
            artifact_label = QLabel(char_label)
            artifact_path = slot.get("artifact")
            if isinstance(artifact_path, str) and os.path.exists(artifact_path):
                pix = QPixmap(artifact_path)
                artifact_label.base_pixmap = pix
                artifact_label.setPixmap(
                    pix.scaled(self.base_artifact_size, self.base_artifact_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
            else:
                artifact_label.base_pixmap = None
            self.artifact_icons.append(artifact_label)

            layout.addWidget(icon_container)

        # ---- времена по этажам ----
        for sec in floor_times:
            lbl = QLabel(str(sec))
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setFixedSize(44, self.ICON_SIZE)
            lbl.base_width = 44
            lbl.base_height = self.ICON_SIZE
            lbl.base_font = 15
            lbl.setStyleSheet("QLabel { border:1px solid #555; background:#111; }")
            self.floor_labels.append(lbl)
            layout.addWidget(lbl)

        # ---- сумма команды ----
        total_lbl = QLabel(str(total_time))
        total_lbl.setAlignment(Qt.AlignCenter)
        total_lbl.setFixedSize(50, self.ICON_SIZE)
        total_lbl.base_width = 50
        total_lbl.base_height = self.ICON_SIZE
        total_lbl.base_font = 15
        total_lbl.setStyleSheet("QLabel { border:1px solid #777; font-weight:bold; background:#111; }")
        self.floor_labels.append(total_lbl)
        layout.addWidget(total_lbl)

        # ---- откладываем позиционирование оружия и артефакта ----
        QTimer.singleShot(0, self.update_icon_positions)

    # -------------------
    # масштабируем команду
    # -------------------
    def set_scale(self, factor):
        for i, char_label in enumerate(self.char_icons):
            size = int(self.base_icon_size * factor)
            char_label.setFixedSize(size, size)
            if char_label.base_pixmap:
                char_label.setPixmap(char_label.base_pixmap.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation))

            # оружие
            weapon_label = self.weapon_icons[i]
            w_size = int(self.base_weapon_size * factor)
            weapon_label.setFixedSize(w_size, w_size)
            if weapon_label.base_pixmap:
                weapon_label.setPixmap(weapon_label.base_pixmap.scaled(w_size, w_size, Qt.KeepAspectRatio, Qt.SmoothTransformation))

            # артефакт
            artifact_label = self.artifact_icons[i]
            a_size = int(self.base_artifact_size * factor)
            artifact_label.setFixedSize(a_size, a_size)
            if artifact_label.base_pixmap:
                artifact_label.setPixmap(artifact_label.base_pixmap.scaled(a_size, a_size, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        # числа
        for lbl in self.floor_labels:
            w = int(lbl.base_width * factor)
            h = int(lbl.base_height * factor)
            lbl.setFixedSize(w, h)
            font = lbl.font()
            font.setPointSizeF(lbl.base_font * factor)
            lbl.setFont(font)

        # обновляем позиции оружия и артефактов
        self.update_icon_positions()

    # -------------------
    # позиционирование оружия и артефактов
    # -------------------
    def update_icon_positions(self):
        for i, char_label in enumerate(self.char_icons):
            weapon_label = self.weapon_icons[i]
            a_size = int(self.base_artifact_size * (char_label.width() / self.base_icon_size))
            w_size = int(self.base_weapon_size * (char_label.width() / self.base_icon_size))

            # оружие в правый нижний угол
            weapon_label.move(char_label.width() - w_size, char_label.height() - w_size)
            # артефакт в левый нижний угол
            artifact_label = self.artifact_icons[i]
            artifact_label.move(2, char_label.height() - a_size)
