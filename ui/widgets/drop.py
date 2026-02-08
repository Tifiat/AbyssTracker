import os

from PySide6.QtWidgets import QLabel
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt


class DropSlot(QLabel):
    def __init__(self, w, h):
        super().__init__()
        self.image_path = None
        self.setFixedSize(w, h)
        self.setAlignment(Qt.AlignCenter)
        self.setAcceptDrops(True)
        self.clear()

    def clear(self):
        self.image_path = None
        self.setPixmap(QPixmap())
        self.setStyleSheet("border:2px dashed #555; background:#222;")

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def dropEvent(self, event):
        path = event.mimeData().text()
        if os.path.exists(path):
            self.image_path = path
            pixmap = QPixmap(path).scaled(
                self.width() - 4,
                self.height() - 4,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.setPixmap(pixmap)
            self.setStyleSheet("border:2px solid #aaa;")

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self.clear()

    def dropEvent_fake(self, path):
        if not os.path.exists(path):
            return

        self.image_path = path
        pixmap = QPixmap(path).scaled(
            self.width() - 4,
            self.height() - 4,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.setPixmap(pixmap)
        self.setStyleSheet("border:2px solid #aaa;")
