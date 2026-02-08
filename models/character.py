from PySide6.QtWidgets import QLabel
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QDrag
from PySide6.QtCore import QMimeData

class CharacterIcon(QLabel):
    def __init__(self, img_path):
        super().__init__()
        self.pixmap = QPixmap(img_path)
        self.setPixmap(self.pixmap.scaled(48, 48, Qt.KeepAspectRatio))
        self.setFixedSize(48, 48)

    def mouseMoveEvent(self, event):
        drag = QDrag(self)
        mime = QMimeData()
        mime.setImageData(self.pixmap.toImage())
        drag.setMimeData(mime)
        drag.setPixmap(self.pixmap)
        drag.exec(Qt.CopyAction)
