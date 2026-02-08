from PySide6.QtWidgets import QLabel
from PySide6.QtGui import QPixmap, QDrag
from PySide6.QtCore import Qt, QMimeData


class DraggableIcon(QLabel):
    def __init__(self, image_path, size):
        super().__init__()
        self.image_path = image_path

        pixmap = QPixmap(image_path).scaled(
            size, size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.setPixmap(pixmap)
        self.setFixedSize(size, size)
        self.setCursor(Qt.OpenHandCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            drag = QDrag(self)
            mime = QMimeData()
            mime.setText(self.image_path)
            drag.setMimeData(mime)
            drag.setPixmap(self.pixmap())
            drag.exec(Qt.CopyAction)
