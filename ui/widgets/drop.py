import os

from PySide6.QtWidgets import QLabel
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt, QEvent, QTimer


class DropSlot(QLabel):
    def __init__(self, w, h):
        super().__init__()
        self.image_path = None
        self._src_pixmap = None

        self.setFixedSize(w, h)
        self.setAlignment(Qt.AlignCenter)
        self.setAcceptDrops(True)
        self.clear()

    def clear(self):
        self.image_path = None
        self._src_pixmap = None
        self.setPixmap(QPixmap())
        self.setStyleSheet("border:2px dashed #555; background:#222;")

    def _update_pixmap(self):
        if self._src_pixmap is None or self._src_pixmap.isNull():
            self.setPixmap(QPixmap())
            return

        dpr = self.devicePixelRatioF()
        target_w = max(1, int((self.width() - 4) * dpr))
        target_h = max(1, int((self.height() - 4) * dpr))

        pixmap = self._src_pixmap.scaled(
            target_w,
            target_h,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        pixmap.setDevicePixelRatio(dpr)
        self.setPixmap(pixmap)

    def _set_image(self, path):
        if not os.path.exists(path):
            return

        self.image_path = path
        self._src_pixmap = QPixmap(path)
        self._update_pixmap()
        QTimer.singleShot(0, self._update_pixmap)
        self.setStyleSheet("border:2px solid #aaa;")

    def event(self, event):
        if event.type() in (
            QEvent.Type.DevicePixelRatioChange,
            QEvent.Type.ScreenChangeInternal,
            QEvent.Type.Resize,
            QEvent.Type.Show,
        ):
            self._update_pixmap()
        return super().event(event)

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def dropEvent(self, event):
        path = event.mimeData().text()
        self._set_image(path)

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self.clear()

    def dropEvent_fake(self, path):
        self._set_image(path)