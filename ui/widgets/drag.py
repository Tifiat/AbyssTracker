import os  # ДОБАВИТЬ этот импорт
from PySide6.QtWidgets import QLabel, QMessageBox, QApplication  # ДОБАВИТЬ QMessageBox, QApplication
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
			# Существующая логика перетаскивания
			drag = QDrag(self)
			mime = QMimeData()
			mime.setText(self.image_path)
			drag.setMimeData(mime)
			drag.setPixmap(self.pixmap())
			drag.exec(Qt.CopyAction)

		elif event.button() == Qt.RightButton:
			# НОВАЯ логика: удаление файла
			self._handle_right_click(event)

	def _handle_right_click(self, event):
		"""Обработка правого клика - удаление файла"""
		# Проверяем, что файл существует и в папке assets
		if not self.image_path or not os.path.exists(self.image_path):
			return

		# Проверяем, что файл в папке assets (безопасность)
		if not self._is_in_assets_folder(self.image_path):
			QMessageBox.warning(self, "Ошибка",
			                    "Можно удалять только файлы из папки assets!")
			return

		# Проверяем зажат ли Ctrl
		ctrl_pressed = QApplication.keyboardModifiers() & Qt.ControlModifier

		if not ctrl_pressed:
			# Показываем диалог подтверждения
			reply = QMessageBox.question(
				self,
				"Удаление файла",
				f"Удалить файл?\n{os.path.basename(self.image_path)}",
				QMessageBox.Yes | QMessageBox.No,
				QMessageBox.No
			)
			if reply != QMessageBox.Yes:
				return

		# Удаляем файл
		try:
			os.remove(self.image_path)
			# Скрываем иконку сразу
			self.hide()

			# Всегда одно и то же уведомление
			self._notify_parent_to_update_grids()


		except Exception as e:

			QMessageBox.warning(self, "Ошибка",

			                    f"Не удалось удалить файл:\n{str(e)}")

	def _is_in_assets_folder(self, filepath):
		"""Проверяем, что файл находится в папке assets"""
		try:
			abs_path = os.path.abspath(filepath)
			assets_dirs = [
				os.path.abspath("assets/characters"),
				os.path.abspath("assets/weapons")
			]

			# Проверяем, что файл находится в одной из папок assets
			for assets_dir in assets_dirs:
				if abs_path.startswith(assets_dir + os.sep):
					return True
			return False
		except:
			return False

	def _notify_parent_to_update_grids(self):
		"""Найти главное окно App и вызвать БЕЗОПАСНОЕ обновление"""
		widget = self.parent()
		while widget:
			if hasattr(widget, 'safe_update_grids'):
				# Используем безопасное обновление
				from PySide6.QtCore import QTimer
				QTimer.singleShot(50, widget.safe_update_grids)
				break
			elif hasattr(widget, 'update_grids'):
				# Резервный вариант (старый код)
				from PySide6.QtCore import QTimer
				QTimer.singleShot(50, widget.update_grids)
				break
			widget = widget.parent()