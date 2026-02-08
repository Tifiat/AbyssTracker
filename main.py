import sys
from PySide6.QtWidgets import QApplication

from ui.main_window import App
def main():
    app = QApplication(sys.argv)
    w = App()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

