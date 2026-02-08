from PySide6.QtWidgets import QWidget

from ui.widgets.drop import DropSlot


class TeamSlot(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(80, 80)

        self.char = DropSlot(72, 72)
        self.char.setParent(self)
        self.char.move(4, 4)

        # слот артефакта (левый нижний угол)
        self.artifact = DropSlot(28, 28)
        self.artifact.setParent(self.char)
        self.artifact.move(2, 42)
        self.artifact.raise_()

        # слот оружия (правый нижний угол)
        self.weapon = DropSlot(28, 28)
        self.weapon.setParent(self.char)
        self.weapon.move(42, 42)
        self.weapon.raise_()

    def to_dict(self):
        return {
            "char": self.char.image_path,
            "weapon": self.weapon.image_path,
            "artifact": self.artifact.image_path
        }

    def from_dict(self, data):
        if data.get("char"):
            self.char.dropEvent_fake(data["char"])
        if data.get("weapon"):
            self.weapon.dropEvent_fake(data["weapon"])
        if data.get("artifact"):
            self.artifact.dropEvent_fake(data["artifact"])
