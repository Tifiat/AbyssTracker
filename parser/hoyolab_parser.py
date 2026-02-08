import os
import cv2
import numpy as np
import pytesseract


class HoyolabParser:
    def __init__(self, image_path: str):
        self.image_path = image_path
        with open(image_path, "rb") as f:
            data = f.read()

        self.image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

        if self.image is None:
            raise ValueError(f"Не удалось открыть изображение: {image_path}")

        # папки для ассетов
        os.makedirs("assets/characters", exist_ok=True)
        os.makedirs("assets/weapons", exist_ok=True)
        os.makedirs("debug", exist_ok=True)

    def parse(self):
        """
        Главная точка входа.
        Вырезает иконки персонажей и оружия.
        """
        cards = self.extract_character_cards()

        characters = []

        for i, card in enumerate(cards):
            char_icon = self.extract_character_icon(card)
            weapon_icon = self.extract_weapon_icon(card)

            if char_icon is not None:
                cv2.imwrite(f"assets/characters/char_{i}.png", char_icon)

            if weapon_icon is not None:
                cv2.imwrite(f"assets/weapons/weapon_{i}.png", weapon_icon)

            characters.append({
                "name": "",
                "level": "",
                "constellation": "C?"
            })

        return characters

    # ===============================
    # НАРЕЗКА КАРТОЧЕК
    # ===============================
    def extract_character_cards(self):
        img = self.image
        h, w, _ = img.shape

        cards = []

        # ===== НАСТРОЙКИ СЕТКИ =====
        columns = 2
        start_x = 30
        start_y = 218

        card_w = 332
        card_h = 167

        gap_x = 21
        gap_y = 20.5

        max_cards = 100

        cards = []

        for idx in range(max_cards):
            row = idx // columns
            col = idx % columns

            x = start_x + col * (card_w + gap_x)
            y = start_y + round(row * (card_h + gap_y))

            card = self.image[y:y + card_h, x:x + card_w]
            cards.append(card)
            if y + card_h > h:
                break

        # --- ДЕБАГ СЕТКИ ---
        debug = self.image.copy()
        for idx in range(len(cards)):
            row = idx // columns
            col = idx % columns
            x = start_x + col * (card_w + gap_x)
            y = start_y + round(row * (card_h + gap_y))
            cv2.rectangle(debug, (x, y), (x + card_w, y + card_h), (0, 255, 0), 2)

        os.makedirs("debug", exist_ok=True)
        cv2.imwrite("debug/grid_check.png", debug)

        return cards

    # ===============================
    # ПЕРСОНАЖ
    # ===============================
    def extract_character_icon(self, card):
        """
        Вырезает иконку персонажа из карточки
        """
        # левая часть карточки
        x, y, w, h = 9, 11, 148, 148
        icon = card[y:y + h, x:x + w]

        if icon.size == 0:
            return None
        return icon

    # ===============================
    # ОРУЖИЕ
    # ===============================
    def extract_weapon_icon(self, card):
        """
        Вырезает иконку оружия из карточки
        """
        # правая часть карточки
        x, y, w, h = 169, 95, 63, 63
        icon = card[y:y + h, x:x + w]

        if icon.size == 0:
            return None
        return icon




