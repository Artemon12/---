import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def cv2_add_russian_text(img, text, position, font_size=30, color=(0, 0, 255)):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    try:
        font = ImageFont.truetype("arial.ttf", font_size, encoding="utf-8")
    except:
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size, encoding="utf-8")
        except:
            font = ImageFont.load_default()
            print("Предупреждение: используем шрифт по умолчанию (русский может не отображаться)")

    color_rgb = (color[2], color[1], color[0])  # BGR -> RGB
    draw.text(position, text, font=font, fill=color_rgb)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def classify_contour(contour, hierarchy, idx):
    # Минимальная площадь для отсеивания шума
    if cv2.contourArea(contour) < 500:
        return None

    # Периметр и аппроксимация контура
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    vertices = len(approx)

    # Ограничивающий прямоугольник и отношение сторон
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 1

    # Проверяем, есть ли признак отверстия
    child = hierarchy[0][idx][2]
    has_hole = child != -1

    if has_hole:
        # Если есть отверстие, смотрим на форму внешнего контура
        if 6 <= vertices <= 9 and aspect_ratio < 1.2:
            # Шестигранная форма – скорее гайка (с отверстием)
            return "Гайка"
        else:
            # Круглая или неправильная форма – шайба
            return "Шайба"
    else:
        # Нет отверстия
        if 6 <= vertices <= 9 and aspect_ratio < 1.4:
            return "Гайка"
        elif aspect_ratio > 1.2:
            return "Болт"
        else:
            return None


# Загрузка изображения
image = cv2.imread('124.jpg')
if image is None:
    print("Ошибка загрузки изображения")
    exit()

output = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Размытие и бинаризация
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

kernel = np.ones((3, 3), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# подсчёт
counts = {"Гайка": 0, "Шайба": 0, "Болт": 0}

# Цвета для обводки
colors = {
    "Гайка": (0, 255, 0),   # зелёный
    "Шайба": (255, 0, 0),   # синий
    "Болт": (0, 0, 255)     # красный
}

for i, contour in enumerate(contours):
    # Рассматриваем только внешние контуры
    if hierarchy[0][i][3] != -1:
        continue

    label = classify_contour(contour, hierarchy, i)
    if label is not None:
        counts[label] += 1
        cv2.drawContours(output, [contour], -1, colors[label], 2)

# Формируем текст с результатами
text = f"Гаек {counts['Гайка']}; Шайб {counts['Шайба']}; Болтов {counts['Болт']}"
output = cv2_add_russian_text(output, text, (10, 30), font_size=30, color=(0, 0, 255))

print("Результаты классификации:")
for obj, cnt in counts.items():
    print(f"{obj}: {cnt}")

cv2.imshow("Результат", output)
cv2.waitKey(0)
cv2.destroyAllWindows()