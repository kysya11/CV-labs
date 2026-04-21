import cv2
import numpy as np
import sys
import os


def find_screen_with_canny(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Размытие для уменьшения шума
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Автоматический подбор порогов на основе медианы
    v = np.median(blurred)
    low_thresh = int(max(0, (0.7) * v))
    high_thresh = int(min(255, (1.3) * v))

    # Применяем детектор границ Canny
    edges = cv2.Canny(blurred, low_thresh, high_thresh)

    # Морфологическое закрытие для соединения близких границ
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Поиск контуров
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Сортируем контуры по площади
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 5000:
            continue

        # Аппроксимируем контур до четырехугольника
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) == 4:
            return approx.reshape(4, 2)

    return None


def order_points(pts):
    # Упорядочивает точки: верхний-левый, верхний-правый, нижний-правый, нижний-левый

    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def replace_screen(frame, screen_pts, replacement_img):
    # Заменяет область экрана на картинку.

    if screen_pts is None:
        return frame

    screen_pts = order_points(screen_pts)

    # Вычисляем размеры
    width = int(
        max(
            np.linalg.norm(screen_pts[1] - screen_pts[0]),
            np.linalg.norm(screen_pts[2] - screen_pts[3]),
        )
    )
    height = int(
        max(
            np.linalg.norm(screen_pts[3] - screen_pts[0]),
            np.linalg.norm(screen_pts[2] - screen_pts[1]),
        )
    )

    # Изменяем размер картинки
    resized_img = cv2.resize(replacement_img, (width, height))

    # Точки назначения
    dst_pts = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )

    # Гомография
    h_matrix = cv2.getPerspectiveTransform(dst_pts, screen_pts)
    warped = cv2.warpPerspective(
        resized_img, h_matrix, (frame.shape[1], frame.shape[0])
    )

    # Маска
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [screen_pts.astype(np.int32)], 255)

    # Накладываем
    result = frame.copy()
    result[mask == 255] = warped[mask == 255]

    return result


def main():
    # Проверка параметров
    if len(sys.argv) != 3:
        print("Использование: python script.py <путь_к_картинке> <путь_к_видео>")
        print("Пример: python script.py picture.jpg video.mp4")
        sys.exit(1)

    image_path = sys.argv[1]
    video_path = sys.argv[2]

    # Проверяем файлы
    if not os.path.exists(image_path):
        print(f"Ошибка: файл {image_path} не найден")
        sys.exit(1)

    if not os.path.exists(video_path):
        print(f"Ошибка: файл {video_path} не найден")
        sys.exit(1)

    # Загружаем картинку
    replacement_img = cv2.imread(image_path)
    if replacement_img is None:
        print(f"Ошибка: не удалось загрузить {image_path}")
        sys.exit(1)

    # Открываем видео
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Ошибка: не удалось открыть {video_path}")
        sys.exit(1)

    print(f"Обработка видео: {video_path}")
    print(f"Вставка картинки: {image_path}")
    print("Нажмите 'q' для выхода")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Ищем экран на каждом кадре
        screen_points = find_screen_with_canny(frame)

        # Заменяем экран
        result = replace_screen(frame, screen_points, replacement_img)

        # Показываем результат
        cv2.imshow("Result", result)

        # Выход по 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Программа завершена. Обработано кадров: {frame_count}")


if __name__ == "__main__":
    main()
