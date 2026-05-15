import argparse
import math
from pathlib import Path

import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

WINDOW_NAME = "People counter"

selected_points = []
first_frame_for_select = None


def mouse_callback(event, x, y, flags, param):
    global selected_points, first_frame_for_select

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(selected_points) < 2:
            selected_points.append((x, y))

            frame_vis = first_frame_for_select.copy()
            for pt in selected_points:
                cv2.circle(frame_vis, pt, 5, (255, 0, 0), -1)  # синие точки

            if len(selected_points) == 2:
                cv2.line(
                    frame_vis, selected_points[0], selected_points[1], (255, 200, 0), 2
                )  # голубая линия

            cv2.imshow(WINDOW_NAME, frame_vis)


def side_of_line(point, line_p1, line_p2):
    x, y = point
    x1, y1 = line_p1
    x2, y2 = line_p2
    return (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)


def distance_point_to_segment(point, seg_a, seg_b):
    px, py = point
    ax, ay = seg_a
    bx, by = seg_b

    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay

    ab_len_sq = abx * abx + aby * aby
    if ab_len_sq == 0:
        return math.hypot(px - ax, py - ay)

    t = (apx * abx + apy * aby) / ab_len_sq
    t = max(0.0, min(1.0, t))

    closest_x = ax + t * abx
    closest_y = ay + t * aby

    return math.hypot(px - closest_x, py - closest_y)


def draw_info(frame, line_p1, line_p2, count_total):
    cv2.line(frame, line_p1, line_p2, (255, 200, 0), 2)  # голубая линия
    cv2.putText(
        frame,
        f"Count: {count_total}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 165, 255),  # оранжевый текст
        2,
        cv2.LINE_AA,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Подсчёт людей, пересекающих линию с помощью YOLO + DeepSORT"
    )
    parser.add_argument(
        "--video",
        required=True,
        help=r"Путь к видео, например: C:\Users\Kysya\Desktop\учёба\Магистратура\CV\CV-labs\CV5\8.avi",
    )
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="Путь к .pt модели YOLO. По умолчанию: yolo11n.pt",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Порог confidence для детекции. По умолчанию: 0.35",
    )
    parser.add_argument(
        "--output_dir",
        default=r"C:\Users\Kysya\Desktop\учёба\Магистратура\CV\CV-labs\CV5",
        help=r"Папка для сохранения выходного видео. По умолчанию: C:\Users\Kysya\Desktop\учёба\Магистратура\CV\CV-labs\CV5",
    )
    return parser.parse_args()


def main():
    global first_frame_for_select, selected_points

    args = parse_args()
    video_path = Path(args.video)
    output_dir = Path(args.output_dir)

    if not video_path.exists():
        print(f"[ERROR] Видео не найдено: {video_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    output_video_path = output_dir / f"{video_path.stem}_tracked.avi"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("[ERROR] Не удалось открыть видео.")
        return

    ret, first_frame = cap.read()
    if not ret:
        print("[ERROR] Не удалось прочитать первый кадр.")
        cap.release()
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    frame_h, frame_w = first_frame.shape[:2]

    # Подготовка VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (frame_w, frame_h))

    if not writer.isOpened():
        print(f"[ERROR] Не удалось открыть VideoWriter для файла: {output_video_path}")
        cap.release()
        return

    # Выбор линии на первом кадре
    first_frame_for_select = first_frame.copy()
    selected_points = []

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    instruction_frame = first_frame.copy()
    cv2.putText(
        instruction_frame,
        "Postavte 2 tochki",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),  # жёлтый текст инструкций
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        instruction_frame,
        "ESC - vihod",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.imshow(WINDOW_NAME, instruction_frame)

    while True:
        key = cv2.waitKey(20) & 0xFF

        if len(selected_points) == 2:
            break

        if key == 27:
            cap.release()
            writer.release()
            cv2.destroyAllWindows()
            return

    line_p1, line_p2 = selected_points

    # Возврат к началу видео
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Предобученная YOLO-модель
    model = YOLO(args.model)

    # DeepSORT
    tracker = DeepSort(
        max_age=30,
        n_init=2,
        nms_max_overlap=1.0,
        max_cosine_distance=0.2,
        nn_budget=None,
        embedder="mobilenet",
        half=True,
        bgr=True,
        embedder_gpu=True,
    )

    track_memory = {}
    total_count = 0
    last_frame_to_show = None

    diag = math.hypot(frame_w, frame_h)
    crossing_distance_threshold = diag * 0.05

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Детекция только людей (COCO class 0 = person)
        results = model.predict(
            source=frame, conf=args.conf, classes=[0], verbose=False
        )

        detections = []

        for result in results:
            if result.boxes is None:
                continue

            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()

            for box, conf in zip(xyxy, confs):
                x1, y1, x2, y2 = box.astype(int)
                w = x2 - x1
                h = y2 - y1

                if w <= 0 or h <= 0:
                    continue

                detections.append(([x1, y1, w, h], float(conf), "person"))

        tracks = tracker.update_tracks(detections, frame=frame)

        draw_info(frame, line_p1, line_p2, total_count)

        for track in tracks:
            if not track.is_confirmed():
                continue

            # Учитываем только реальную детекцию, без предсказаний трекера
            if track.time_since_update > 0:
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)

            cx = int((x1 + x2) / 2)
            cy = int(y2)
            center_point = (cx, cy)

            current_side = side_of_line(center_point, line_p1, line_p2)

            if track_id not in track_memory:
                track_memory[track_id] = {"last_side": current_side, "counted": False}
            else:
                prev_side = track_memory[track_id]["last_side"]

                changed_side = (prev_side < 0 and current_side > 0) or (
                    prev_side > 0 and current_side < 0
                )

                dist_to_line = distance_point_to_segment(center_point, line_p1, line_p2)

                if (
                    changed_side
                    and not track_memory[track_id]["counted"]
                    and dist_to_line < crossing_distance_threshold
                ):
                    total_count += 1
                    track_memory[track_id]["counted"] = True

                track_memory[track_id]["last_side"] = current_side

            # отрисовка: рамка, точка, ID
            cv2.rectangle(
                frame, (x1, y1), (x2, y2), (0, 255, 100), 2
            )  # светло-зелёная рамка
            cv2.circle(frame, center_point, 4, (0, 0, 255), -1)  # красная точка
            cv2.putText(
                frame,
                f"ID {track_id}",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 100),  # светло-зелёный ID
                2,
                cv2.LINE_AA,
            )

        # Обновим текст после возможного увеличения счётчика
        draw_info(frame, line_p1, line_p2, total_count)

        # Сохраняем кадр в выходное видео
        writer.write(frame)

        last_frame_to_show = frame.copy()
        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    writer.release()

    if last_frame_to_show is not None:
        draw_info(last_frame_to_show, line_p1, line_p2, total_count)
        cv2.putText(
            last_frame_to_show,
            f"Saved to: {output_video_path}",
            (20, frame_h - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),  # жёлтый
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            last_frame_to_show,
            "Video ended. Press any key to close.",
            (20, frame_h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(WINDOW_NAME, last_frame_to_show)
        print(f"Итоговое количество людей, пересекших линию: {total_count}")
        print(f"Выходное видео сохранено: {output_video_path}")
        cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
