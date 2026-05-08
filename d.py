# MVP
from ultralytics import YOLO
import cv2
import numpy as np
import math
import time
import os
import subprocess
from collections import defaultdict, deque

# =========================
# НАСТРОЙКИ
# =========================
VIDEO_PATH = "v3.mp4"       # путь к видео или 0 для камеры
MODEL_PATH = "yolov8n.pt"

OUT_WIDTH, OUT_HEIGHT = 960, 540
CONF_THRES = 0.30
FRAME_SKIP = 2

# COCO classes:
# car=2, motorcycle=3, bus=5, truck=7
VEHICLE_CLASSES = {2, 5, 7}

TRACK_HISTORY = 12
MIN_TRACK_LEN = 3

# Логика ДТП
CONTACT_IOU = 0.03
CONTACT_GAP = 6
MIN_APPROACH_PX = 28
MIN_IMPACT_DROP = 0.28
PAIR_SCORE_HIT = 4.2
PAIR_SCORE_DECAY = 0.40
ALERT_COOLDOWN_FRAMES = 50

# Перевод px/s -> км/ч (примерная калибровка)
PX_TO_METER = 0.045

SAVE_ALERT = True
ALERT_DIR = "alerts"
ALERT_FILE = os.path.join(ALERT_DIR, "alert.txt")

TELEGRAM_PYTHON = r"C:\Users\Int_1\Desktop\ResQCam\venv310\Scripts\python.exe"
TELEGRAM_SCRIPT = r"C:\Users\Int_1\Desktop\ResQCam\tele.py"

SHOW_WINDOW = True

# Фильтры
MIN_BOX_AREA = 1400
MIN_EVENT_KMH = 10.0
SUSPECT_LINE_SCORE = 2.5

# ASCII-текст для кадра, чтобы не было ??????
IMPACT_TYPE_ASCII = {
    "лобовое": "LOBOVOE",
    "боковое": "BOKOVOE",
    "сзади": "SZADI",
    "попутное": "POPUTNOE",
    "неопределено": "UNKNOWN",
}


# =========================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =========================
def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def iou_xyxy(a, b):
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0

    area_a = max(1, (a[2] - a[0])) * max(1, (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0])) * max(1, (b[3] - b[1]))
    return inter / (area_a + area_b - inter + 1e-6)


def edge_gap(a, b):
    dx = max(0, max(a[0] - b[2], b[0] - a[2]))
    dy = max(0, max(a[1] - b[3], b[1] - a[3]))
    return math.hypot(dx, dy)


def box_center(box):
    return np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0], dtype=np.float32)


def box_area(box):
    return max(1, box[2] - box[0]) * max(1, box[3] - box[1])


def l2(a, b):
    return float(np.linalg.norm(a - b))


def unit_vector(v):
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return np.array([0.0, 0.0], dtype=np.float32)
    return v / n


def angle_deg_between(v1, v2):
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0
    cosv = float(np.dot(v1, v2) / (n1 * n2 + 1e-6))
    cosv = clamp(cosv, -1.0, 1.0)
    return math.degrees(math.acos(cosv))


def speed_px_per_s(history, fps, frame_skip):
    if len(history) < 2:
        return 0.0, np.array([0.0, 0.0], dtype=np.float32)

    p1 = history[-2]
    p2 = history[-1]
    dt = frame_skip / max(fps, 1e-6)
    vec = (p2 - p1) / max(dt, 1e-6)
    speed = float(np.linalg.norm(vec))
    return speed, vec


def avg_speed_px_per_s(history, fps, frame_skip, k=4):
    if len(history) < 2:
        return 0.0

    pts = list(history)[-k:]
    if len(pts) < 2:
        return 0.0

    total = 0.0
    count = 0
    dt = frame_skip / max(fps, 1e-6)

    for i in range(1, len(pts)):
        total += l2(pts[i], pts[i - 1]) / max(dt, 1e-6)
        count += 1

    return total / max(count, 1)


def approach_speed(c1, v1, c2, v2):
    """
    Положительное значение => объекты сближаются
    """
    d = c2 - c1
    norm = float(np.linalg.norm(d))
    if norm < 1e-6:
        return 0.0

    n = d / norm
    rel_v = v1 - v2
    appr = float(np.dot(rel_v, n))
    return appr


def px_to_kmh(px_s, px_to_m=PX_TO_METER):
    return px_s * px_to_m * 3.6


def detect_impact_type(c1, v1, c2, v2):
    """
    Более строгая классификация типа удара.
    """
    sp1 = float(np.linalg.norm(v1))
    sp2 = float(np.linalg.norm(v2))

    if sp1 < 1e-6 or sp2 < 1e-6:
        return "неопределено"

    angle = angle_deg_between(v1, v2)

    # очень близко к встречному движению
    if angle >= 160:
        return "лобовое"

    # почти в одном направлении
    if angle <= 30:
        direction = c2 - c1
        direction_u = unit_vector(direction)
        lead1 = float(np.dot(unit_vector(v1), direction_u))
        lead2 = float(np.dot(unit_vector(v2), -direction_u))

        if lead1 > 0.45 or lead2 > 0.45:
            return "сзади"
        return "попутное"

    # средние углы считаем боковым
    if 45 <= angle < 160:
        return "боковое"

    return "неопределено"


def severity_score(approach_px, overlap_iou, gap_px, drop_ratio, size_factor, impact_type):
    """
    Условная тяжесть удара 0..100
    """
    s_approach = clamp(approach_px / 70.0, 0.0, 1.0) * 40.0
    s_overlap = clamp(overlap_iou / 0.20, 0.0, 1.0) * 18.0
    s_gap = clamp((12.0 - gap_px) / 12.0, 0.0, 1.0) * 10.0
    s_drop = clamp(drop_ratio / 0.65, 0.0, 1.0) * 20.0
    s_size = clamp(size_factor, 0.0, 1.0) * 6.0

    impact_bonus = 0.0
    if impact_type == "лобовое":
        impact_bonus = 10.0
    elif impact_type == "боковое":
        impact_bonus = 7.0
    elif impact_type == "сзади":
        impact_bonus = 4.0

    total = s_approach + s_overlap + s_gap + s_drop + s_size + impact_bonus
    return clamp(total, 0.0, 100.0)


def survival_from_severity(sev, impact_type):
    """
    Условная оценка выживаемости.
    """
    base = 99.0 - sev * 0.76

    if impact_type == "лобовое":
        base -= 9.0
    elif impact_type == "боковое":
        base -= 5.0
    elif impact_type == "сзади":
        base -= 2.0

    return clamp(base, 2.0, 99.0)


def draw_box(frame, box, color, label=None, thickness=2):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    if label:
        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )


def save_alert(frame, lines):
    if not SAVE_ALERT:
        return None

    os.makedirs(ALERT_DIR, exist_ok=True)
    fname = os.path.join(ALERT_DIR, f"alert_{int(time.time())}_{frame_id}.jpg")
    cv2.imwrite(fname, frame)

    with open(ALERT_FILE, "a", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
        f.write("\n")

    return fname.replace(os.sep, "/")


def build_event_lines(event_num, event_data):
    iou_percent = event_data["iou"] * 100.0
    v_rel = event_data["appr_px"]
    v_kmh = event_data["appr_kmh"]
    deform = iou_percent
    survival = event_data["survival"]

    lines = [
        f"Событие #{event_num}",
        "⚠️ Возможное СТОЛКНОВЕНИЕ!",
        f"Тип: {event_data['impact_type']}",
        f"IoU: {iou_percent:.1f}%  v_rel: {v_rel:.0f}px/s  ≈ {v_kmh:.1f} км/ч",
        f"Деформация: {deform:.1f}%  Выживаемость: {survival:.1f}%",
        f"Фото: alerts/alert_{event_data['save_stub']}.jpg",
        f"Время: {event_data['event_time']}",
    ]
    return lines


# =========================
# ЗАПУСК
# =========================
model = YOLO(MODEL_PATH)
model.to("cpu")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Не удалось открыть видео/камеру")

fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 1:
    fps = 25.0

if SAVE_ALERT:
    os.makedirs(ALERT_DIR, exist_ok=True)

# История по track_id
track_centers = defaultdict(lambda: deque(maxlen=TRACK_HISTORY))
track_speeds = {}
track_vectors = {}

# Состояние пар
pair_score = defaultdict(float)
pair_last_iou = defaultdict(float)
pair_last_gap = defaultdict(lambda: 9999.0)
pair_last_alert_frame = defaultdict(lambda: -999999)

frame_id = 0
event_happened = False
cooldown = 0
event_counter = 0

print("Старт анализа... Нажми Q для выхода.")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame_id += 1
    if frame_id % FRAME_SKIP != 0:
        continue

    frame = cv2.resize(frame, (OUT_WIDTH, OUT_HEIGHT))

    results = model.track(
        source=frame,
        conf=CONF_THRES,
        persist=True,
        verbose=False
    )

    current_tracks = {}

    if results and len(results) > 0:
        r = results[0]
        boxes = r.boxes

        if boxes is not None and boxes.id is not None:
            ids = boxes.id.int().cpu().tolist()
            xyxy = boxes.xyxy.cpu().tolist()
            cls = boxes.cls.int().cpu().tolist()
            confs = boxes.conf.cpu().tolist()

            for tid, box, c, cf in zip(ids, xyxy, cls, confs):
                if c not in VEHICLE_CLASSES:
                    continue

                box = list(map(int, box))
                if box_area(box) < MIN_BOX_AREA:
                    continue

                current_tracks[tid] = {
                    "box": box,
                    "cls": c,
                    "conf": float(cf),
                    "center": box_center(box),
                    "area": box_area(box),
                }

    for tid, data in current_tracks.items():
        track_centers[tid].append(data["center"])

        spd, vec = speed_px_per_s(track_centers[tid], fps, FRAME_SKIP)
        track_speeds[tid] = spd
        track_vectors[tid] = vec

    # обычные боксы
    for tid, data in current_tracks.items():
        spd_px = track_speeds.get(tid, 0.0)
        spd_kmh = px_to_kmh(spd_px)
        draw_box(
            frame,
            data["box"],
            (0, 255, 0),
            label=f"ID {tid} | {spd_kmh:.0f} km/h"
        )

    event_data = None
    track_ids = list(current_tracks.keys())

    for a in range(len(track_ids)):
        for b in range(a + 1, len(track_ids)):
            id1 = track_ids[a]
            id2 = track_ids[b]

            d1 = current_tracks[id1]
            d2 = current_tracks[id2]

            if len(track_centers[id1]) < MIN_TRACK_LEN or len(track_centers[id2]) < MIN_TRACK_LEN:
                continue

            box1, box2 = d1["box"], d2["box"]
            c1, c2 = d1["center"], d2["center"]
            v1 = track_vectors.get(id1, np.array([0.0, 0.0], dtype=np.float32))
            v2 = track_vectors.get(id2, np.array([0.0, 0.0], dtype=np.float32))

            iou_now = iou_xyxy(box1, box2)
            gap_now = edge_gap(box1, box2)
            appr = approach_speed(c1, v1, c2, v2)

            avg1 = avg_speed_px_per_s(track_centers[id1], fps, FRAME_SKIP, k=5)
            avg2 = avg_speed_px_per_s(track_centers[id2], fps, FRAME_SKIP, k=5)
            now1 = track_speeds.get(id1, 0.0)
            now2 = track_speeds.get(id2, 0.0)

            before_speed = max(avg1 + avg2, 1e-6)
            current_speed = now1 + now2
            drop_ratio = clamp((before_speed - current_speed) / before_speed, 0.0, 1.0)

            size_factor = min(d1["area"], d2["area"]) / max(max(d1["area"], d2["area"]), 1.0)

            key = tuple(sorted((id1, id2)))

            # жесткие отсечки ложных срабатываний
            if px_to_kmh(appr) < MIN_EVENT_KMH:
                pair_score[key] = max(0.0, pair_score[key] - PAIR_SCORE_DECAY)
                pair_last_iou[key] = iou_now
                pair_last_gap[key] = gap_now
                continue

            if iou_now < CONTACT_IOU and gap_now > CONTACT_GAP:
                pair_score[key] = max(0.0, pair_score[key] - PAIR_SCORE_DECAY)
                pair_last_iou[key] = iou_now
                pair_last_gap[key] = gap_now
                continue

            if appr < MIN_APPROACH_PX:
                pair_score[key] = max(0.0, pair_score[key] - PAIR_SCORE_DECAY)
                pair_last_iou[key] = iou_now
                pair_last_gap[key] = gap_now
                continue

            impact_drop = drop_ratio >= MIN_IMPACT_DROP
            diou = max(0.0, iou_now - pair_last_iou[key])

            add_score = 0.0

            add_score += clamp(appr / 55.0, 0.0, 2.0)

            if iou_now >= CONTACT_IOU:
                add_score += 1.2

            if gap_now <= CONTACT_GAP:
                add_score += 0.8

            if diou > 0.004:
                add_score += clamp(diou * 50.0, 0.0, 1.0)

            if impact_drop:
                add_score += 1.5

            # если контакта почти нет и нет падения скорости, сильно режем
            if iou_now < CONTACT_IOU and not impact_drop:
                add_score *= 0.4

            pair_score[key] = max(0.0, pair_score[key] - PAIR_SCORE_DECAY) + add_score

            pair_last_iou[key] = iou_now
            pair_last_gap[key] = gap_now

            impact_type = detect_impact_type(c1, v1, c2, v2)
            sev = severity_score(
                approach_px=appr,
                overlap_iou=iou_now,
                gap_px=gap_now,
                drop_ratio=drop_ratio,
                size_factor=size_factor,
                impact_type=impact_type
            )
            surv = survival_from_severity(sev, impact_type)

            if pair_score[key] >= SUSPECT_LINE_SCORE:
                cv2.line(
                    frame,
                    tuple(c1.astype(int)),
                    tuple(c2.astype(int)),
                    (0, 165, 255),
                    2
                )

            if (
                pair_score[key] >= PAIR_SCORE_HIT
                and cooldown == 0
                and (frame_id - pair_last_alert_frame[key]) > ALERT_COOLDOWN_FRAMES
            ):
                event_time = time.strftime('%Y-%m-%d %H:%M:%S')
                save_stub = f"{int(time.time())}_{frame_id}"

                event_data = {
                    "pair": key,
                    "box1": box1,
                    "box2": box2,
                    "appr_px": appr,
                    "appr_kmh": px_to_kmh(appr),
                    "iou": iou_now,
                    "gap": gap_now,
                    "drop_ratio": drop_ratio,
                    "severity": sev,
                    "survival": surv,
                    "impact_type": impact_type,
                    "event_time": event_time,
                    "save_stub": save_stub,
                }
                pair_last_alert_frame[key] = frame_id
                break

        if event_data:
            break

    if event_data:
        event_happened = True
        event_counter += 1
        cooldown = ALERT_COOLDOWN_FRAMES

        box1 = event_data["box1"]
        box2 = event_data["box2"]
        impact_type = event_data["impact_type"]
        impact_type_ascii = IMPACT_TYPE_ASCII.get(impact_type, "UNKNOWN")

        draw_box(frame, box1, (0, 0, 255), thickness=3)
        draw_box(frame, box2, (0, 0, 255), thickness=3)

        # Только ASCII на кадре
        t1 = "POSSIBLE ACCIDENT!"
        t2 = f"TYPE: {impact_type_ascii}"
        t3 = f"IoU: {event_data['iou']*100:.1f}% | Vrel: {event_data['appr_px']:.0f}px/s | {event_data['appr_kmh']:.1f} km/h"
        t4 = f"Deform: {event_data['iou']*100:.1f}% | Survival: {event_data['survival']:.1f}%"

        cv2.putText(frame, t1, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.90, (0, 0, 255), 3)
        cv2.putText(frame, t2, (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, t3, (20, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
        cv2.putText(frame, t4, (20, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2)

        # сохраняем уже подписанный кадр
        real_fname = os.path.join(ALERT_DIR, f"alert_{event_data['save_stub']}.jpg")
        cv2.imwrite(real_fname, frame)

        lines = [
            f"Событие #{event_counter}",
            "⚠️ Возможное СТОЛКНОВЕНИЕ!",
            f"Тип: {impact_type}",
            f"IoU: {event_data['iou']*100:.1f}%  v_rel: {event_data['appr_px']:.0f}px/s  ≈ {event_data['appr_kmh']:.1f} км/ч",
            f"Деформация: {event_data['iou']*100:.1f}%  Выживаемость: {event_data['survival']:.1f}%",
            f"Фото: {real_fname.replace(os.sep, '/')}",
            f"Время: {event_data['event_time']}",
        ]

        with open(ALERT_FILE, "a", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
            f.write("\n")

        for line in lines:
            print(line)
        print()

    else:
        cv2.putText(frame, "Status: monitoring", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 220, 50), 2)

    if cooldown > 0:
        cooldown -= 1

    if SHOW_WINDOW:
        cv2.imshow("Accident Detection Improved", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

if event_happened:
    print("📤 Анализ завершен. Есть подозрительные ДТП. Запуск Telegram уведомления...")
    try:
        subprocess.run([TELEGRAM_PYTHON, TELEGRAM_SCRIPT], check=False)
    except Exception as e:
        print("Ошибка запуска Telegram-скрипта:", e)
else:
    print("🚗 Анализ завершен — ДТП не обнаружено.")