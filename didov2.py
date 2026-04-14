# dido_accident_mvp.py 
# Детекция столкновений (YOLOv8n) + визуализация боксов + автозапуск tele.py
# Улучшенная логика: dIoU, гистерезис, статичное ДТП, сглаживание скорости

from ultralytics import YOLO
import cv2
import numpy as np
import math, time, os, subprocess
from collections import deque, defaultdict

# ---------- Настройки ----------
VIDEO_PATH = 'v3.mp4'              # путь к видео (или 0 для веб-камеры)
CONF_THRES = 0.2                   # уверенность модели
FRAME_SKIP = 5                     # обрабатываем каждый 5-й кадр
OUT_WIDTH, OUT_HEIGHT = 960, 540   # размер кадра

# Динамическое столкновение (при движении)
IOU_HIT = 0.02                     # минимальное перекрытие
DIOU_MIN = 0.01                    # минимальный рост IoU между кадрами
MIN_APPROACH_SPEED = 40            # мин. относит. скорость (px/s)

# Статичное ДТП (после столкновения, машины почти стоят)
STATIC_IOU = 0.20                  # если IoU >= 20%
STATIC_GAP_PX = 18                 # или зазор между боксами <= 18px

# Гистерезис (устойчивость события)
K_HOLD = 3                         # сколько подряд кадров должны выполняться условия
ALERT_COOLDOWN = 30                # пауза между тревогами

SAVE_ALERT = True
ALERT_DIR = 'alerts'
ALERT_FILE = 'alerts/alert.txt'

# Сглаживание скорости
EMA_ALPHA = 0.6                    # 0..1 (чем выше, тем более гладко)

# ---------- Вспомогательные ----------
def iou_xyxy(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter <= 0:
        return 0.0
    A = max(1, (a[2]-a[0])) * max(1, (a[3]-a[1]))
    B = max(1, (b[2]-b[0])) * max(1, (b[3]-b[1]))
    return inter / (A + B - inter + 1e-6)

def box_center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)

def l2(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])

def edge_gap(a, b):
    # расстояние между границами двух боксов (0 если касаются или пересекаются)
    dx = max(0, max(a[0] - b[2], b[0] - a[2]))
    dy = max(0, max(a[1] - b[3], b[1] - a[3]))
    return math.hypot(dx, dy)

def px_to_kmh(px_s, px_to_m=0.05):
    return px_s * px_to_m * 3.6

def survival(v, defp):
    # грубая эвристика
    return max(1.0, min(99.0, 100.0 - (0.8 * v + 0.6 * defp)))

# ---------- Подготовка ----------
model = YOLO('yolov8n.pt')
model.to('cpu')

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Не удалось открыть видео")

fps = cap.get(cv2.CAP_PROP_FPS) or 25

if SAVE_ALERT:
    os.makedirs(ALERT_DIR, exist_ok=True)

prev_boxes = []
prev_centers = []
prev_speeds_raw = deque(maxlen=10)    # для оценки средней активности
track_speeds = []                     # EMA по объектам

pair_iou_prev = defaultdict(float)    # (id_i, id_j) -> прошлый IoU
pair_hold = defaultdict(int)          # счётчик кадров с выполненными условиями

cooldown = 0
frame_id = 0
event_happened = False
event_counter = 0

# ---------- Основной цикл ----------
while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame_id += 1
    if frame_id % FRAME_SKIP:
        continue

    frame = cv2.resize(frame, (OUT_WIDTH, OUT_HEIGHT))

    # Детекция
    res = model.predict(frame, conf=CONF_THRES, verbose=False)
    boxes_now = []

    for r in res:
        if not getattr(r, "boxes", None):
            continue
        for b in r.boxes:
            cls_name = model.names[int(b.cls[0])]
            if cls_name in ("car", "truck", "bus"):
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                boxes_now.append([x1, y1, x2, y2])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    centers_now = [box_center(bx) for bx in boxes_now]
    n = len(boxes_now)
    speeds_px = [0.0] * n

    # Оценка скорости по смещению центров (жадное сопоставление)
    if prev_boxes and prev_centers:
        used_prev = set()
        for i, c_now in enumerate(centers_now):
            best = None
            for j, c_prev in enumerate(prev_centers):
                if j in used_prev:
                    continue
                d = l2(c_now, c_prev)
                if best is None or d < best[1]:
                    best = (j, d)
            if best is not None:
                j_best, dmin = best
                used_prev.add(j_best)
                v = dmin * fps / max(1, FRAME_SKIP)
                speeds_px[i] = v
                prev_speeds_raw.append(v)

    # Сгладим скорости (EMA)
    if len(track_speeds) != n:
        track_speeds = speeds_px.copy()
    else:
        for i in range(n):
            track_speeds[i] = EMA_ALPHA * track_speeds[i] + (1 - EMA_ALPHA) * speeds_px[i]

    avg_scene_speed = sum(prev_speeds_raw) / len(prev_speeds_raw) if prev_speeds_raw else 0.0
    is_static_scene = avg_scene_speed < 5.0  # почти нет движения

    # ---------- Анализ столкновений ----------
    event = None
    hit_pair = None

    for i in range(n):
        for j in range(i + 1, n):
            bx_i, bx_j = boxes_now[i], boxes_now[j]

            iou_now = iou_xyxy(bx_i, bx_j)
            gap = edge_gap(bx_i, bx_j)
            v_rel = track_speeds[i] + track_speeds[j]

            key = (i, j)
            diou = max(0.0, iou_now - pair_iou_prev[key])
            pair_iou_prev[key] = iou_now

            # Условие динамического удара
            dyn_hit = (
                iou_now >= IOU_HIT and
                diou >= DIOU_MIN and
                v_rel >= MIN_APPROACH_SPEED
            )

            # Условие статического ДТП
            static_hit = is_static_scene and (
                iou_now >= STATIC_IOU or gap <= STATIC_GAP_PX
            )

            if dyn_hit or static_hit:
                pair_hold[key] += 1
            else:
                pair_hold[key] = max(0, pair_hold[key] - 1)

            if pair_hold[key] >= K_HOLD and cooldown == 0:
                deform = iou_now * 100.0
                base_v = max(v_rel, avg_scene_speed)
                v_kmh = px_to_kmh(base_v)
                surv = survival(v_kmh, deform)
                event = (iou_now, v_rel, v_kmh, deform, surv)
                hit_pair = (i, j)
                break

        if event:
            break

    # ---------- Отрисовка и сохранение ----------
    if event and hit_pair:
        iou, v_rel, v_kmh, deform, surv = event
        i_hit, j_hit = hit_pair

        text1 = "⚠️ Возможное СТОЛКНОВЕНИЕ!"
        text2 = f"IoU: {iou * 100:.1f}%  v_rel: {v_rel:.0f}px/s  ≈ {v_kmh:.1f} км/ч"
        text3 = f"Деформация: {deform:.1f}%  Выживаемость: {surv:.1f}%"

        print(text1, "|", text2, "|", text3)

        # красные рамки на подозрительной паре
        for idx in hit_pair:
            bx = boxes_now[idx]
            cv2.rectangle(frame, (bx[0], bx[1]), (bx[2], bx[3]), (0, 0, 255), 3)

        if SAVE_ALERT:
            fname = f"{ALERT_DIR}/alert_{int(time.time())}_{frame_id}.jpg"
            cv2.imwrite(fname, frame)
            event_counter += 1
            with open(ALERT_FILE, "a", encoding="utf-8") as f:
                f.write(f"{'-'*60}\n")
                f.write(f"Событие #{event_counter}\n")
                f.write(f"{text1}\n{text2}\n{text3}\n")
                f.write(f"Фото: {fname}\n")
                f.write(f"Время: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        cooldown = ALERT_COOLDOWN
        event_happened = True

        cv2.putText(frame, text1, (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    else:
        cv2.putText(frame, "Status: OK", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (40, 180, 40), 2)

    if cooldown > 0:
        cooldown -= 1

    prev_boxes, prev_centers = boxes_now, centers_now

    cv2.imshow("Accident Detection (main)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ---------- После завершения ----------
if event_happened:
    print("\n📤 Завершено. Запуск отправки уведомления в Telegram...")
    try:
        subprocess.run(
            [
                r"C:\Users\Int_1\Desktop\ResQCam\venv310\Scripts\python.exe",
                r"C:\Users\Int_1\Desktop\ResQCam\tele.py"
            ],
            check=False
        )
    except Exception as e:
        print("Ошибка при запуске Telegram-скрипта:", e)

else:
    print("\n🚗 Анализ завершён — столкновений не обнаружено.")
