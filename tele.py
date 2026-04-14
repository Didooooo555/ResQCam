# tele.py — отправка alert.txt и всех фото в Telegram
import os, time, urllib.request, json

TG_TOKEN = "8535000501:AAEdba8UPmH3Iuu2Dhag95nVqynoSAUsi6E"
TG_CHAT_ID = "7875346189"
BASE_URL = f"https://api.telegram.org/bot{TG_TOKEN}"
ALERT_FILE = "alerts/alert.txt"


def send_message(text: str):
    """Отправка текста"""
    print("➡️ Отправка текста...")
    data = json.dumps({"chat_id": TG_CHAT_ID, "text": text}).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}/sendMessage", data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        print("Ответ Telegram:", resp.read().decode())


def send_photo(path: str, caption: str = ""):
    """Отправка фото"""
    path = path.strip().replace("\\", "/")
    if not os.path.isabs(path):
        path = os.path.abspath(path).replace("\\", "/")
    if not os.path.exists(path):
        print("⚠️ Фото не найдено:", path)
        return
    print("➡️ Отправка фото:", path)
    os.system(
        f'curl -F "chat_id={TG_CHAT_ID}" -F "caption={caption}" -F "photo=@{path}" {BASE_URL}/sendPhoto'
    )


# ---------- Основной блок ----------
if not os.path.exists(ALERT_FILE):
    print("❌ Файл alert.txt не найден.")
else:
    with open(ALERT_FILE, "r", encoding="utf-8") as f:
        content = f.read().strip()

    send_message(content)

    # Собираем все пути фото
    photos = []
    for line in content.splitlines():
        if line.strip().lower().startswith("фото:"):
            p = line.split(":", 1)[1].strip()
            photos.append(p)

    if not photos:
        print("⚠️ Фото не найдено в alert.txt.")
    else:
        print(f"📸 Найдено {len(photos)} фото.")
        for p in photos:
            time.sleep(1)
            send_photo(p, caption="🚗💥 Обнаружено возможное столкновение!")

    print("✅ Все фото и текст отправлены!")
