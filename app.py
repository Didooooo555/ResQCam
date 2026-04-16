import customtkinter as ctk
import threading
import subprocess
import sys
import os

# ---------- ДАННЫЕ ----------
dpt_stats = [
    "More than 1.3 million people die in road accidents every year.",
    "More than 50 million people suffer injuries.",
    "More than 70% of crashes are not assessed in time.",
    "ResQCam analyzes collisions and survival probability."
]

system_info = [
    "ResQCam system:",
    "• Video collision detection (YOLOv8n)",
    "• IoU / dIoU and object speed calculation",
    "• Speed smoothing for stable analysis",
    "• Automatic event notification pipeline",
    "• Survival analysis support for accidents"
]

why_it_matters = [
    "• Detect road accidents faster from video streams",
    "• Provide clear metrics for severity estimation",
    "• Help responders react faster to critical situations"
]

# ---------- НАСТРОЙКИ ----------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ---------- ОСНОВНОЕ ОКНО ----------
app = ctk.CTk()
app.title("ResQCam — AI Collision Dashboard")
app.geometry("900x660")
app.resizable(False, False)

# ---------- ЦВЕТА ----------
BG = "#070A12"
SHELL = "#0B1020"
PANEL = "#0C1224"
CARD = "#0F1830"

BORDER = "#223055"
BORDER_SOFT = "#1A2443"

TXT = "#EAF0FF"
MUTED = "#95A3C9"
MUTED_2 = "#6F7BA6"

ACCENT = "#00E6FF"
ACCENT_SOFT = "#3A7BFF"
SUCCESS = "#2CFFB6"
ERROR = "#FF5C7A"

R_SHELL = 26
R_PANEL = 22
R_CARD = 18

# ---------- ШРИФТЫ ----------
FONT_TITLE = ctk.CTkFont(size=32, weight="bold")
FONT_16_BOLD = ctk.CTkFont(size=16, weight="bold")
FONT_15_BOLD = ctk.CTkFont(size=15, weight="bold")
FONT_13_BOLD = ctk.CTkFont(size=13, weight="bold")
FONT_12 = ctk.CTkFont(size=12)
FONT_12_BOLD = ctk.CTkFont(size=12, weight="bold")
FONT_11 = ctk.CTkFont(size=11)
FONT_11_BOLD = ctk.CTkFont(size=11, weight="bold")

# ---------- СЛУЖЕБНЫЕ ПЕРЕМЕННЫЕ ----------
status_animation_job = None
status_animation_text = ""
status_animation_colors = [SUCCESS, ACCENT, "#8A7CFF", SUCCESS]
status_animation_index = 0
is_closing = False


# ---------- БЕЗОПАСНЫЕ ФУНКЦИИ ----------
def widget_exists(widget):
    try:
        return widget is not None and widget.winfo_exists()
    except Exception:
        return False


def safe_configure(widget, **kwargs):
    try:
        if widget_exists(widget):
            widget.configure(**kwargs)
    except Exception:
        pass


def safe_after(delay, func):
    try:
        if widget_exists(app) and not is_closing:
            return app.after(delay, func)
    except Exception:
        pass
    return None


# ---------- СТАТУС ----------
def stop_status_animation():
    global status_animation_job
    if status_animation_job is not None:
        try:
            app.after_cancel(status_animation_job)
        except Exception:
            pass
        status_animation_job = None


def animate_status(text):
    global status_animation_job, status_animation_text, status_animation_index

    stop_status_animation()
    status_animation_text = text
    status_animation_index = 0

    safe_configure(status_label, text=text)

    if text == "IDLE":
        safe_configure(status_label, text_color=SUCCESS)
        return

    if text == "ERROR":
        safe_configure(status_label, text_color=ERROR)
        return

    def pulse():
        global status_animation_job, status_animation_index

        if not widget_exists(status_label) or is_closing:
            status_animation_job = None
            return

        try:
            if status_label.cget("text") != status_animation_text:
                status_animation_job = None
                return
        except Exception:
            status_animation_job = None
            return

        color = status_animation_colors[status_animation_index % len(status_animation_colors)]
        status_animation_index += 1
        safe_configure(status_label, text_color=color)

        status_animation_job = safe_after(160, pulse)

    pulse()


# ---------- АНАЛИЗ ----------
def finish_analysis(success=True):
    progress_bar.stop()
    if success:
        animate_status("IDLE")
    else:
        animate_status("ERROR")
    safe_configure(btn_start, state="normal")


def run_analysis():
    safe_configure(btn_start, state="disabled")
    animate_status("RUNNING ANALYSIS...")
    progress_bar.start()

    def task():
        ok = True
        try:
            if not os.path.exists("d.py"):  #!!!!!!!!!!!
                print("Файл didov2.py не найден.")
                ok = False
            else:
                result = subprocess.run([sys.executable, "d.py"], check=False)
                if result.returncode != 0:
                    ok = False
        except Exception as e:
            print("Ошибка запуска didov2.py:", e)
            ok = False
        finally:
            safe_after(0, lambda: finish_analysis(ok))

    threading.Thread(target=task, daemon=True).start()


# ---------- ФОН ----------
root = ctk.CTkFrame(app, fg_color=BG, corner_radius=0)
root.pack(fill="both", expand=True)

glow1 = ctk.CTkFrame(root, fg_color="#071D2A", corner_radius=300, width=420, height=420)
glow1.place(x=-120, y=-120)

glow2 = ctk.CTkFrame(root, fg_color="#120A2A", corner_radius=380, width=520, height=520)
glow2.place(x=520, y=60)

# ---------- ОБЩИЙ КОНТЕЙНЕР ----------
shell = ctk.CTkFrame(root, fg_color=SHELL, corner_radius=R_SHELL, border_width=1, border_color=BORDER)
shell.pack(padx=18, pady=18, fill="both", expand=True)

shell.grid_columnconfigure(0, weight=1)
shell.grid_columnconfigure(1, weight=0)
shell.grid_rowconfigure(1, weight=1)

# ---------- HEADER ----------
header = ctk.CTkFrame(shell, fg_color="transparent")
header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=18, pady=(18, 10))
header.grid_columnconfigure(0, weight=1)

left_head = ctk.CTkFrame(header, fg_color="transparent")
left_head.grid(row=0, column=0, sticky="w")

title_row = ctk.CTkFrame(left_head, fg_color="transparent")
title_row.pack(anchor="w")

ctk.CTkLabel(
    title_row,
    text="ResQCam",
    font=FONT_TITLE,
    text_color=ACCENT
).pack(side="left")

tag = ctk.CTkFrame(
    title_row,
    fg_color="#071C2A",
    corner_radius=999,
    border_width=1,
    border_color="#0A2F45"
)
tag.pack(side="left", padx=12)

ctk.CTkLabel(
    tag,
    text="AI | ДТП | 2026",
    font=FONT_11_BOLD,
    text_color=ACCENT
).pack(padx=12, pady=4)

ctk.CTkLabel(
    left_head,
    text="Collision Detection & Survival Analysis | YOLOv8n | CPU Mode",
    font=FONT_12,
    text_color=MUTED
).pack(anchor="w", pady=(6, 0))

right_head = ctk.CTkFrame(header, fg_color="transparent")
right_head.grid(row=0, column=1, sticky="e")


def pill(parent, text, fg, border, dot_color):
    p = ctk.CTkFrame(parent, fg_color=fg, corner_radius=999, border_width=1, border_color=border)
    row = ctk.CTkFrame(p, fg_color="transparent")
    row.pack(padx=10, pady=5)

    ctk.CTkLabel(row, text="●", font=FONT_12_BOLD, text_color=dot_color).pack(side="left")
    ctk.CTkLabel(row, text=f"  {text}", font=FONT_11_BOLD, text_color=TXT).pack(side="left")

    p.pack(anchor="e", pady=4)


pill(right_head, "STREAM READY", "#0B2016", "#124A2B", SUCCESS)
pill(right_head, "MODEL LOADED", "#101736", "#242A55", ACCENT_SOFT)

# ---------- ЛЕВАЯ ЗОНА ----------
left = ctk.CTkFrame(shell, fg_color=PANEL, corner_radius=R_PANEL, border_width=1, border_color=BORDER_SOFT)
left.grid(row=1, column=0, sticky="nsew", padx=(18, 10), pady=(0, 18))

left.grid_rowconfigure(1, weight=1)
left.grid_columnconfigure(0, weight=1)

sec = ctk.CTkFrame(left, fg_color="transparent")
sec.grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 10))
sec.grid_columnconfigure(0, weight=1)

ctk.CTkLabel(sec, text="Dashboard", font=FONT_15_BOLD, text_color=TXT).grid(row=0, column=0, sticky="w")
ctk.CTkLabel(sec, text="cards | scroll", font=FONT_12, text_color=MUTED_2).grid(row=0, column=1, sticky="e")

scroll = ctk.CTkScrollableFrame(left, fg_color="transparent", corner_radius=16, width=590, height=430)
scroll.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 14))


def card(parent, title, subtitle, lines, icon):
    c = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=R_CARD, border_width=1, border_color=BORDER_SOFT)
    c.pack(fill="x", padx=8, pady=10)

    top = ctk.CTkFrame(c, fg_color="transparent")
    top.pack(fill="x", padx=16, pady=(14, 6))
    top.grid_columnconfigure(0, weight=1)

    ctk.CTkLabel(
        top,
        text=f"{icon}  {title}",
        font=FONT_13_BOLD,
        text_color=ACCENT
    ).grid(row=0, column=0, sticky="w")

    ctk.CTkLabel(
        top,
        text=subtitle,
        font=FONT_11,
        text_color=MUTED_2
    ).grid(row=0, column=1, sticky="e")

    ctk.CTkFrame(c, height=1, fg_color="#1C2748").pack(fill="x", padx=16, pady=(2, 10))

    for ln in lines:
        ctk.CTkLabel(
            c,
            text=ln,
            font=FONT_12,
            text_color=TXT,
            justify="left",
            anchor="w",
            wraplength=520
        ).pack(anchor="w", fill="x", padx=16, pady=3)


card(scroll, "REAL WORLD DATA", "Road safety", dpt_stats, "[DATA]")
card(scroll, "SYSTEM INFO", "Pipeline", system_info, "[INFO]")
card(scroll, "WHY IT MATTERS", "Mission", why_it_matters, "[WHY]")

# ---------- ПРАВАЯ ЗОНА ----------
right = ctk.CTkFrame(shell, fg_color=PANEL, corner_radius=R_PANEL, border_width=1, border_color=BORDER_SOFT)
right.grid(row=1, column=1, sticky="ns", padx=(10, 18), pady=(0, 18))
right.grid_columnconfigure(0, weight=1)

ctk.CTkLabel(right, text="Control", font=FONT_15_BOLD, text_color=TXT).pack(anchor="w", padx=16, pady=(16, 8))

status_label = ctk.CTkLabel(right, text="IDLE", font=FONT_16_BOLD, text_color=SUCCESS)
status_label.pack(pady=10)


class DotSpinner(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, fg_color="transparent")
        self._running = False
        self._job = None
        self._dots = []
        self._index = 0

        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(pady=8)

        for _ in range(3):
            dot = ctk.CTkLabel(row, text="●", font=FONT_16_BOLD, text_color="#2A355A")
            dot.pack(side="left", padx=4)
            self._dots.append(dot)

    def _animate(self):
        if not self._running or not widget_exists(self):
            self._job = None
            return

        for i, dot in enumerate(self._dots):
            color = ACCENT if i == self._index else "#2A355A"
            safe_configure(dot, text_color=color)

        self._index = (self._index + 1) % len(self._dots)
        self._job = safe_after(180, self._animate)

    def start(self):
        if self._running:
            return
        self._running = True
        self._index = 0
        self._animate()

    def stop(self):
        self._running = False
        if self._job is not None:
            try:
                app.after_cancel(self._job)
            except Exception:
                pass
            self._job = None

        for dot in self._dots:
            safe_configure(dot, text_color="#2A355A")


progress_bar = DotSpinner(right)
progress_bar.pack(pady=(4, 12))

btn_start = ctk.CTkButton(
    right,
    text="START ANALYSIS",
    command=run_analysis,
    width=180,
    height=42,
    corner_radius=14,
    fg_color=ACCENT_SOFT,
    hover_color="#2F68D8",
    text_color="white",
    font=FONT_12_BOLD
)
btn_start.pack(pady=20)


def on_close():
    global is_closing
    is_closing = True

    try:
        stop_status_animation()
    except Exception:
        pass

    try:
        progress_bar.stop()
    except Exception:
        pass

    try:
        app.destroy()
    except Exception:
        pass


app.protocol("WM_DELETE_WINDOW", on_close)

# ---------- ЗАПУСК ----------
app.mainloop()