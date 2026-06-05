#!/usr/bin/env python3
"""
VK-бот для удалённого управления Claude Code (личное использование).

Логика:
  /start         -> новая сессия, показывает папки в /home/user, ждёт путь
  <путь>         -> запоминает рабочую папку, "теперь можете написать первое сообщение"
  <текст>        -> уходит в Claude Code; каждое событие (текст / тул) = отдельное сообщение в чат
  /stop          -> убивает текущий ответ ИИ сразу (сессия сохраняется на диске)
  <текст во время работы> -> прерывает текущий ответ и сразу уходит в ИИ как новое сообщение

Доступ только для одного VK user id (allowlist по from_id). Чужие — молча игнорируются.
"""
import os
import json
import time
import random
import threading
import subprocess

import requests


def load_env(path):
    """Простой парсер .env: KEY=VALUE построчно (без зависимостей)."""
    env = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            env[key.strip()] = value.strip()
    return env


# --- конфигурация из .env ---
_env = load_env(os.path.join(os.path.dirname(__file__), ".env"))
TOKEN = _env["VK_TOKEN"]
GROUP_ID = int(_env["VK_GROUP_ID"])
ALLOWED_USER_ID = int(_env["ALLOWED_USER_ID"])

API = "https://api.vk.com/method/"
V = "5.199"                       # актуальная стабильная версия API
HOME = "/home/user"              # корень, от которого показываем папки при /start
MSG_LIMIT = 4000                 # запас от лимита VK в 4096 символов

session = requests.Session()


# ============================================================
#  Низкоуровневые вызовы VK API
# ============================================================
def api(method, **params):
    """Вызов метода VK API с обработкой rate limit (код 6)."""
    params.update(access_token=TOKEN, v=V)
    for attempt in range(5):
        r = session.post(API + method, data=params, timeout=30).json()
        if "error" in r:
            code = r["error"].get("error_code")
            if code == 6:                       # too many requests — backoff и повтор
                time.sleep(0.4 * (attempt + 1))
                continue
            raise RuntimeError(r["error"])
        return r["response"]
    raise RuntimeError("rate limited")


def split_text(text, limit=MSG_LIMIT):
    """Разбивает длинный текст на куски <= limit, по возможности по строкам."""
    parts, cur = [], ""
    for line in text.split("\n"):
        # очень длинная строка без переносов — режем жёстко
        while len(line) > limit:
            parts.append(line[:limit])
            line = line[limit:]
        if len(cur) + len(line) + 1 > limit:
            if cur:
                parts.append(cur)
            cur = line
        else:
            cur = cur + "\n" + line if cur else line
    if cur:
        parts.append(cur)
    return parts


def send(peer_id, text):
    """Отправляет текстовое сообщение пользователю (с разбивкой длинных)."""
    if not text or not text.strip():
        return
    for chunk in split_text(text):
        api(
            "messages.send",
            peer_id=peer_id,
            random_id=random.getrandbits(31),   # защита VK от дублей
            message=chunk,
        )


# ============================================================
#  Состояние единственной сессии (пользователь один)
# ============================================================
state = {
    "phase": "idle",        # idle | await_folder | ready | running
    "cwd": None,            # рабочая папка для claude
    "session_id": None,     # id сессии claude для --resume
    "proc": None,           # текущий процесс claude (Popen) для kill
    "worker": None,         # поток, читающий поток claude
}
state_lock = threading.Lock()


# ============================================================
#  Запуск и стриминг Claude Code
# ============================================================
def kill_current_proc():
    """Убивает текущий процесс claude, если он жив. Сессия остаётся на диске."""
    proc = state.get("proc")
    if proc and proc.poll() is None:
        proc.terminate()                     # сначала мягко (SIGTERM)
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()                      # затем жёстко (SIGKILL)
    state["proc"] = None


def build_cmd(user_text):
    """Собирает аргументы запуска claude. С --resume, если сессия уже есть."""
    cmd = [
        "claude", "-p", user_text,
        "--output-format", "stream-json",
        "--verbose",                         # обязателен для stream-json
        "--dangerously-skip-permissions",
        "--model", "opus",
    ]
    if state["session_id"]:
        cmd += ["--resume", state["session_id"]]
    return cmd


def stream_claude(peer_id, user_text):
    """
    Запускает claude и построчно читает stream-json,
    отправляя каждое событие в чат отдельным сообщением.
    Выполняется в отдельном потоке.
    """
    cmd = build_cmd(user_text)
    proc = subprocess.Popen(
        cmd,
        cwd=state["cwd"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,                           # построчная буферизация
    )
    state["proc"] = proc

    try:
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue

            etype = ev.get("type")

            if etype == "system":
                # init-событие: запоминаем session_id для будущих --resume
                if ev.get("session_id"):
                    state["session_id"] = ev["session_id"]

            elif etype == "assistant":
                # текст ИИ -> как есть; вызов тула -> только название
                for block in ev.get("message", {}).get("content", []):
                    btype = block.get("type")
                    if btype == "text":
                        send(peer_id, block.get("text", ""))
                    elif btype == "tool_use":
                        send(peer_id, f"🔧 {block.get('name')}")

            elif etype == "result":
                # turn завершён
                send(peer_id, "✅ Готово")

    finally:
        # процесс мог быть убит через /stop — это нормально
        if proc.poll() is None:
            proc.wait()
        with state_lock:
            # снимаем флаг работы, только если это всё ещё наш процесс
            if state["proc"] is proc:
                state["proc"] = None
                if state["phase"] == "running":
                    state["phase"] = "ready"


def start_claude(peer_id, user_text):
    """Запускает обработку сообщения ИИ в отдельном потоке."""
    state["phase"] = "running"
    worker = threading.Thread(
        target=stream_claude, args=(peer_id, user_text), daemon=True
    )
    state["worker"] = worker
    worker.start()


# ============================================================
#  Обработка команд и сообщений
# ============================================================
def list_folders():
    """Возвращает текст со списком папок в /home/user."""
    try:
        dirs = sorted(
            d for d in os.listdir(HOME)
            if os.path.isdir(os.path.join(HOME, d)) and not d.startswith(".")
        )
    except OSError as e:
        return f"Не удалось прочитать {HOME}: {e}"
    if not dirs:
        return f"В {HOME} нет папок."
    listing = "\n".join(f"  • {d}" for d in dirs)
    return f"Папки в {HOME}:\n{listing}"


def _skill_description(skill_dir):
    """Достаёт строку description из frontmatter SKILL.md, если есть."""
    skill_md = os.path.join(skill_dir, "SKILL.md")
    if not os.path.isfile(skill_md):
        return ""
    try:
        with open(skill_md, encoding="utf-8") as f:
            for line in f:
                if line.startswith("description:"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return ""


def _collect_skills(skills_root):
    """Возвращает список (имя, описание) скиллов из каталога skills_root."""
    result = []
    if not os.path.isdir(skills_root):
        return result
    for name in sorted(os.listdir(skills_root)):
        path = os.path.join(skills_root, name)
        if os.path.isdir(path):
            result.append((name, _skill_description(path)))
    return result


def list_skills():
    """Текст со списком глобальных и проектных (из текущей cwd) скиллов."""
    blocks = []

    glob_root = os.path.expanduser("~/.claude/skills")
    glob = _collect_skills(glob_root)
    if glob:
        lines = "\n".join(
            f"  • /{name}" + (f" — {desc[:80]}" if desc else "")
            for name, desc in glob
        )
        blocks.append(f"🌐 Глобальные скиллы:\n{lines}")

    # проектные — только если выбрана рабочая папка
    if state.get("cwd"):
        proj_root = os.path.join(state["cwd"], ".claude", "skills")
        proj = _collect_skills(proj_root)
        if proj:
            lines = "\n".join(
                f"  • /{name}" + (f" — {desc[:80]}" if desc else "")
                for name, desc in proj
            )
            blocks.append(f"📁 Скиллы проекта ({state['cwd']}):\n{lines}")

    if not blocks:
        return "Скиллы не найдены."
    return "\n\n".join(blocks)


def handle_message(peer_id, text):
    """Главный роутер: команды и обычные сообщения."""
    text = text.strip()

    # --- /start: новая сессия ---
    if text == "/start":
        with state_lock:
            kill_current_proc()
            state.update(phase="await_folder", cwd=None, session_id=None)
        send(peer_id, "🆕 Новая сессия.\n\n" + list_folders())
        send(peer_id, "Напишите название папки (например: meet) или полный путь, начинающийся с /home")
        return

    # --- /stop: остановить текущий ответ ИИ ---
    if text == "/stop":
        with state_lock:
            if state["proc"] and state["proc"].poll() is None:
                kill_current_proc()
                if state["session_id"]:
                    state["phase"] = "ready"
                send(peer_id, "⏹ Остановлено.")
            else:
                send(peer_id, "Сейчас ничего не выполняется.")
        return

    # --- /skills: показать доступные скиллы ---
    if text == "/skills":
        send(peer_id, list_skills())
        return

    # --- ожидаем путь к рабочей папке ---
    if state["phase"] == "await_folder":
        # путь, начинающийся с /home — считаем полным; иначе относительный от /home/user
        path = text if text.startswith("/home") else os.path.join(HOME, text)
        if os.path.isdir(path):
            with state_lock:
                state.update(phase="ready", cwd=path)
            send(peer_id, f"📂 Рабочая папка: {path}\n\nТеперь можете написать первое сообщение.")
        else:
            # папки нет — предложить создать
            send(peer_id, f"Папки {path} нет. Создать её? Напишите 'да' для создания или другой путь.")
            state["_pending_create"] = path
        return

    # --- подтверждение создания папки ---
    if state.get("_pending_create") and text.lower() in ("да", "yes", "y"):
        path = state.pop("_pending_create")
        try:
            os.makedirs(path, exist_ok=True)
            with state_lock:
                state.update(phase="ready", cwd=path)
            send(peer_id, f"📂 Папка создана: {path}\n\nТеперь можете написать первое сообщение.")
        except OSError as e:
            send(peer_id, f"Не удалось создать папку: {e}")
        return
    else:
        state.pop("_pending_create", None)

    # --- ещё не выбрана папка / нет сессии ---
    if state["phase"] == "idle":
        send(peer_id, "Сначала начните сессию командой /start")
        return

    if not state["cwd"]:
        send(peer_id, "Сначала укажите рабочую папку.")
        return

    # --- сообщение во время работы ИИ: прерываем и запускаем заново ---
    if state["phase"] == "running":
        with state_lock:
            kill_current_proc()
        send(peer_id, "⏹ Прервал предыдущий ответ, обрабатываю новое сообщение…")

    # --- обычное сообщение -> в ИИ ---
    send(peer_id, "🤔 Думаю…")
    start_claude(peer_id, text)


# ============================================================
#  Long Poll
# ============================================================
def get_lp():
    """Поднимает Long Poll сервер сообщества."""
    r = api("groups.getLongPollServer", group_id=GROUP_ID)
    return r["server"], r["key"], r["ts"]


def run():
    server, key, ts = get_lp()
    print(f"Бот запущен. group_id={GROUP_ID}, allowed_user={ALLOWED_USER_ID}", flush=True)

    while True:
        try:
            resp = session.get(
                server,
                params={"act": "a_check", "key": key, "ts": ts, "wait": 25},
                timeout=35,
            ).json()
        except requests.RequestException:
            time.sleep(3)
            continue

        # обработка ошибок Long Poll
        if "failed" in resp:
            f = resp["failed"]
            if f == 1:                       # история устарела — новый ts
                ts = resp["ts"]
            else:                            # 2/3 — пересоздаём сессию long poll
                server, key, ts = get_lp()
            continue

        ts = resp["ts"]
        for upd in resp.get("updates", []):
            if upd.get("type") != "message_new":
                continue
            msg = upd["object"]["message"]
            from_id = msg["from_id"]

            # --- allowlist: реагируем ТОЛЬКО на свой user id ---
            # from_id из Long Poll проставляет VK, подделать нельзя.
            # Чужих и сообщения от имени сообществ (from_id < 0) молча игнорируем.
            if from_id != ALLOWED_USER_ID:
                continue

            peer_id = msg["peer_id"]
            text = msg.get("text", "")
            print(f"[msg] from={from_id}: {text!r}", flush=True)
            try:
                handle_message(peer_id, text)
            except Exception as e:
                # не роняем бота из-за ошибки в обработчике
                print(f"[error] {e}", flush=True)
                try:
                    send(peer_id, f"⚠ Ошибка: {e}")
                except Exception:
                    pass


if __name__ == "__main__":
    run()
