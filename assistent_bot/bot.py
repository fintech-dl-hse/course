import logging
import os
import re
import time
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import requests
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

try:
    from assistent_bot.telegram_client import TelegramClient
except ImportError:  # pragma: no cover
    from telegram_client import TelegramClient

README_URL = "https://raw.githubusercontent.com/fintech-dl-hse/course/refs/heads/main/README.md"
OPENAI_BASE_URL = "https://foundation-models.api.cloud.ru/v1"
OPENAI_MODEL = "openai/gpt-oss-120b"

# In-memory wizard state for quiz creation (keyed by admin user_id)
_QUIZ_WIZARD_STATE: dict[int, Dict[str, Any]] = {}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fintech DL HSE assistant Telegram bot")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).with_name("bot_config.json")),
        help="Path to JSON config file (default: assistent_bot/bot_config.json)",
    )
    parser.add_argument(
        "--pm-log-file",
        type=str,
        default=str(Path(__file__).with_name("private_messages.jsonl")),
        help="Path to JSONL log file for private chats (default: assistent_bot/private_messages.jsonl)",
    )
    parser.add_argument(
        "--quizzes-file",
        type=str,
        default=str(Path(__file__).with_name("quizzes.json")),
        help="Path to JSON file with quizzes (default: assistent_bot/quizzes.json)",
    )
    return parser.parse_args(argv)


def _load_settings(config_path: str) -> Dict[str, Any]:
    """
    Load bot settings from JSON file.

    Expected schema:
      - admin_users: list[int|str] (Telegram user IDs and/or usernames)
      - course_chat_id: int|null (Telegram chat ID for the course)

    The file is intentionally read on every request.
    """
    fallback: Dict[str, Any] = {"admin_users": [], "course_chat_id": None}
    try:
        path = Path(config_path)
        if not path.exists():
            example_path = Path(__file__).with_name("bot_config_example.json")
            if example_path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(example_path.read_text(encoding="utf-8"), encoding="utf-8")
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        if not isinstance(data, dict):
            return fallback
        admin_users = data.get("admin_users", [])
        if not isinstance(admin_users, list):
            admin_users = []
        course_chat_id_raw = data.get("course_chat_id", None)
        course_chat_id: int | None
        if isinstance(course_chat_id_raw, int):
            course_chat_id = course_chat_id_raw
        elif isinstance(course_chat_id_raw, str):
            try:
                course_chat_id = int(course_chat_id_raw.strip())
            except ValueError:
                course_chat_id = None
        else:
            course_chat_id = None
        return {"admin_users": admin_users, "course_chat_id": course_chat_id}
    except Exception:
        logging.getLogger(__name__).warning(
            "Failed to load config %s; using defaults",
            config_path,
            exc_info=True,
        )
        return fallback


def _save_settings(config_path: str, settings: Dict[str, Any]) -> None:
    """
    Save bot settings to JSON file.

    Uses atomic write (tmp file + replace) to reduce risk of corruption.
    """
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "admin_users": settings.get("admin_users") or [],
        "course_chat_id": settings.get("course_chat_id", None),
    }
    raw = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    tmp_path.write_text(raw, encoding="utf-8")
    tmp_path.replace(path)


def _quiz_sort_key(q: Dict[str, Any]) -> tuple[int, int | str]:
    qid = q.get("id")
    if isinstance(qid, int):
        return (0, qid)
    if isinstance(qid, str):
        s = qid.strip()
        if s.lstrip("-").isdigit():
            return (0, int(s))
        return (1, s)
    return (1, str(qid))


def _load_quizzes(quizzes_file: str) -> list[Dict[str, Any]]:
    path = Path(quizzes_file)
    if not path.exists():
        return []
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        if not isinstance(data, list):
            return []
        quizzes: list[Dict[str, Any]] = []
        for item in data:
            if isinstance(item, dict) and "id" in item:
                quizzes.append(item)
        quizzes.sort(key=_quiz_sort_key)
        return quizzes
    except Exception:
        logging.getLogger(__name__).warning(
            "Failed to load quizzes file %s; using empty list",
            quizzes_file,
            exc_info=True,
        )
        return []


def _save_quizzes(quizzes_file: str, quizzes: list[Dict[str, Any]]) -> None:
    path = Path(quizzes_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    quizzes_sorted = list(quizzes)
    quizzes_sorted.sort(key=_quiz_sort_key)
    raw = json.dumps(quizzes_sorted, ensure_ascii=False, indent=2) + "\n"
    tmp_path.write_text(raw, encoding="utf-8")
    tmp_path.replace(path)


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        raise RuntimeError(f"Required env var is not set: {name}")
    return value


def _extract_command(text: str) -> Tuple[str, str]:
    """
    Returns (command, args).

    Handles /cmd and /cmd@botname forms.
    """
    text = (text or "").strip()
    if not text.startswith("/"):
        return "", ""

    parts = text.split(maxsplit=1)
    raw_cmd = parts[0]
    args = parts[1] if len(parts) > 1 else ""

    cmd = raw_cmd.split("@", maxsplit=1)[0]
    return cmd, args.strip()


def _get_message_basics(message: Dict[str, Any]) -> Tuple[int, int, int]:
    chat_id = int(message["chat"]["id"])
    message_id = int(message["message_id"])
    message_thread_id = int(message.get("message_thread_id") or 0)
    return chat_id, message_id, message_thread_id


def _get_sender(message: Dict[str, Any]) -> Tuple[int, str]:
    sender = message.get("from") or {}
    user_id = int(sender.get("id") or 0)
    username = str(sender.get("username") or "").strip()
    return user_id, username


def _is_admin(settings: Dict[str, Any], user_id: int, username: str) -> bool:
    admins = settings.get("admin_users") or []
    username_norm = username.lstrip("@").lower()
    for entry in admins:
        if isinstance(entry, int) and entry == user_id:
            return True
        if isinstance(entry, str) and entry.strip().lstrip("@").lower() == username_norm and username_norm:
            return True
    return False


def _log_private_message(message: Dict[str, Any], pm_log_file: str) -> None:
    chat = message.get("chat") or {}
    if not isinstance(chat, dict):
        return
    if chat.get("type") != "private":
        return

    sender = message.get("from") or {}
    if not isinstance(sender, dict):
        sender = {}

    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "message_date": int(message.get("date") or 0),
        "chat_id": int(chat.get("id") or 0),
        "user_id": int(sender.get("id") or 0),
        "username": str(sender.get("username") or ""),
        "first_name": str(sender.get("first_name") or ""),
        "last_name": str(sender.get("last_name") or ""),
        "message_id": int(message.get("message_id") or 0),
        "text": str(message.get("text") or ""),
    }

    try:
        path = Path(pm_log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        logging.getLogger(__name__).warning(
            "Failed to write private message log to %s",
            pm_log_file,
            exc_info=True,
        )


def _fetch_readme() -> str:
    resp = requests.get(README_URL, timeout=20)
    resp.raise_for_status()
    return resp.text



def _escape_markdown_v2(text: str) -> str:
    """
    Escapes text for Telegram MarkdownV2.
    Docs: https://core.telegram.org/bots/api#markdownv2-style
    """
    if text is None:
        return ""
    text = text.replace("**", "*")
    text = text.replace("\\", "\\\\")
    specials = r"_()[].!-"
    return re.sub(rf"([{re.escape(specials)}])", r"\\\1", text)



def _send_with_formatting_fallback(
    tg: TelegramClient,
    chat_id: int,
    message_thread_id: int,
    text: str,
) -> None:
    escaped = _escape_markdown_v2(text)
    print("escaped", escaped)
    resp2 = tg.send_message(
        chat_id=chat_id,
        message_thread_id=message_thread_id,
        parse_mode="MarkdownV2",
        message=escaped,
    )
    if getattr(resp2, "status_code", 500) == 200:
        return

    tg.send_message(
        chat_id=chat_id,
        message_thread_id=message_thread_id,
        parse_mode=None,
        message=text,
    )


def _build_messages(readme: str, user_question: str) -> list[ChatCompletionMessageParam]:
    system = (
        "You are a teaching assistant bot for the HSE Fintech Deep Learning course.\n"
        "You must answer ONLY questions that are relevant to the course topics/materials.\n"
        "Use the provided README as the primary context. If the question is off-topic, "
        "or asks for disallowed content (cheating, hacking, illegal harm, etc.), refuse briefly "
        "and suggest asking a course-related question.\n"
        "Be concise, technically correct, and prefer practical guidance.\n"
        "Answer in the same language as the user's question.\n"
        "Do NOT use markdown tables. Prefer bullet lists.\n"
    )

    user = (
        "Course README context:\n"
        "-----\n"
        f"{readme}\n"
        "-----\n\n"
        "User question:\n"
        f"{user_question}\n"
    )

    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return messages


def _answer_question(client: OpenAI, readme: str, question: str) -> str:
    messages = _build_messages(readme=readme, user_question=question)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        max_tokens=2500,
        temperature=0.5,
        presence_penalty=0,
        top_p=0.95,
        messages=messages,
    )
    content = response.choices[0].message.content or ""
    return content.strip() or "Не смог сформировать ответ. Попробуйте переформулировать вопрос."


def _handle_message(
    tg: TelegramClient,
    llm: OpenAI,
    message: Dict[str, Any],
    config_path: str,
    pm_log_file: str,
    quizzes_file: str,
    bot_user_id: int,
) -> None:
    _log_private_message(message=message, pm_log_file=pm_log_file)
    settings = _load_settings(config_path)
    text = (message.get("text") or "").strip()
    cmd, args = _extract_command(text)
    chat_id, message_id, message_thread_id = _get_message_basics(message)
    user_id, username = _get_sender(message)
    is_admin = _is_admin(settings=settings, user_id=user_id, username=username)
    chat_type = str((message.get("chat") or {}).get("type") or "")

    # Continue quiz creation wizard (non-command messages)
    if cmd == "" and chat_type == "private" and is_admin and user_id in _QUIZ_WIZARD_STATE:
        state = _QUIZ_WIZARD_STATE.get(user_id) or {}
        stage = str(state.get("stage") or "")
        quiz_id = str(state.get("quiz_id") or "").strip()
        if stage == "await_question":
            question = text.strip()
            if not question:
                _send_with_formatting_fallback(
                    tg=tg,
                    chat_id=chat_id,
                    message_thread_id=message_thread_id,
                    text="Пожалуйста, отправьте непустой вопрос для квиза.",
                )
                return
            state["question"] = question
            state["stage"] = "await_answer"
            _QUIZ_WIZARD_STATE[user_id] = state
            _send_with_formatting_fallback(
                tg=tg,
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text=f"Квиз {quiz_id}: теперь отправьте правильный ответ.",
            )
            return
        if stage == "await_answer":
            answer = text.strip()
            if not answer:
                _send_with_formatting_fallback(
                    tg=tg,
                    chat_id=chat_id,
                    message_thread_id=message_thread_id,
                    text="Пожалуйста, отправьте непустой правильный ответ для квиза.",
                )
                return

            question = str(state.get("question") or "").strip()
            quiz = {"id": quiz_id, "question": question, "answer": answer}

            quizzes = _load_quizzes(quizzes_file)
            if any(str(q.get("id") or "") == quiz_id for q in quizzes):
                _QUIZ_WIZARD_STATE.pop(user_id, None)
                _send_with_formatting_fallback(
                    tg=tg,
                    chat_id=chat_id,
                    message_thread_id=message_thread_id,
                    text=f"Квиз с id={quiz_id} уже существует. Создание отменено.",
                )
                return

            quizzes.append(quiz)
            try:
                _save_quizzes(quizzes_file=quizzes_file, quizzes=quizzes)
            except Exception as e:
                _send_with_formatting_fallback(
                    tg=tg,
                    chat_id=chat_id,
                    message_thread_id=message_thread_id,
                    text=f"Не удалось сохранить квиз: {type(e).__name__}: {e}",
                )
                return
            _QUIZ_WIZARD_STATE.pop(user_id, None)
            _send_with_formatting_fallback(
                tg=tg,
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text=f"Готово. Квиз {quiz_id} сохранён.",
            )
            return

    if cmd not in {
        "/qa",
        "/get_chat_id",
        "/help",
        "/add_admin",
        "/course_chat",
        "/course_members",
        "/quiz_create",
        "/quiz_list",
    }:
        return

    try:
        tg.send_message_reaction(chat_id=chat_id, message_id=message_id, reaction_emoji="👀")
    except Exception:
        logging.getLogger(__name__).debug("Failed to set reaction", exc_info=True)

    if cmd == "/help":
        lines = [
            "Доступные команды:",
            "- /help",
            "- /qa <вопрос>",
            "- /get_chat_id",
        ]
        if is_admin:
            lines.append("- /add_admin <user_id>")
            lines.append("- /course_chat <chat_id>")
            lines.append("- /course_members")
            lines.append("- /quiz_create <quiz_id>")
            lines.append("- /quiz_list")
        _send_with_formatting_fallback(
            tg=tg,
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            text="\n".join(lines),
        )
        return
    elif cmd == "/add_admin":
        if not is_admin:
            _send_with_formatting_fallback(
                tg=tg,
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text="Недостаточно прав: команда доступна только администраторам.",
            )
            return

        raw_user_id = (args or "").strip()
        if not raw_user_id:
            _send_with_formatting_fallback(
                tg=tg,
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text="Usage: /add_admin <user_id>",
            )
            return

        try:
            new_admin_id = int(raw_user_id)
        except ValueError:
            _send_with_formatting_fallback(
                tg=tg,
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text="Usage: /add_admin <user_id> (user_id должен быть числом)",
            )
            return

        admin_users = settings.get("admin_users") or []
        if not isinstance(admin_users, list):
            admin_users = []

        already = False
        for entry in admin_users:
            if isinstance(entry, int) and entry == new_admin_id:
                already = True
                break
            if isinstance(entry, str) and entry.strip().isdigit() and int(entry.strip()) == new_admin_id:
                already = True
                break

        if already:
            _send_with_formatting_fallback(
                tg=tg,
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text=f"Пользователь {new_admin_id} уже является администратором.",
            )
            return

        admin_users.append(new_admin_id)
        settings["admin_users"] = admin_users
        try:
            _save_settings(config_path=config_path, settings=settings)
        except Exception as e:
            _send_with_formatting_fallback(
                tg=tg,
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text=f"Не удалось сохранить конфиг: {type(e).__name__}: {e}",
            )
            return

        _send_with_formatting_fallback(
            tg=tg,
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            text=f"Готово. Добавил администратора: {new_admin_id}",
        )
        return
    elif cmd == "/course_chat":
        if not is_admin:
            _send_with_formatting_fallback(
                tg=tg,
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text="Недостаточно прав: команда доступна только администраторам.",
            )
            return

        raw_chat_id = (args or "").strip()
        if not raw_chat_id:
            _send_with_formatting_fallback(
                tg=tg,
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text="Usage: /course_chat <chat_id>",
            )
            return

        try:
            course_chat_id = int(raw_chat_id)
        except ValueError:
            _send_with_formatting_fallback(
                tg=tg,
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text="Usage: /course_chat <chat_id> (chat_id должен быть числом)",
            )
            return

        if bot_user_id <= 0:
            _send_with_formatting_fallback(
                tg=tg,
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text="Не удалось определить bot_id через Telegram API. Попробуйте перезапустить бота.",
            )
            return

        try:
            member = tg.get_chat_member(chat_id=course_chat_id, user_id=bot_user_id)
            status = str((member.get("result") or {}).get("status") or "")
        except Exception as e:
            _send_with_formatting_fallback(
                tg=tg,
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text=f"Не удалось проверить права бота в чате: {type(e).__name__}: {e}",
            )
            return

        if status not in {"administrator", "creator"}:
            _send_with_formatting_fallback(
                tg=tg,
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text=(
                    "Бот должен быть администратором (суперпользователем) в этом чате.\n"
                    f"Текущий статус: {status or 'unknown'}"
                ),
            )
            return

        settings["course_chat_id"] = course_chat_id
        try:
            _save_settings(config_path=config_path, settings=settings)
        except Exception as e:
            _send_with_formatting_fallback(
                tg=tg,
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text=f"Не удалось сохранить конфиг: {type(e).__name__}: {e}",
            )
            return

        _send_with_formatting_fallback(
            tg=tg,
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            text=f"Готово. Установил чат курса: {course_chat_id}",
        )
        return
    elif cmd == "/course_members":
        if not is_admin:
            _send_with_formatting_fallback(
                tg=tg,
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text="Недостаточно прав: команда доступна только администраторам.",
            )
            return

        course_chat_id = settings.get("course_chat_id")
        if not isinstance(course_chat_id, int) or course_chat_id == 0:
            _send_with_formatting_fallback(
                tg=tg,
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text="Чат курса не настроен. Сначала выполните: /course_chat <chat_id>",
            )
            return

        path = Path(pm_log_file)
        if not path.exists():
            _send_with_formatting_fallback(
                tg=tg,
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text="Файл логов не найден. Пользователей: 0",
            )
            return

        users: set[int] = set()
        total_lines = 0
        bad_lines = 0
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    total_lines += 1
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        uid = int((rec or {}).get("user_id") or 0)
                        if uid > 0:
                            users.add(uid)
                    except Exception:
                        bad_lines += 1
        except Exception as e:
            _send_with_formatting_fallback(
                tg=tg,
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text=f"Не удалось прочитать файл логов: {type(e).__name__}: {e}",
            )
            return

        in_course_users: set[int] = set()
        checked = 0
        check_errors = 0
        for uid in users:
            checked += 1
            try:
                member = tg.get_chat_member(chat_id=course_chat_id, user_id=uid)
                status = str((member.get("result") or {}).get("status") or "")
                if status in {"creator", "administrator", "member", "restricted"}:
                    in_course_users.add(uid)
            except Exception:
                check_errors += 1

        _send_with_formatting_fallback(
            tg=tg,
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            text=(
                "Статистика по личным сообщениям:\n"
                f"- пользователей (всего в логе): {len(users)}\n"
                f"- пользователей (в чате курса): {len(in_course_users)}\n"
                f"- строк в логе: {total_lines}\n"
                f"- битых строк: {bad_lines}\n"
                f"- проверено membership: {checked}\n"
                f"- ошибок проверки membership: {check_errors}"
            ),
        )
        return
    elif cmd == "/quiz_create":
        if chat_type != "private":
            _send_with_formatting_fallback(
                tg=tg,
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text="Команда доступна только в личных сообщениях с ботом.",
            )
            return
        if not is_admin:
            _send_with_formatting_fallback(
                tg=tg,
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text="Недостаточно прав: команда доступна только администраторам.",
            )
            return

        quiz_id = (args or "").strip()
        if not quiz_id:
            _send_with_formatting_fallback(
                tg=tg,
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text="Usage: /quiz_create <quiz_id>",
            )
            return

        quizzes = _load_quizzes(quizzes_file)
        if any(str(q.get("id") or "") == quiz_id for q in quizzes):
            _send_with_formatting_fallback(
                tg=tg,
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text=f"Квиз с id={quiz_id} уже существует.",
            )
            return

        _QUIZ_WIZARD_STATE[user_id] = {"stage": "await_question", "quiz_id": quiz_id}
        _send_with_formatting_fallback(
            tg=tg,
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            text=f"Создание квиза {quiz_id}. Отправьте вопрос для квиза.",
        )
        return
    elif cmd == "/quiz_list":
        if not is_admin:
            _send_with_formatting_fallback(
                tg=tg,
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text="Недостаточно прав: команда доступна только администраторам.",
            )
            return

        quizzes = _load_quizzes(quizzes_file)
        if not quizzes:
            _send_with_formatting_fallback(
                tg=tg,
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text="Список квизов пуст.",
            )
            return

        for q in quizzes:
            qid = str(q.get("id") or "").strip()
            question = str(q.get("question") or "").strip()
            answer = str(q.get("answer") or "").strip()
            _send_with_formatting_fallback(
                tg=tg,
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text=(
                    f"Квиз: {qid}\n"
                    f"Вопрос: {question}\n"
                    f"Ответ: {answer}"
                ),
            )
        return
    elif cmd == "/get_chat_id":
        _send_with_formatting_fallback(
            tg=tg,
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            text=(
                "chat_id: "
                f"{chat_id}\n"
                "message_thread_id: "
                f"{message_thread_id}\n"
            ),
        )
        return
    elif cmd == "/qa":
        if not args:
            _send_with_formatting_fallback(
                tg=tg,
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text="Usage: /qa <вопрос>",
            )
            return

        try:
            readme = _fetch_readme()
        except Exception as e:
            _send_with_formatting_fallback(
                tg=tg,
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text=f"Failed to fetch README context: {type(e).__name__}: {e}",
            )
            return

        try:
            answer = _answer_question(client=llm, readme=readme, question=args)
        except Exception as e:
            _send_with_formatting_fallback(
                tg=tg,
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text=f"LLM request failed: {type(e).__name__}: {e}",
            )
            return

        _send_with_formatting_fallback(
            tg=tg,
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            text=answer,
        )
    else:
        pass

def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    _require_env("TELEGRAM_BOT_TOKEN")
    api_key = _require_env("API_KEY")

    tg = TelegramClient()
    llm = OpenAI(
        api_key=api_key,
        base_url=OPENAI_BASE_URL,
    )

    bot_user_id = 0
    try:
        me = tg.get_me()
        bot_user_id = int((me.get("result") or {}).get("id") or 0)
        logger.info("Bot started: %s", me.get("result", {}).get("username"))
    except Exception:
        logger.info("Bot started")

    offset = 0
    while True:
        try:
            data = tg.get_updates(offset=offset)
            results = data.get("result") or []

            for update in results:
                update_id = update.get("update_id")
                if isinstance(update_id, int):
                    offset = max(offset, update_id + 1)

                message = update.get("message")
                if isinstance(message, dict):
                    _handle_message(
                        tg=tg,
                        llm=llm,
                        message=message,
                        config_path=args.config,
                        pm_log_file=args.pm_log_file,
                        quizzes_file=args.quizzes_file,
                        bot_user_id=bot_user_id,
                    )

        except requests.exceptions.RequestException as e:
            logger.warning("Polling error: %s", e)
            time.sleep(2)
        except Exception:
            logger.exception("Unexpected error in polling loop")
            time.sleep(2)


if __name__ == "__main__":
    main()

