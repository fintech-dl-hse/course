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
    parser.add_argument(
        "--quiz-state-file",
        type=str,
        default=str(Path(__file__).with_name("quiz_state.json")),
        help="Path to JSON file with per-user quiz state (default: assistent_bot/quiz_state.json)",
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
                quiz = dict(item)
                if "processed" not in quiz:
                    quiz["processed"] = False
                quiz["processed"] = bool(quiz.get("processed"))
                quizzes.append(quiz)
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
    normalized: list[Dict[str, Any]] = []
    for q in quizzes_sorted:
        if not isinstance(q, dict):
            continue
        normalized.append(
            {
                "id": q.get("id"),
                "question": q.get("question"),
                "answer": q.get("answer"),
                "processed": bool(q.get("processed")),
            }
        )
    raw = json.dumps(normalized, ensure_ascii=False, indent=2) + "\n"
    tmp_path.write_text(raw, encoding="utf-8")
    tmp_path.replace(path)

def _load_quiz_state(quiz_state_file: str) -> Dict[str, Any]:
    path = Path(quiz_state_file)
    if not path.exists():
        return {"users": {}}
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        if not isinstance(data, dict):
            return {"users": {}}
        users = data.get("users")
        if not isinstance(users, dict):
            users = {}
        return {"users": users}
    except Exception:
        logging.getLogger(__name__).warning(
            "Failed to load quiz state file %s; using empty state",
            quiz_state_file,
            exc_info=True,
        )
        return {"users": {}}


def _save_quiz_state(quiz_state_file: str, state: Dict[str, Any]) -> None:
    path = Path(quiz_state_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    users = state.get("users")
    if not isinstance(users, dict):
        users = {}

    def _user_key_sort(k: str) -> tuple[int, int | str]:
        s = str(k)
        if s.lstrip("-").isdigit():
            return (0, int(s))
        return (1, s)

    normalized_users: Dict[str, Any] = {}
    for user_key in sorted(users.keys(), key=_user_key_sort):
        u = users.get(user_key)
        if not isinstance(u, dict):
            continue
        active_quiz_id = u.get("active_quiz_id")
        if active_quiz_id is not None:
            active_quiz_id = str(active_quiz_id)

        results = u.get("results")
        if not isinstance(results, dict):
            results = {}
        norm_results: Dict[str, Any] = {}
        for qid in sorted(results.keys(), key=_user_key_sort):
            r = results.get(qid)
            if not isinstance(r, dict):
                continue
            norm_results[str(qid)] = {
                "correct": bool(r.get("correct")),
                "attempts": int(r.get("attempts") or 0),
            }

        answers = u.get("answers")
        if not isinstance(answers, dict):
            answers = {}
        norm_answers: Dict[str, Any] = {}
        for qid in sorted(answers.keys(), key=_user_key_sort):
            arr = answers.get(qid)
            if not isinstance(arr, list):
                continue
            norm_answers[str(qid)] = [
                {
                    "answer": str(a.get("answer") or "") if isinstance(a, dict) else str(a),
                    "ts": str(a.get("ts") or "") if isinstance(a, dict) else "",
                    "correct": bool(a.get("correct")) if isinstance(a, dict) else False,
                }
                for a in arr
                if isinstance(a, (dict, str))
            ]

        normalized_users[str(user_key)] = {
            "active_quiz_id": active_quiz_id,
            "results": norm_results,
            "answers": norm_answers,
        }

    payload = {"users": normalized_users}
    raw = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    tmp_path.write_text(raw, encoding="utf-8")
    tmp_path.replace(path)


def _get_user_quiz_state(state: Dict[str, Any], user_id: int) -> Dict[str, Any]:
    users = state.get("users")
    if not isinstance(users, dict):
        users = {}
        state["users"] = users
    key = str(int(user_id))
    u = users.get(key)
    if not isinstance(u, dict):
        u = {"active_quiz_id": None, "results": {}, "answers": {}}
        users[key] = u
    if "results" not in u or not isinstance(u.get("results"), dict):
        u["results"] = {}
    if "answers" not in u or not isinstance(u.get("answers"), dict):
        u["answers"] = {}
    if "active_quiz_id" not in u:
        u["active_quiz_id"] = None
    return u


def _append_user_answer(
    user_state: Dict[str, Any],
    quiz_id: str,
    answer: str,
    is_correct: bool,
) -> None:
    answers = user_state.get("answers")
    if not isinstance(answers, dict):
        answers = {}
        user_state["answers"] = answers
    qkey = str(quiz_id)
    arr = answers.get(qkey)
    if not isinstance(arr, list):
        arr = []
        answers[qkey] = arr
    arr.append(
        {
            "answer": answer,
            "ts": datetime.now(timezone.utc).isoformat(),
            "correct": bool(is_correct),
        }
    )


def _handle_callback_query(
    tg: TelegramClient,
    callback_query: Dict[str, Any],
    config_path: str,
    pm_log_file: str,
    quizzes_file: str,
    quiz_state_file: str,
) -> None:
    settings = _load_settings(config_path)
    sender = callback_query.get("from") or {}
    user_id = int((sender.get("id") or 0) if isinstance(sender, dict) else 0)
    username = str((sender.get("username") or "") if isinstance(sender, dict) else "").strip()
    is_admin = _is_admin(settings=settings, user_id=user_id, username=username)

    callback_query_id = str(callback_query.get("id") or "")
    data = str(callback_query.get("data") or "")

    if not is_admin:
        try:
            tg.answer_callback_query(
                callback_query_id=callback_query_id,
                text="Недостаточно прав.",
                show_alert=True,
            )
        except Exception:
            logging.getLogger(__name__).debug("Failed to answer callback_query", exc_info=True)
        return

    action = ""
    quiz_id = ""
    if data.startswith("quiz_send_all:"):
        action = "send_all"
        quiz_id = data.split(":", 1)[1].strip()
    elif data.startswith("quiz_send_admins:"):
        action = "send_admins"
        quiz_id = data.split(":", 1)[1].strip()
    else:
        try:
            tg.answer_callback_query(callback_query_id=callback_query_id, text="Неизвестная кнопка.")
        except Exception:
            logging.getLogger(__name__).debug("Failed to answer callback_query", exc_info=True)
        return

    if not quiz_id:
        try:
            tg.answer_callback_query(callback_query_id=callback_query_id, text="Некорректный quiz_id.")
        except Exception:
            logging.getLogger(__name__).debug("Failed to answer callback_query", exc_info=True)
        return

    quizzes = _load_quizzes(quizzes_file)
    quiz: Dict[str, Any] | None = None
    for q in quizzes:
        if str(q.get("id") or "") == quiz_id:
            quiz = q
            break

    if quiz is None:
        try:
            tg.answer_callback_query(callback_query_id=callback_query_id, text="Квиз не найден.", show_alert=True)
        except Exception:
            logging.getLogger(__name__).debug("Failed to answer callback_query", exc_info=True)
        return

    if action == "send_all" and bool(quiz.get("processed")):
        try:
            tg.answer_callback_query(callback_query_id=callback_query_id, text="Квиз уже помечен как processed.")
        except Exception:
            logging.getLogger(__name__).debug("Failed to answer callback_query", exc_info=True)
        return

    try:
        tg.answer_callback_query(callback_query_id=callback_query_id, text="Начинаю отправку...")
    except Exception:
        logging.getLogger(__name__).debug("Failed to answer callback_query", exc_info=True)

    question = str(quiz.get("question") or "").strip()
    sent_ok = 0
    sent_fail = 0
    total_targets = 0

    if action == "send_admins":
        admin_users = settings.get("admin_users") or []
        admin_ids: set[int] = set()
        if isinstance(admin_users, list):
            for entry in admin_users:
                if isinstance(entry, int) and entry > 0:
                    admin_ids.add(entry)
                elif isinstance(entry, str) and entry.strip().isdigit():
                    admin_ids.add(int(entry.strip()))
        targets = sorted(admin_ids)
        total_targets = len(targets)
        state = _load_quiz_state(quiz_state_file)
        sent_admin_users: list[int] = []
        for uid in targets:
            try:
                resp = tg.send_message(chat_id=uid, message=question, parse_mode=None)
                if getattr(resp, "status_code", 500) == 200:
                    sent_ok += 1
                    sent_admin_users.append(uid)
                else:
                    sent_fail += 1
            except Exception:
                sent_fail += 1
        for uid in sent_admin_users:
            u = _get_user_quiz_state(state, uid)
            u["active_quiz_id"] = str(quiz_id)
        try:
            _save_quiz_state(quiz_state_file, state)
        except Exception:
            logging.getLogger(__name__).warning("Failed to save quiz state file %s", quiz_state_file, exc_info=True)
    else:
        course_chat_id = settings.get("course_chat_id")
        if not isinstance(course_chat_id, int) or course_chat_id == 0:
            try:
                tg.answer_callback_query(
                    callback_query_id=callback_query_id,
                    text="Чат курса не настроен. Сначала: /course_chat <chat_id>",
                    show_alert=True,
                )
            except Exception:
                logging.getLogger(__name__).debug("Failed to answer callback_query", exc_info=True)
            return

        path = Path(pm_log_file)
        users: set[int] = set()
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                            uid = int((rec or {}).get("user_id") or 0)
                            if uid > 0:
                                users.add(uid)
                        except Exception:
                            continue
            except Exception:
                logging.getLogger(__name__).warning("Failed to read pm log file %s", pm_log_file, exc_info=True)

        in_course_users: list[int] = []
        for uid in sorted(users):
            try:
                member = tg.get_chat_member(chat_id=course_chat_id, user_id=uid)
                status = str((member.get("result") or {}).get("status") or "")
                if status in {"creator", "administrator", "member", "restricted"}:
                    in_course_users.append(uid)
            except Exception:
                continue

        total_targets = len(in_course_users)
        state = _load_quiz_state(quiz_state_file)
        sent_users: list[int] = []
        for uid in in_course_users:
            try:
                resp = tg.send_message(chat_id=uid, message=question, parse_mode=None)
                if getattr(resp, "status_code", 500) == 200:
                    sent_ok += 1
                    sent_users.append(uid)
                else:
                    sent_fail += 1
            except Exception:
                sent_fail += 1

        processed_now = sent_fail == 0
        quiz["processed"] = processed_now
        try:
            _save_quizzes(quizzes_file=quizzes_file, quizzes=quizzes)
        except Exception:
            logging.getLogger(__name__).warning("Failed to save quizzes file %s", quizzes_file, exc_info=True)
        for uid in sent_users:
            u = _get_user_quiz_state(state, uid)
            u["active_quiz_id"] = str(quiz_id)
        try:
            _save_quiz_state(quiz_state_file, state)
        except Exception:
            logging.getLogger(__name__).warning("Failed to save quiz state file %s", quiz_state_file, exc_info=True)

    msg = callback_query.get("message") or {}
    if isinstance(msg, dict):
        cb_chat_id = int((msg.get("chat") or {}).get("id") or 0)
        cb_message_id = int(msg.get("message_id") or 0)
        prev_text = str(msg.get("text") or "").strip()
        status_line = ""
        if action == "send_admins":
            status_line = f"Отправлено администраторам: {sent_ok}/{total_targets}\nОшибок: {sent_fail}"
        else:
            status_line = (
                f"Отправлено: {sent_ok}/{total_targets}\n"
                f"Ошибок: {sent_fail}\n"
                f"processed: {str(bool(quiz.get('processed'))).lower()}"
            )
        new_text = f"{prev_text}\n\n{status_line}".strip()
        try:
            tg.edit_message_text(chat_id=cb_chat_id, message_id=cb_message_id, text=new_text, parse_mode=None)
        except Exception:
            logging.getLogger(__name__).debug("Failed to edit message text", exc_info=True)
        if action == "send_all":
            try:
                tg.edit_message_reply_markup(
                    chat_id=cb_chat_id,
                    message_id=cb_message_id,
                    reply_markup={"inline_keyboard": []},
                )
            except Exception:
                logging.getLogger(__name__).debug("Failed to edit reply markup", exc_info=True)


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
    quiz_state_file: str,
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
            quiz = {"id": quiz_id, "question": question, "answer": answer, "processed": False}

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

    # User answer processing (non-command private messages)
    if cmd == "" and chat_type == "private" and not is_admin:
        state = _load_quiz_state(quiz_state_file)
        user_state = _get_user_quiz_state(state, user_id)
        active_quiz_id = user_state.get("active_quiz_id")
        if active_quiz_id is None or str(active_quiz_id).strip() == "":
            return
        active_quiz_id = str(active_quiz_id).strip()

        quizzes = _load_quizzes(quizzes_file)
        quiz: Dict[str, Any] | None = None
        for q in quizzes:
            if str(q.get("id") or "").strip() == active_quiz_id:
                quiz = q
                break
        if quiz is None:
            user_state["active_quiz_id"] = None
            try:
                _save_quiz_state(quiz_state_file, state)
            except Exception:
                logging.getLogger(__name__).warning("Failed to save quiz state file %s", quiz_state_file, exc_info=True)
            return

        correct_answer = str(quiz.get("answer") or "").strip()
        user_answer = text.strip()
        qkey = str(active_quiz_id)
        results = user_state.get("results")
        if not isinstance(results, dict):
            results = {}
            user_state["results"] = results
        prev = results.get(qkey)
        prev_attempts = int((prev or {}).get("attempts") or 0) if isinstance(prev, dict) else 0
        attempts_now = prev_attempts + 1

        is_correct = user_answer == correct_answer
        _append_user_answer(user_state=user_state, quiz_id=qkey, answer=user_answer, is_correct=is_correct)

        if not is_correct:
            results[qkey] = {"correct": False, "attempts": attempts_now}
            try:
                _save_quiz_state(quiz_state_file, state)
            except Exception:
                logging.getLogger(__name__).warning("Failed to save quiz state file %s", quiz_state_file, exc_info=True)
            _send_with_formatting_fallback(
                tg=tg,
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text=f"Неправильно. Попыток: {attempts_now}. Попробуйте ещё раз.",
            )
            return

        results[qkey] = {"correct": True, "attempts": attempts_now}
        user_state["active_quiz_id"] = None
        try:
            _save_quiz_state(quiz_state_file, state)
        except Exception:
            logging.getLogger(__name__).warning("Failed to save quiz state file %s", quiz_state_file, exc_info=True)
        _send_with_formatting_fallback(
            tg=tg,
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            text="Правильно! Поздравляю.",
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
        "/quiz_delete",
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
            lines.append("- /quiz_delete <quiz_id>")
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
            processed = bool(q.get("processed"))
            reply_markup = None
            buttons: list[list[Dict[str, str]]] = [
                [{"text": "Отправить администраторам", "callback_data": f"quiz_send_admins:{qid}"}]
            ]
            if not processed:
                buttons.append([{"text": "Отправить всем", "callback_data": f"quiz_send_all:{qid}"}])
            reply_markup = {"inline_keyboard": buttons}
            tg.send_message(
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                parse_mode=None,
                message=(
                    f"Квиз: {qid}\n"
                    f"processed: {str(processed).lower()}\n"
                    f"Вопрос: {question}\n"
                    f"Ответ: {answer}"
                ),
                reply_markup=reply_markup,
            )
        return
    elif cmd == "/quiz_delete":
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
                text="Usage: /quiz_delete <quiz_id>",
            )
            return

        quizzes = _load_quizzes(quizzes_file)
        before = len(quizzes)
        quizzes = [q for q in quizzes if str(q.get("id") or "") != quiz_id]
        after = len(quizzes)

        if after == before:
            _send_with_formatting_fallback(
                tg=tg,
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text=f"Квиз с id={quiz_id} не найден.",
            )
            return

        try:
            _save_quizzes(quizzes_file=quizzes_file, quizzes=quizzes)
        except Exception as e:
            _send_with_formatting_fallback(
                tg=tg,
                chat_id=chat_id,
                message_thread_id=message_thread_id,
                text=f"Не удалось сохранить файл квизов: {type(e).__name__}: {e}",
            )
            return

        # If wizard was creating this quiz, cancel it
        state = _QUIZ_WIZARD_STATE.get(user_id) or {}
        if str(state.get("quiz_id") or "") == quiz_id:
            _QUIZ_WIZARD_STATE.pop(user_id, None)

        _send_with_formatting_fallback(
            tg=tg,
            chat_id=chat_id,
            message_thread_id=message_thread_id,
            text=f"Готово. Удалил квиз: {quiz_id}",
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
                        quiz_state_file=args.quiz_state_file,
                        bot_user_id=bot_user_id,
                    )

                callback_query = update.get("callback_query")
                if isinstance(callback_query, dict):
                    _handle_callback_query(
                        tg=tg,
                        callback_query=callback_query,
                        config_path=args.config,
                        pm_log_file=args.pm_log_file,
                        quizzes_file=args.quizzes_file,
                        quiz_state_file=args.quiz_state_file,
                    )

        except requests.exceptions.RequestException as e:
            logger.warning("Polling error: %s", e)
            time.sleep(2)
        except Exception:
            logger.exception("Unexpected error in polling loop")
            time.sleep(2)


if __name__ == "__main__":
    main()

