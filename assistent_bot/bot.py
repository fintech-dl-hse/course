import logging
import os
import re
import time
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
    specials = r"_.!"
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
) -> None:
    text = (message.get("text") or "").strip()
    cmd, args = _extract_command(text)

    if cmd not in {"/qa", "/get_chat_id"}:
        return

    chat_id, message_id, message_thread_id = _get_message_basics(message)

    try:
        tg.send_message_reaction(chat_id=chat_id, message_id=message_id, reaction_emoji="👀")
    except Exception:
        logging.getLogger(__name__).debug("Failed to set reaction", exc_info=True)

    if cmd == "/get_chat_id":
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

    if cmd == "/qa":
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


def main() -> None:
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

    try:
        me = tg.get_me()
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
                    _handle_message(tg=tg, llm=llm, message=message)

        except requests.exceptions.RequestException as e:
            logger.warning("Polling error: %s", e)
            time.sleep(2)
        except Exception:
            logger.exception("Unexpected error in polling loop")
            time.sleep(2)


if __name__ == "__main__":
    main()

