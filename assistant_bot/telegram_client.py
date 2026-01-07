import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests

MAX_PHOTO_CAPTION_LENGTH = 512


class TelegramAPIError(Exception):
    """Exception raised when Telegram API returns a non-200 status code."""

    def __init__(self, status_code: int, endpoint: str, content: bytes, method: str):
        self.status_code = status_code
        self.endpoint = endpoint
        self.content = content
        self.method = method
        message = f"{method} {endpoint} returned status {status_code}"
        super().__init__(message)


class TelegramClient:
    """Client for interacting with Telegram Bot API."""

    def __init__(self, bot_token: Optional[str] = None):
        self._telegram_bot_token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN")
        if self._telegram_bot_token is None:
            raise ValueError("TELEGRAM_BOT_TOKEN env var is required")

        self.logger = logging.getLogger(__name__)

    def _truncate_caption(self, caption: str) -> str:
        """Trim captions to Telegram-safe length."""
        if caption is None:
            return ""
        if len(caption) <= MAX_PHOTO_CAPTION_LENGTH:
            return caption

        truncated = caption[: MAX_PHOTO_CAPTION_LENGTH - 3] + "..."
        self.logger.warning(
            "Truncated photo caption from %d to %d characters",
            len(caption),
            len(truncated),
        )
        return truncated

    def _request(
        self,
        method: str,
        endpoint: str,
        timeout: int = 10,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        enable_debug_logs=True,
    ) -> requests.Response:
        """Make a request to Telegram Bot API with logging and error handling.

        Args:
            method: HTTP method ('GET' or 'POST')
            endpoint: API endpoint (e.g., 'sendMessage', 'getMe')
            timeout: Request timeout in seconds
            params: Query parameters for GET requests
            json_data: JSON payload for POST requests
            data: Form data for POST requests
            files: Files to upload (for multipart/form-data)

        Returns:
            Response object from requests library

        Raises:
            requests.exceptions.RequestException: On network errors or timeouts
            ValueError: On invalid HTTP method
            TelegramAPIError: On non-200 status code responses
        """
        url = f"https://api.telegram.org/bot{self._telegram_bot_token}/{endpoint}"

        if enable_debug_logs:
            self.logger.debug(
                "api request: %s %s",
                method,
                endpoint,
            )

        start_time = time.perf_counter()
        try:
            if method.upper() == "GET":
                resp = requests.get(
                    url,
                    params=params,
                    timeout=timeout,
                )
            elif method.upper() == "POST":
                resp = requests.post(
                    url,
                    params=params,
                    json=json_data,
                    data=data,
                    files=files,
                    timeout=timeout,
                )
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            elapsed_time = time.perf_counter() - start_time

            # Log response status
            if enable_debug_logs:
                self.logger.debug(
                    "api response: %s %s %d %.3fs",
                    method,
                    endpoint,
                    resp.status_code,
                    elapsed_time,
                )

            if resp.status_code != 200:
                elapsed_time = time.perf_counter() - start_time
                self.logger.warning(
                    "%s %s returned status %d in %.3fs. params: %s resp.content: %s",
                    method,
                    endpoint,
                    resp.status_code,
                    elapsed_time,
                    str(json_data),
                    resp.content[:500] if resp.content else "No content",
                )
                # raise TelegramAPIError(
                #     status_code=resp.status_code,
                #     endpoint=endpoint,
                #     content=resp.content,
                #     method=method,
                # )

            return resp

        except requests.exceptions.Timeout as e:
            elapsed_time = time.perf_counter() - start_time
            self.logger.error(
                "Timeout error when making %s request to %s %.3fs: %s",
                method,
                endpoint,
                elapsed_time,
                e,
            )
            raise
        except requests.exceptions.RequestException as e:
            elapsed_time = time.perf_counter() - start_time
            self.logger.error(
                "Request error when making %s request to %s %.3fs: %s",
                method,
                endpoint,
                elapsed_time,
                e,
            )
            raise
        except Exception as e:
            elapsed_time = time.perf_counter() - start_time
            self.logger.error(
                "Unexpected error when making %s request to %s %.3fs: %s",
                method,
                endpoint,
                elapsed_time,
                e,
                exc_info=True,
            )
            raise

    def send_photo(
        self,
        chat_id: int,
        message_thread_id: int,
        photo: Path,
        caption: str,
    ) -> requests.Response:
        """Send a photo to a Telegram chat."""

        truncated_caption = self._truncate_caption(caption)

        with open(photo, "rb") as photo_file:
            resp = self._request(
                method="POST",
                endpoint="sendPhoto",
                data={
                    "chat_id": chat_id,
                    "message_thread_id": message_thread_id,
                    "caption": truncated_caption,
                    # "parse_mode": "MarkdownV2",
                },
                files={"photo": (photo.name, photo_file, "image/png")},
                timeout=10,
            )

        return resp

    def send_message(
        self,
        chat_id: int,
        message: str,
        message_thread_id: int = 0,
        parse_mode: Optional[str] = "MarkdownV2",
        **kwargs,
    ) -> requests.Response:
        """Send a message to a Telegram chat."""
        extra_params = {**kwargs}
        if parse_mode is not None:
            extra_params["parse_mode"] = parse_mode

        resp = self._request(
            method="POST",
            endpoint="sendMessage",
            json_data={
                "text": message,
                "link_preview_options": {"is_disabled": True},
                "chat_id": chat_id,
                "message_thread_id": message_thread_id,
                **extra_params,
            },
            timeout=10,
        )

        return resp

    def send_message_reaction(
        self, chat_id: int, message_id: int, reaction_emoji: str, **kwargs
    ) -> requests.Response:
        """Send a reaction to a message."""
        resp = self._request(
            method="POST",
            endpoint="setMessageReaction",
            json_data={
                "chat_id": chat_id,
                "message_id": message_id,
                "reaction": [{"type": "emoji", "emoji": reaction_emoji}],
                **kwargs,
            },
            timeout=10,
        )

        return resp

    def get_me(self) -> Dict[str, Any]:
        """Get bot information from Telegram."""
        resp = self._request(
            method="GET",
            endpoint="getMe",
            timeout=10,
        )

        if resp.status_code != 200:
            raise ValueError("Failed to get bot info", resp.content)

        return resp.json()

    def get_updates(self, offset: int = 0) -> Dict[str, Any]:
        """Get updates from Telegram."""
        resp = self._request(
            method="GET",
            endpoint="getUpdates",
            params={
                "offset": offset,
                "allowed_updates": '["message", "callback_query"]',
            },
            timeout=300,
            enable_debug_logs=False,
        )

        if resp.status_code != 200:
            raise ValueError("Failed to get updates", resp.content)

        return resp.json()

    def get_chat_member(self, chat_id: int, user_id: int) -> Dict[str, Any]:
        """Get information about a member of a chat (used for permission checks)."""
        resp = self._request(
            method="GET",
            endpoint="getChatMember",
            params={
                "chat_id": chat_id,
                "user_id": user_id,
            },
            timeout=10,
        )

        if resp.status_code != 200:
            raise ValueError("Failed to get chat member", resp.content)

        return resp.json()

    def answer_callback_query(
        self,
        callback_query_id: str,
        text: Optional[str] = None,
        show_alert: bool = False,
    ) -> requests.Response:
        """Answer a callback query.

        Args:
            callback_query_id: Callback query ID
            text: Optional text to show to user
            show_alert: Whether to show as alert or notification

        Returns:
            Response from Telegram API
        """
        resp = self._request(
            method="POST",
            endpoint="answerCallbackQuery",
            json_data={
                "callback_query_id": callback_query_id,
                "text": text,
                "show_alert": show_alert,
            },
            timeout=10,
        )

        return resp

    def edit_message_text(
        self,
        chat_id: int,
        message_id: int,
        text: str,
        parse_mode: Optional[str] = "MarkdownV2",
    ) -> requests.Response:
        """Edit message text.

        Args:
            chat_id: Chat ID
            message_id: Message ID
            text: New message text
            parse_mode: Parse mode for the message (default: MarkdownV2)

        Returns:
            Response from Telegram API
        """
        payload = {
            "chat_id": chat_id,
            "message_id": message_id,
            "text": text,
            "link_preview_options": {"is_disabled": True},
        }

        if parse_mode is not None:
            payload["parse_mode"] = parse_mode

        resp = self._request(
            method="POST",
            endpoint="editMessageText",
            json_data=payload,
            timeout=10,
        )

        self.logger.debug(
            "editMessageText response: status=%d, content=%s",
            resp.status_code,
            resp.content[:200] if resp.content else "No content",
        )

        return resp

    def edit_message_reply_markup(
        self,
        chat_id: int,
        message_id: int,
        reply_markup: Optional[Dict[str, Any]] = None,
    ) -> requests.Response:
        """Edit message reply markup (inline keyboard).

        Args:
            chat_id: Chat ID
            message_id: Message ID
            reply_markup: Inline keyboard markup (None to remove keyboard)

        Returns:
            Response from Telegram API
        """
        payload: Dict[str, Any] = {
            "chat_id": chat_id,
            "message_id": message_id,
        }

        if reply_markup is not None:
            payload["reply_markup"] = reply_markup

        resp = self._request(
            method="POST",
            endpoint="editMessageReplyMarkup",
            json_data=payload,
            timeout=10,
        )

        self.logger.debug(
            "editMessageReplyMarkup response: status=%d, content=%s",
            resp.status_code,
            resp.content[:200] if resp.content else "No content",
        )

        return resp
