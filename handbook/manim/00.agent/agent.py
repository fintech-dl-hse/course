import argparse
import datetime as dt
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openai import OpenAI

try:
    from prompt_toolkit.application import Application
    from prompt_toolkit.filters import Condition
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import HSplit, Layout, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.styles import Style
    from prompt_toolkit.shortcuts import input_dialog
except Exception as exc:  # noqa: BLE001
    print(
        "Для интерактивного режима требуется пакет 'prompt_toolkit'.\n"
        "Установите его: pip install prompt_toolkit",
        file=sys.stderr,
    )
    raise


# ============================
# Data models
# ============================


@dataclass
class IdeaItem:
    text: str
    accepted: bool = False
    rejected: bool = False
    source: str = "model"  # model | critic | user | model_extra
    critic_accept: bool = False
    priority: Optional[int] = None
    critic_reason: Optional[str] = None


# ============================
# OpenAI client helpers
# ============================


def create_openai_client(api_key: Optional[str], base_url: str) -> OpenAI:
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        print("Не найден OPENAI_API_KEY в окружении и не передан через аргумент.", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=key, base_url=base_url)


def call_chat(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_completion_tokens: int = 1500,
    top_p: float = 0.95,
) -> str:
    response = client.chat.completions.create(
        model=model,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
        presence_penalty=0,
        top_p=top_p,
        messages=messages,
    )
    if not response.choices[0].message.content:
        print("No content in response", response)
        breakpoint()

    return response.choices[0].message.content or ""


# ============================
# LLM prompts and parsing
# ============================


def brainstorm_ideas(
    client: OpenAI,
    model: str,
    topic: str,
    num_ideas: int,
    temperature: float,
) -> List[str]:
    # Stage 1: free-form brainstorm without output format constraints
    min_brainstorm = max(num_ideas * 2, num_ideas + 10)
    system_brainstorm = (
        "You are a senior researcher and engineer in Deep Learning. Goal: explain proposed theme. Discuss interesting insights, questions and gotchas around the topic. "
    )
    user_brainstorm = (
        f"Topic: {topic}\n"
        f"Generate at least {min_brainstorm} distinct bullets. Merge near-duplicates, but keep strong diversity.\n"
        "Organize with the following section headers (exactly these):\n"
        "- Overview\n"
        "- Metrics & Evaluation\n"
        "- Failure Modes\n"
        "- Data & Augmentation\n"
        "- Optimization & Training Tricks\n"
        "- Inference & Deployment\n"
        "- Efficiency & Hardware\n"
        "- Real-world Examples\n"
        "- Pitfalls & Misconceptions\n"
        "Within each section, write bullet points. Each bullet should have: a short title, then 1–2 sentences of elaboration.\n"
        "Prefer specific, actionable, and testable angles; avoid generic advice. Highlight novelty, counterintuitive insights, and quantification where possible.\n"
        "Do not output JSON. Do not add conclusions. Keep it compact and skimmable."
    )
    raw_notes = call_chat(
        client,
        model,
        messages=[{"role": "system", "content": system_brainstorm}, {"role": "user", "content": user_brainstorm}],
        temperature=temperature,
        max_completion_tokens=2000,
    )

    print("raw_notes", raw_notes)

    input()

    # Stage 2: structure and select the best ideas in the target format
    system_select = (
        "You are an editorial assistant. "
        "Format notes into list of strings. Each string is a single bullet. Do not modify initial text."
        "Output policy: return a strict JSON arrays of strings only. No markdown, no numbering, no extra text."
    )
    user_select = (
        f"Topic: {topic}\n"
        "Between '---' are the rough brainstorm notes.\n"
        "Answer: strictly a JSON array of strings only. Format: [\"theme 1\", \"theme 2\", ...]. No extra commentary.\n"
        "---\n"
        f"{raw_notes}\n"
        "---"
    ).replace("{N}", str(num_ideas))
    raw = call_chat(
        client,
        model,
        messages=[{"role": "system", "content": system_select}, {"role": "user", "content": user_select}],
        temperature=max(0.2, temperature - 0.2),
        max_completion_tokens=10000,
    )
    ideas = parse_string_array(raw)

    breakpoint()

    return [s for s in ideas if s]

def generate_script(
    client: OpenAI,
    model: str,
    topic: str,
    final_ideas: List[str],
    temperature: float,
) -> str:
    system = (
        "Ты — сценарист вирусных TikTok/Shorts. Пиши динамично, по-сценам, с четкими переходами."
    )
    user = (
        f"Тема: {topic}\n"
        f"Список утвержденных тем/углов: {json.dumps(final_ideas, ensure_ascii=False)}\n\n"
        "Сгенерируй сценарий по-сценно. Требования:\n"
        "- Короткие сцены (1–6 секунд), быстрый темп, крючки в начале\n"
        "- Ясные формулировки, без воды\n"
        "- Возможные визуальные подсказки/действия ведущего\n"
        "- Финальный call-to-action или сильная концовка\n\n"
        "Формат: Markdown. Используй заголовки сцен и краткие реплики."
    )
    return call_chat(
        client,
        model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
        max_completion_tokens=2500,
    )


def parse_string_array(text: str) -> List[str]:
    text = text.strip()
    # Try JSON first
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [str(x).strip() for x in data]
    except Exception:  # noqa: BLE001
        pass
    # Fallback: try to find JSON-like array in text
    match = re.search(r"\[(.|\n)*\]", text)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, list):
                return [str(x).strip() for x in data]
        except Exception:  # noqa: BLE001
            pass
    # Last resort: split by lines
    lines = [ln.strip("-• \t").strip() for ln in text.splitlines()]
    return [ln for ln in lines if ln]


def parse_object_array(text: str) -> List[Dict[str, Any]]:
    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [dict(x) for x in data if isinstance(x, dict)]
    except Exception:  # noqa: BLE001
        pass
    match = re.search(r"\[(.|\n)*\]", text)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, list):
                return [dict(x) for x in data if isinstance(x, dict)]
        except Exception:  # noqa: BLE001
            pass
    return []


# ============================
# Interactive TUI
# ============================


class IdeasTUI:
    def __init__(self, items: List[IdeaItem], on_generate_more: Optional[callable] = None) -> None:
        self.items = items
        self.current_index = 0
        self.on_generate_more = on_generate_more
        self.kb = KeyBindings()
        self.style = Style.from_dict(
            {
                "cursor": "reverse",
                "accepted": "fg:ansigreen",
                "rejected": "fg:ansired",
                "dim": "fg:#666666",
            }
        )
        self._bind_keys()
        self.control = FormattedTextControl(self._render)
        self.window = Window(content=self.control, wrap_lines=False)
        self.app = Application(layout=Layout(HSplit([self.window])), key_bindings=self.kb, style=self.style, full_screen=True)

    def _bind_keys(self) -> None:
        @self.kb.add("up")
        def _up(event) -> None:  # noqa: ANN001
            self.current_index = (self.current_index - 1) % len(self.items)

        @self.kb.add("down")
        def _down(event) -> None:  # noqa: ANN001
            self.current_index = (self.current_index + 1) % len(self.items)

        @self.kb.add(" ")
        def _toggle_accept(event) -> None:  # noqa: ANN001
            item = self.items[self.current_index]
            if item.accepted:
                item.accepted = False
            else:
                item.accepted = True


        @self.kb.add("a")
        @self.kb.add("A")
        def _add(event) -> None:  # noqa: ANN001
            # Ask user to add manual, or leave empty to generate
            text = input_dialog(title="Добавить идеи", text=(
                "Введите новую идею (или несколько через новую строку).\n"
                "Оставьте пустым, чтобы сгенерировать дополнительные идеи."
            )).run()
            if text is None:
                return
            text = text.strip()
            if text:
                for line in [ln.strip() for ln in text.splitlines() if ln.strip()]:
                    self.items.append(IdeaItem(text=line, source="user"))
                return
            # Generate more
            if self.on_generate_more is None:
                return
            n_raw = input_dialog(title="Сгенерировать идеи", text="Сколько новых идей добавить?").run()
            if n_raw is None:
                return
            try:
                n = max(1, int(n_raw))
            except Exception:  # noqa: BLE001
                n = 5
            new_ideas = self.on_generate_more(n)
            for idea in new_ideas:
                self.items.append(IdeaItem(text=idea, source="model_extra"))

        @self.kb.add("enter")
        def _accept_and_exit(event) -> None:  # noqa: ANN001
            event.app.exit(result=self.items)

        @self.kb.add("c-c")
        @self.kb.add("q")
        def _quit(event) -> None:  # noqa: ANN001
            event.app.exit(result=self.items)

    def _render(self) -> List[Tuple[str, str]]:
        lines: List[Tuple[str, str]] = []
        lines.append(("", "Навигация: ↑/↓ — перемещение, [Space] — принять, A — добавить/сгенерировать, Enter — подтвердить (неотмеченные будут отклонены)\n"))
        lines.append(("class:dim", "\n"))
        for idx, item in enumerate(self.items):
            pointer = "> " if idx == self.current_index else "  "
            status = "[✓]" if item.accepted else "[ ]"
            critic_tag = " (critic)" if item.critic_accept else ""
            style = "class:accepted" if item.accepted else ""
            text = f"{pointer}{status} {item.text}{critic_tag}\n"
            if idx == self.current_index:
                lines.append(("class:cursor", text))
            else:
                lines.append((style, text))
        return lines

    def run(self) -> List[IdeaItem]:
        return self.app.run()


# ============================
# Persistence helpers
# ============================


def slugify(value: str, max_len: int = 60) -> str:
    val = value.lower()
    val = re.sub(r"[^a-z0-9\-\s_]+", "", val)
    val = re.sub(r"[\s_]+", "-", val).strip("-")
    return val[:max_len] or "topic"


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def save_text(path: Path, text: str) -> None:
    path.write_text(text)


# ============================
# Main CLI
# ============================


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Интерактивный CLI для генерации сценариев TikTok")
    parser.add_argument("--topic", type=str, help="Тема видео", required=False)
    parser.add_argument("--num-ideas", type=int, default=20, help="Количество идей на этапе брейншторма")
    parser.add_argument("--model", type=str, default="GigaChat/GigaChat-2-Max", help="LLM модель")
    parser.add_argument("--base-url", type=str, default="https://foundation-models.api.cloud.ru/v1", help="Base URL провайдера")
    parser.add_argument("--api-key", type=str, default=None, help="API ключ (иначе возьмется из OPENAI_API_KEY)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Температура генерации")
    parser.add_argument("--outdir", type=str, default=str(Path(__file__).parent), help="Директория для сохранения файлов")
    args = parser.parse_args(list(argv) if argv is not None else None)

    topic = args.topic or input("Введите тему видео: ").strip()
    if not topic:
        print("Тема не может быть пустой.", file=sys.stderr)
        return 2

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    topic_slug = slugify(topic)

    client = create_openai_client(args.api_key, args.base_url)

    # 1) Брейншторм
    initial_ideas = brainstorm_ideas(
        client=client,
        model=args.model,
        topic=topic,
        num_ideas=args.num_ideas,
        temperature=args.temperature,
    )

    if not initial_ideas:
        print("LLM не сгенерировал идеи. Попробуйте увеличить температуру или задать другую тему.", file=sys.stderr)
        return 3

    # 2) Критик: ранжирование и фильтрация
    critic_items = [{"idea": idea, "accept": False, "priority": 1, "reason": "Initial idea"} for idea in initial_ideas]
    # critic_items = critic_rank(
    #     client=client,
    #     model=args.model,
    #     topic=topic,
    #     ideas=initial_ideas,
    #     temperature=max(0.2, args.temperature - 0.2),
    # )

    # Преобразуем к IdeaItem и пометим предложения критика
    items: List[IdeaItem] = []
    seen_texts: set[str] = set()
    for it in critic_items:
        txt = it["idea"]
        if txt in seen_texts:
            continue
        seen_texts.add(txt)
        items.append(
            IdeaItem(
                text=txt,
                accepted=bool(it.get("accept", False)),
                rejected=False,
                source="critic" if bool(it.get("accept", False)) else "model",
                critic_accept=bool(it.get("accept", False)),
                priority=it.get("priority"),
                critic_reason=it.get("reason"),
            )
        )

    # Добавим любые идеи, которых нет в ранжировании (на случай несовпадений)
    for txt in initial_ideas:
        if txt not in seen_texts:
            items.append(IdeaItem(text=txt, source="model"))
            seen_texts.add(txt)

    # Функция для генерации дополнительных идей из TUI
    def generate_more(n: int) -> List[str]:
        extra = brainstorm_ideas(
            client=client,
            model=args.model,
            topic=topic,
            num_ideas=n,
            temperature=args.temperature,
        )
        return [x for x in extra if x not in seen_texts]

    tui = IdeasTUI(items, on_generate_more=generate_more)
    final_items = tui.run()

    # Подготовим данные для сохранения истории
    user_added = [it.text for it in final_items if it.source == "user"]
    user_rejected = [it.text for it in final_items if not it.accepted]
    final_selected = [it.text for it in final_items if it.accepted]

    history = {
        "timestamp": timestamp,
        "topic": topic,
        "model": args.model,
        "proposed_ideas_initial": initial_ideas,
        "critic_ranked": critic_items,
        "all_ideas_after_user": [
            {
                "text": it.text,
                "accepted": it.accepted,
                "rejected": (not it.accepted),
                "source": it.source,
                "critic_accept": it.critic_accept,
                "priority": it.priority,
            }
            for it in final_items
        ],
        "user_added": user_added,
        "user_rejected": user_rejected,
        "final_selected": final_selected,
    }

    history_path = outdir / f"{timestamp}_ideas_{topic_slug}.json"
    save_json(history_path, history)

    # 3) Генерация сценария
    if not final_selected:
        print("Финальный список пуст. Сценарий не будет сгенерирован.", file=sys.stderr)
        print(f"История сохранена: {history_path}")
        return 0

    script_md = generate_script(
        client=client,
        model=args.model,
        topic=topic,
        final_ideas=final_selected,
        temperature=args.temperature,
    )
    script_path = outdir / f"{timestamp}_script_{topic_slug}.md"
    save_text(script_path, script_md)

    print(f"История сохранена: {history_path}")
    print(f"Сценарий сохранен: {script_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

