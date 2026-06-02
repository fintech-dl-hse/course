"""Generator for Seminar 12 notebook: Agents, tool calling, code agents.

Run:
    ~/miniconda3/envs/audio/bin/python seminars/12_agents/build_notebook.py

Produces 12_agents.ipynb with properly formatted cells (each line is a
separate array element, as required by CLAUDE.md).
"""
import json
import os

CELLS = []


def md(text: str) -> None:
    """Append a markdown cell from a raw string."""
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": _src(text)})


def code(text: str) -> None:
    """Append a code cell from a raw string."""
    CELLS.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _src(text),
    })


def _src(text: str):
    """Split text into a Jupyter source array (each line keeps its \\n)."""
    text = text.strip("\n")
    lines = text.split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


# ---------------------------------------------------------------------------
# Cell 0: Colab badge
# ---------------------------------------------------------------------------
md(
    '<a target="_blank" href="https://colab.research.google.com/github/fintech-dl-hse/course/blob/main/seminars/12_agents/12_agents.ipynb">\n'
    '  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>\n'
    "</a>"
)

# Title
md("# Агенты, tool calling и кодовые агенты")

md(
    '<img src="static/agents_for_loop_meme.jpg" width=600 />\n'
    "\n"
    "*Спойлер семинара: под капотом «агент» — это во многом обычный `while`-цикл вокруг вызова LLM.*"
)

md("---")

# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------
md(
    """
## План семинара

Мы уже умеем запускать LLM (сем. 10) и оценивать/ускорять их (сем. 11). Сегодня дадим модели **руки** — инструменты — и **цикл**, который превращает «вызов модели» в **агента**.

1. **Tool calling** — system prompt, формат тулзов (на примере Qwen), парсинг аргументов и исполнение.
2. **Workflow vs Agent** — чем отличаются по степени неопределённости. `Agent = LLM + Harness`. Бенчмарки агентов (SWE-bench и компания).
3. **Свой мини-агент (openclaw)** — собираем agent loop с файловыми инструментами своими руками.
4. **Кодовые агенты** — харнессы (Claude Code / Codex SDK, opencode, pi), базовые тулзы, память, plan mode; как этим пользоваться на практике.
5. **MCP** — это просто «вызов тула по сети».
6. **Как оптимизировать харнесс** — эволюция и рефлексия по трейсам; сходимся на GEPA.
7. **Блиц** — закрепляем: workflow vs agent и связка модель↔харнесс.

> **Главная мысль семинара:** агент — это не «умная модель», а **модель внутри харнесса**. Качество агента = качество модели × качество харнесса, и разделить их нельзя.

> **Терминология.** Говорим **«агентный»** (агентный паттерн, агентный бенч, агентная задача), а не «агентский». Чтобы запомнить — рифмуем с DL-словом: **агентный — как градиентный** (оба на «-ентный»).
"""
)

# ---------------------------------------------------------------------------
# Block 0: setup
# ---------------------------------------------------------------------------
md(
    """
## 0. Recap и setup

Напоминание из прошлых семинаров: LLM — это функция `messages -> текст`. Сама по себе она не умеет ничего, кроме генерации токенов. Чтобы она «что-то сделала» (узнала погоду, прочитала файл, посчитала), мы:

1. описываем доступные **инструменты (tools)**;
2. модель в ответ генерирует **запрос на вызов** инструмента;
3. **наш код** исполняет вызов и возвращает результат модели;
4. повторяем, пока задача не решена.

В качестве модели используем **реальную** Qwen через [Cloud.ru Foundation Models](https://cloud.ru/docs/foundation-models/ug/index) — это OpenAI-совместимый API. Нужен только ключ в переменной окружения `API_KEY` (получить в личном кабинете Cloud.ru) и пакет `openai`. GPU не требуется — модель крутится на стороне облака.
"""
)

code(
    """
# В Colab при необходимости раскомментируйте:
# !pip install -q openai

import getpass
import json
import os
import re
import textwrap
from dataclasses import dataclass, field
from typing import Any, Callable

import random

random.seed(0)

# --- Конфиг Cloud.ru Foundation Models (OpenAI-совместимый API) ---
# Документация: https://cloud.ru/docs/foundation-models/ug/index
CLOUDRU_BASE_URL = "https://foundation-models.api.cloud.ru/v1"
CLOUDRU_MODEL = "Qwen/Qwen3-Coder-Next"

# Ключ берём из переменной окружения API_KEY (получить в ЛК Cloud.ru).
# Если не задан — спросим интерактивно с маскированием ввода (как пароль).
if not os.environ.get("API_KEY"):
    try:
        os.environ["API_KEY"] = getpass.getpass(
            "Введите API-ключ Cloud.ru (ввод скрыт; Enter — пропустить, демо тогда не запустятся): "
        ).strip()
    except Exception:  # noqa: BLE001
        pass  # неинтерактивный запуск (nbconvert): остаёмся без ключа

if os.environ.get("API_KEY"):
    print("Ключ Cloud.ru принят. Модель по умолчанию:", CLOUDRU_MODEL)
else:
    print("⚠️  Ключ не задан — демо с вызовом модели будут пропущены.")
"""
)

# ---------------------------------------------------------------------------
# Block 1: Tool calling
# ---------------------------------------------------------------------------
md(
    """
## 1. Tool calling — как дать LLM руки

**Tool call** (он же function calling) — это соглашение: модель вместо обычного текста выдаёт структурированный запрос «вызови функцию `name` с аргументами `args`». Дальше работает уже не модель, а наш код.

Полный цикл одного tool call:

```md
система: вот список инструментов (имя, описание, схема аргументов)
user:    "какая погода в Париже?"
model:   <tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>
  ── наш код парсит запрос, вызывает get_weather("Paris") ──
tool:    "Paris: +18°C, ясно"
model:   "В Париже сейчас +18°C и ясно."
```

Три вещи, которые делает **наш код** (харнесс), а не модель:
- **system prompt**: рассказываем модели, какие тулы есть и в каком формате звать;
- **parsing**: вытаскиваем `name`/`arguments` из ответа (и валидируем их);
- **execution**: реально вызываем функцию и кладём результат обратно в диалог.

<img src="static/tool_calling_cycle.png" width=700 />
"""
)

md(
    """
### 1.1 Что вообще стоит делать инструментом?

Хорошие кандидаты в тулы — всё, что модель делает **плохо, недетерминированно или вообще не может**:

| Делаем тулом ✅ | Оставляем модели ❌ |
|---|---|
| Точные вычисления (калькулятор, SQL) | Рассуждение, планирование |
| Доступ к свежим/приватным данным (поиск, БД, API) | Формулировка ответа |
| Действия с побочными эффектами (запись файла, отправка письма) | Выбор, *какой* тул позвать |
| Детерминированные операции (парсинг, конвертация) | Интерпретация результата тула |

Правило: **тул — это узкая, проверяемая, желательно идемпотентная операция**. Чем уже контракт, тем надёжнее агент.
"""
)

md(
    """
#### ❓ **Вопрос**: зачем вообще много узких тулов, если один тул `bash` (или `python`) решает практически любую задачу?

<details>

<summary><strong>Ответ</strong></summary>

И то, и другое — рабочие стратегии, это компромисс между гибкостью и контролем.</br>

Универсальный тул (`bash`/`python`): максимум гибкости, не надо предугадывать все операции — сильная модель сама пишет команду. Поэтому продвинутые кодовые агенты (Claude Code, Codex) держатся на `bash` + правках файлов.</br>

Узкие тулы: их легко ограничить и потребовать подтверждение на опасных операциях (безопасность); чёткий контракт оставляет меньше способов ошибиться (надёжность); дискретные вызовы удобно логировать и кэшировать (наблюдаемость).</br>

Вывод: чем способнее модель и выше доверие к среде — тем выгоднее универсальный тул; чем выше цена ошибки — тем нужнее узкие тулзы как рамки. На практике берут и то, и другое.

</details>
"""
)

md(
    """
#### ❓ **Вопрос**: пусть тулзы узкие — но что плохого в том, чтобы подключить агенту сразу 100500 инструментов «на все случаи жизни»?

<details>

<summary><strong>Ответ</strong></summary>

Тулзы не бесплатны: **схема каждого висит в контексте** (имя, описание, параметры) на каждом шаге — сотни тулзов раздувают промпт, жгут токены и вытесняют полезное.</br>
Хуже того, чем больше похожих инструментов, тем чаще модель **выбирает не тот** или путается в близких по смыслу вызовах — точность выбора падает.</br>
Поэтому подключают **минимальный релевантный набор** (а лишние тулзы/MCP/скиллы выключают) — ровно тот же принцип, что «не плоди скиллы» в блоке про эффективное использование контекста.

</details>
"""
)

code(
    '''
# --- Определяем инструменты как обычные python-функции ---
# Docstring и type hints важны: из них строится схема, которую видит модель.

def get_weather(city: str) -> str:
    """Вернуть текущую погоду в городе.

    Args:
        city: Название города на английском, например "Paris".
    """
    # В реальности тут был бы HTTP-запрос к weather API.
    fake = {"paris": "+18°C, ясно", "moscow": "+9°C, дождь", "tokyo": "+24°C, облачно"}
    return f"{city}: " + fake.get(city.lower(), "нет данных")


def calculator(expression: str) -> str:
    """Посчитать арифметическое выражение.

    Args:
        expression: Выражение на python-арифметике, например "2 + 2 * 10".
    """
    # eval только над арифметикой — в проде так делать нельзя, нужен безопасный парсер.
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "Ошибка: недопустимые символы"
    try:
        return str(eval(expression))  # noqa: S307 (демо)
    except Exception as exc:  # noqa: BLE001
        return f"Ошибка: {exc}"


# Реестр инструментов: имя -> функция
TOOLS: dict[str, Callable[..., str]] = {
    "get_weather": get_weather,
    "calculator": calculator,
}

# JSON-схемы инструментов (то, что отдаётся модели). Формат — OpenAI/JSON-Schema.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Вернуть текущую погоду в городе.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "Город (англ.)"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Посчитать арифметическое выражение.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
]

print("Инструментов зарегистрировано:", len(TOOLS))
'''
)

md(
    """
### 1.2 Формат тулзов у Qwen

Разные модели обучены на разный «диалект» tool calling. Qwen (как и многие на базе Hermes) использует теги `<tool_call>...</tool_call>` с JSON внутри:

```md
<tool_call>
{"name": "get_weather", "arguments": {"city": "Paris"}}
</tool_call>
```

Описание инструментов кладётся в **system prompt**. Когда мы пользуемся `transformers`, это делает `tokenizer.apply_chat_template(..., tools=...)` за нас. Но чтобы понять, что «под капотом», соберём system prompt руками.
"""
)

md(
    """
### Qwen3 Chat Template


- [Playground](https://huggingface.co/spaces/huggingfacejs/chat-template-playground?modelId=Qwen%2FQwen3-235B-A22B)
- [Blog](https://huggingface.co/blog/qwen-3-chat-template-deep-dive)


<div style="font-family: ui-monospace, 'JetBrains Mono', Menlo, Consolas, monospace; font-size: 13px; line-height: 1.5;">

<div style="border: 1px solid #ddd6fe; border-left: 4px solid #7c3aed; background: #faf5ff; padding: 10px 14px; border-radius: 6px; margin: 8px 0; color: #1f2937;">
<div style="font-size: 11px; letter-spacing: 0.05em; color: #7c3aed; font-weight: 700; text-transform: uppercase; margin-bottom: 6px;">▌ SYSTEM</div>
<div style="color: #9ca3af; font-size: 11px;">&lt;|im_start|&gt;<span style="color:#7c3aed; font-weight:600;">system</span></div>
<div>You are a helpful assistant that can use tools to get information for the user.</div>

<div style="color: #b45309; font-weight: 700; margin-top: 10px;"># Tools</div>
<div style="margin-top: 4px;">You may call one or more functions to assist with the user query.</div>
<div style="margin-top: 4px;">You are provided with function signatures within <span style="color:#0891b2; font-weight:600;">&lt;tools&gt;&lt;/tools&gt;</span> XML tags:</div>
<div style="color:#0891b2; font-weight:600; margin-top: 4px;">&lt;tools&gt;</div>
<pre style="margin: 4px 0; padding: 8px 10px; background: #ffffff; border: 1px solid #e5e7eb; border-radius: 4px; white-space: pre-wrap; word-break: break-word; color: #111827; font-size: 12px;">{"name": "get_weather", "description": "Get current weather information for a location", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit of temperature to use"}}, "required": ["location"]}}</pre>
<div style="color:#0891b2; font-weight:600;">&lt;/tools&gt;</div>
<div style="margin-top: 4px;">For each function call, return a json object with function name and arguments within <span style="color:#ea580c; font-weight:600;">&lt;tool_call&gt;&lt;/tool_call&gt;</span> XML tags:</div>
<div style="color:#ea580c; font-weight:600; margin-top: 4px;">&lt;tool_call&gt;</div>
<pre style="margin: 4px 0; padding: 8px 10px; background: #ffffff; border: 1px solid #e5e7eb; border-radius: 4px; color: #111827; font-size: 12px;">{"name": &lt;function-name&gt;, "arguments": &lt;args-json-object&gt;}</pre>
<div style="color:#ea580c; font-weight:600;">&lt;/tool_call&gt;</div>
<div style="color: #9ca3af; font-size: 11px; margin-top: 4px;">&lt;|im_end|&gt;</div>
</div>

<div style="border: 1px solid #bbf7d0; border-left: 4px solid #16a34a; background: #f0fdf4; padding: 10px 14px; border-radius: 6px; margin: 8px 0; color: #1f2937;">
<div style="font-size: 11px; letter-spacing: 0.05em; color: #16a34a; font-weight: 700; text-transform: uppercase; margin-bottom: 6px;">▌ USER</div>
<div style="color: #9ca3af; font-size: 11px;">&lt;|im_start|&gt;<span style="color:#16a34a; font-weight:600;">user</span></div>
<div>What's the weather like in New York?</div>
<div style="color: #9ca3af; font-size: 11px;">&lt;|im_end|&gt;</div>
</div>

<div style="border: 1px solid #bfdbfe; border-left: 4px solid #2563eb; background: #eff6ff; padding: 10px 14px; border-radius: 6px; margin: 8px 0; color: #1f2937;">
<div style="font-size: 11px; letter-spacing: 0.05em; color: #2563eb; font-weight: 700; text-transform: uppercase; margin-bottom: 6px;">▌ ASSISTANT <span style="color:#ea580c; font-weight:500; text-transform: none; letter-spacing: 0;">→ tool call</span></div>
<div style="color: #9ca3af; font-size: 11px;">&lt;|im_start|&gt;<span style="color:#2563eb; font-weight:600;">assistant</span></div>
<div>I'll check the current weather in New York for you.</div>
<div style="color:#ea580c; font-weight:600; margin-top: 4px;">&lt;tool_call&gt;</div>
<pre style="margin: 4px 0; padding: 8px 10px; background: #ffffff; border: 1px solid #e5e7eb; border-radius: 4px; color: #111827; font-size: 12px;">{"name": "get_weather", "arguments": {"location": "New York", "unit": "celsius"}}</pre>
<div style="color:#ea580c; font-weight:600;">&lt;/tool_call&gt;</div>
<div style="color: #9ca3af; font-size: 11px;">&lt;|im_end|&gt;</div>
</div>

<div style="border: 1px solid #bbf7d0; border-left: 4px solid #16a34a; background: #f0fdf4; padding: 10px 14px; border-radius: 6px; margin: 8px 0; color: #1f2937;">
<div style="font-size: 11px; letter-spacing: 0.05em; color: #16a34a; font-weight: 700; text-transform: uppercase; margin-bottom: 6px;">▌ USER <span style="color:#0891b2; font-weight:500; text-transform: none; letter-spacing: 0;">← tool response</span></div>
<div style="color: #9ca3af; font-size: 11px;">&lt;|im_start|&gt;<span style="color:#16a34a; font-weight:600;">user</span></div>
<div style="color:#0891b2; font-weight:600; margin-top: 4px;">&lt;tool_response&gt;</div>
<pre style="margin: 4px 0; padding: 8px 10px; background: #ffffff; border: 1px solid #e5e7eb; border-radius: 4px; color: #111827; font-size: 12px;">{"temperature": 22, "condition": "Sunny", "humidity": 45, "wind_speed": 10}</pre>
<div style="color:#0891b2; font-weight:600;">&lt;/tool_response&gt;</div>
<div style="color: #9ca3af; font-size: 11px;">&lt;|im_end|&gt;</div>
</div>

<div style="border: 1px solid #bfdbfe; border-left: 4px solid #2563eb; background: #eff6ff; padding: 10px 14px; border-radius: 6px; margin: 8px 0; color: #1f2937;">
<div style="font-size: 11px; letter-spacing: 0.05em; color: #2563eb; font-weight: 700; text-transform: uppercase; margin-bottom: 6px;">▌ ASSISTANT</div>
<div style="color: #9ca3af; font-size: 11px;">&lt;|im_start|&gt;<span style="color:#2563eb; font-weight:600;">assistant</span></div>
<div>The weather in New York is currently sunny with a temperature of 22°C. The humidity is at 45% with a wind speed of 10 km/h. It's a great day to be outside!</div>
<div style="color: #9ca3af; font-size: 11px;">&lt;|im_end|&gt;</div>
</div>

<div style="border: 1px solid #bbf7d0; border-left: 4px solid #16a34a; background: #f0fdf4; padding: 10px 14px; border-radius: 6px; margin: 8px 0; color: #1f2937;">
<div style="font-size: 11px; letter-spacing: 0.05em; color: #16a34a; font-weight: 700; text-transform: uppercase; margin-bottom: 6px;">▌ USER</div>
<div style="color: #9ca3af; font-size: 11px;">&lt;|im_start|&gt;<span style="color:#16a34a; font-weight:600;">user</span></div>
<div>Thanks! What about Boston?</div>
<div style="color: #9ca3af; font-size: 11px;">&lt;|im_end|&gt;</div>
</div>

<div style="border: 1px dashed #93c5fd; border-left: 4px solid #2563eb; background: #eff6ff; padding: 10px 14px; border-radius: 6px; margin: 8px 0; color: #1f2937;">
<div style="font-size: 11px; letter-spacing: 0.05em; color: #2563eb; font-weight: 700; text-transform: uppercase; margin-bottom: 6px;">▌ ASSISTANT <span style="color:#9ca3af; font-weight:500; text-transform: none; letter-spacing: 0;">— модель продолжает генерацию отсюда</span></div>
<div style="color: #9ca3af; font-size: 11px;">&lt;|im_start|&gt;<span style="color:#2563eb; font-weight:600;">assistant</span></div>
<div style="color:#9ca3af; font-style: italic;">▍</div>
</div>

</div>

"""
)

md(
    """
Выше — схема диалога. Соберём **то же самое реальным токенизатором Qwen3**: передаём наши python-функции в `tools=`, а весь `# Tools`-блок системного промта (JSON-схемы + формат `<tool_call>`) строит сам чат-темплейт. Скачивается только **токенизатор** (без весов модели) — это быстро.
"""
)

code(
    '''
def render_qwen3_chat_template(model_name: str = "Qwen/Qwen3-0.6B") -> str | None:
    """Показать РЕАЛЬНЫЙ рендер чат-темплейта Qwen3 с нашими тулзами.

    tools=[...] — список python-функций; transformers строит JSON-схему из
    docstring и type hints (те же get_weather/calculator из блока 1.1).
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("Нет transformers (`pip install transformers`) — пропускаем рендер.")
        return None
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
    except Exception as exc:  # noqa: BLE001  (нет сети/доступа к HuggingFace)
        print(f"Не удалось загрузить токенизатор ({exc}); нужен доступ к HuggingFace.")
        return None

    # Полный диалог: system + user + ответ-tool_call + результат тула.
    messages = [
        {"role": "system",
         "content": "You are a helpful assistant that can use tools to get information for the user."},
        {"role": "user", "content": "What's the weather like in New York?"},
        {"role": "assistant", "content": "I'll check the current weather in New York for you.",
         "tool_calls": [{"type": "function",
                         "function": {"name": "get_weather", "arguments": {"city": "New York"}}}]},
        {"role": "tool", "name": "get_weather", "content": "New York: +22C, sunny"},
    ]
    text = tok.apply_chat_template(
        messages,
        tools=[get_weather, calculator],   # реальные тулзы -> реальный # Tools блок
        add_generation_prompt=True,        # дописать приглашение ответить ассистенту
        tokenize=False,                    # вернуть строку, а не token ids
    )
    print(text)
    return text


_ = render_qwen3_chat_template()
'''
)

code(
    '''
SYSTEM_TEMPLATE = """Ты — ассистент с доступом к инструментам.

Доступные инструменты (JSON-схемы):
{tools_json}

Чтобы вызвать инструмент, верни блок ровно такого вида (можно несколько подряд):
<tool_call>
{{"name": "имя_инструмента", "arguments": {{...}}}}
</tool_call>

Если инструменты больше не нужны — просто ответь пользователю текстом."""


def render_system_prompt(schemas: list[dict]) -> str:
    """Собрать system prompt со списком инструментов (Qwen-style)."""
    tools_json = json.dumps(schemas, ensure_ascii=False, indent=2)
    return SYSTEM_TEMPLATE.format(tools_json=tools_json)


print(render_system_prompt(TOOL_SCHEMAS)[:600], "...")
'''
)

code(
    '''
# --- Парсинг: вытащить запросы на вызов из текста модели ---
TOOL_CALL_RE = re.compile(r"<tool_call>\\s*(\\{.*?\\})\\s*</tool_call>", re.DOTALL)


def parse_tool_calls(text: str) -> list[dict]:
    """Найти все <tool_call>{...}</tool_call> и распарсить JSON.

    Возвращает список dict вида {"name": ..., "arguments": {...}}.
    Невалидный JSON пропускаем (на практике — повод вернуть модели ошибку).
    """
    calls = []
    for match in TOOL_CALL_RE.finditer(text):
        try:
            calls.append(json.loads(match.group(1)))
        except json.JSONDecodeError:
            continue
    return calls


demo = 'Сейчас проверю. <tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
print(parse_tool_calls(demo))
'''
)

code(
    '''
# --- Исполнение: вызвать функции и собрать результаты ---

def execute_tool_calls(calls: list[dict]) -> list[dict]:
    """Исполнить распарсенные tool calls. Вернуть tool-сообщения для диалога."""
    results = []
    for call in calls:
        name = call.get("name")
        args = call.get("arguments", {}) or {}
        if name not in TOOLS:
            content = f"Ошибка: неизвестный инструмент '{name}'"
        else:
            try:
                content = TOOLS[name](**args)
            except TypeError as exc:
                content = f"Ошибка аргументов: {exc}"
        results.append({"role": "tool", "name": name, "content": content})
    return results


print(execute_tool_calls(parse_tool_calls(demo)))
'''
)

md(
    """
### 1.3 Подключаем реальную модель (Cloud.ru Foundation Models)

Возьмём настоящую модель `Qwen/Qwen3-Coder-Next` через [Cloud.ru Foundation Models](https://cloud.ru/docs/foundation-models/ug/index). Она обучена на tool calling в формате `<tool_call>...</tool_call>`, поэтому наш самодельный харнесс (system prompt + regex-парсер + исполнитель) работает с ней напрямую.

Завернём вызов API в класс с методом `.generate(messages) -> str` — такой интерфейс ждёт `run_tool_loop`, поэтому харнесс не меняется. Результат инструмента возвращаем модели user-сообщением в обёртке `<tool_response>...</tool_response>` — так его ждёт chat template Qwen (нативное поле `tool_calls` не используем: парсим теги из текста сами).
"""
)

code(
    '''
from openai import OpenAI


class CloudRuLLM:
    """Реальная модель поверх Cloud.ru Foundation Models (OpenAI-совместимый API).

    Интерфейс — .generate(messages) -> str, поэтому весь харнесс
    (run_tool_loop / run_agent) работает без изменений.
    """

    def __init__(self, model: str = CLOUDRU_MODEL, temperature: float = 0.3,
                 max_tokens: int = 1024):
        self.client = OpenAI(api_key=os.environ["API_KEY"], base_url=CLOUDRU_BASE_URL)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @staticmethod
    def _to_api(msg: dict) -> dict:
        """Привести наше внутреннее сообщение к формату chat-completions API."""
        # Tool calling делаем «вручную» (теги <tool_call> в тексте), поэтому
        # результат инструмента возвращаем как user-сообщение в обёртке
        # <tool_response> — так его ждёт chat template Qwen.
        if msg["role"] == "tool":
            return {"role": "user",
                    "content": f"<tool_response>\\n{msg['content']}\\n</tool_response>"}
        return {"role": msg["role"], "content": msg["content"]}

    def generate(self, messages: list[dict]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[self._to_api(m) for m in messages],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=0.95,
        )
        return resp.choices[0].message.content or ""


def make_llm(**kwargs) -> "CloudRuLLM | None":
    """Вернуть модель Cloud.ru, либо None, если ключа нет (демо тогда пропускаем)."""
    if not os.environ.get("API_KEY"):
        print("API_KEY не задан — пропускаем демо (задайте ключ в ячейке setup).")
        return None
    return CloudRuLLM(**kwargs)


print("LLM-обёртка готова: CloudRuLLM.")
'''
)

code(
    '''
def run_tool_loop(llm, user_msg: str, schemas: list[dict], max_steps: int = 5, verbose: bool = True):
    """Минимальный цикл tool calling: модель -> парсинг -> исполнение -> повтор."""
    messages = [
        {"role": "system", "content": render_system_prompt(schemas)},
        {"role": "user", "content": user_msg},
    ]
    for step in range(max_steps):
        reply = llm.generate(messages)
        messages.append({"role": "assistant", "content": reply})
        calls = parse_tool_calls(reply)
        if verbose:
            print(f"[step {step}] assistant: {reply!r}")
        if not calls:
            return reply  # модель ответила текстом — задача решена
        tool_msgs = execute_tool_calls(calls)
        for tm in tool_msgs:
            if verbose:
                print(f"[step {step}] tool {tm['name']}: {tm['content']!r}")
        messages.extend(tool_msgs)
    return "Достигнут лимит шагов"


llm = make_llm()
if llm is not None:
    final = run_tool_loop(llm, "Погода в Париже и сколько 2+2*10?", TOOL_SCHEMAS)
    print("\\nИТОГ:", final)
'''
)

md(
    """
#### ❓ **Вопрос**: зачем вообще возвращать результат тула обратно модели отдельным сообщением? Почему не подставить ответ функции сразу пользователю?

<details>

<summary><strong>Ответ</strong></summary>

Потому что результат тула — это **промежуточные данные**, а не финальный ответ.</br>
Модель должна их *интерпретировать*: выбрать нужное, объединить несколько вызовов, обработать ошибку тула и решить, звать ли ещё инструменты.</br>
Если отдать сырой вывод функции пользователю — мы превратим агента обратно в жёсткий workflow и потеряем гибкость.

</details>
"""
)

md(
    """
### 1.4 Альтернатива: локальный Qwen через transformers (опционально)

Если хочется крутить модель **локально** (а не через Cloud.ru), всё то же самое, но шаблон с инструментами строит сам `transformers`. Передаём `tools=` как список python-функций — схема собирается из docstring и type hints.
"""
)

code(
    '''
# ОПЦИОНАЛЬНО: запустится только при наличии torch+transformers (лучше GPU).
def try_qwen_tool_call(user_msg: str, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("transformers/torch не установлены — пропускаем (используйте Cloud.ru выше).")
        return None

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    messages = [{"role": "user", "content": user_msg}]
    # tools — список callables; transformers сам построит JSON-схему из docstring.
    text = tok.apply_chat_template(
        messages, tools=[get_weather, calculator],
        add_generation_prompt=True, tokenize=False,
    )
    inputs = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=128)
    reply = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print("Qwen ответила:\\n", reply)
    print("\\nРаспарсили:", parse_tool_calls(reply))
    return reply


# try_qwen_tool_call("Какая погода в Москве?")
print("Ячейка с Qwen готова — раскомментируйте вызов, если есть окружение.")
'''
)

md(
    """
### 1.5 Нативный tool calling через API (структурно)

Выше мы парсили `<tool_call>` регулярками — так видно, что «под капотом». Но OpenAI-совместимый API (включая Cloud.ru) умеет возвращать вызовы инструментов **структурно**: передаём `tools=` со схемами, а в ответе получаем готовое поле `tool_calls` (имя + JSON-аргументы) — парсить регэкспами не нужно. Это и есть «промышленный» путь поверх того же Cloud.ru.
"""
)

code(
    '''
def cloudru_native_tool_call(user_msg: str, model: str = CLOUDRU_MODEL):
    """Структурный tool calling: tools= на входе, готовое поле tool_calls на выходе."""
    if not os.environ.get("API_KEY"):
        print("API_KEY не задан — пропускаем (нужен ключ Cloud.ru).")
        return None

    client = OpenAI(api_key=os.environ["API_KEY"], base_url=CLOUDRU_BASE_URL)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": user_msg}],
        tools=TOOL_SCHEMAS,        # те же схемы, что мы отдавали вручную в system prompt
        temperature=0.3,
    )
    msg = resp.choices[0].message
    if msg.tool_calls:
        for tc in msg.tool_calls:
            # tc.function.name / tc.function.arguments — уже структурно, без regex.
            print(f"tool_call: {tc.function.name}({tc.function.arguments})")
    else:
        print("Модель ответила текстом:", msg.content)
    return msg


cloudru_native_tool_call("Какая погода в Москве?")
'''
)

# ---------------------------------------------------------------------------
# Block 2: Workflow vs Agent
# ---------------------------------------------------------------------------
md(
    """
## 2. Workflow vs Agent

Простой вызов LLM в коде — это **ещё не агент**. И tool calling сам по себе — тоже не обязательно агент.

> **Agent = LLM + Harness**

<img src="static/agent_equals_llm_harness.png" width=700 />

Разница между workflow и агентом — в **степени неопределённости** и в том, **кто решает, что делать дальше**.

| | Workflow | Agent |
|---|---|---|
| Управление потоком | задано кодом заранее | модель решает на каждом шаге |
| Шаги известны? | да, фиксированный граф | нет, выясняются по ходу |
| Кол-во итераций | детерминированное | заранее неизвестно |
| Пример | «суммаризируй → переведи → отправь» | «разберись с этим багом» |
| Надёжность | высокая, предсказуемая | ниже, зато гибкая |

Эвристика: если вы можете нарисовать блок-схему заранее и она не меняется от входа — это **workflow**. Если модель сама выбирает следующий шаг в цикле «подумал → сделал → посмотрел на результат» — это **agent**.

<img src="static/workflow_vs_agent.png" width=600 />

**Harness** (обвязка) — это всё вокруг модели: цикл, набор тулзов, system prompt, управление контекстом/памятью, лимиты шагов, обработка ошибок, логирование.
"""
)

md(
    """
#### ❓ **Вопрос**: мы написали `for city in cities: llm("погода в " + city)`. Это агент?

<details>

<summary><strong>Ответ</strong></summary>

Нет. Это **workflow**: поток управления (`for` по списку городов) задан нами заранее, модель не принимает решений о следующем шаге.</br>
Агентом это станет, если модель *сама* решает, какие города опросить, нужно ли звать другие тулы и когда остановиться.

</details>
"""
)

md(
    """
### 2.1 Бенчмарки агентов

Агентов оценивают не по перплексии, а по **решённым задачам в реальной среде**. Самое популярное — кодовые бенчи на базе реальных GitHub-issue:

| Бенч | Что это | Особенность |
|---|---|---|
| **SWE-bench** | ~2.3k реальных issue+PR из python-репозиториев; агент должен дать патч, проходящий тесты | большой, но часть задач «протекла» в претрейн |
| **SWE-bench Verified** | 500 задач, вручную провалидированных людьми (решаемость, корректность тестов) | «чистое» подмножество, текущий стандарт сравнения |
| **SWE-rebench** | непрерывно обновляемый набор свежих задач | борется с контаминацией: новые issue, которых не было в обучении |

Что важно понимать про эти числа:
- метрика — **% resolved** (патч прошёл скрытые тесты);
- результат — это связка **модель + харнесс**, а не «голая модель» (модель и харнесс затачиваются друг под друга: GPT↔Codex, Claude↔Claude Code — поэтому чужой харнесс занижает результат);
- бенч на коде, потому что там есть **автоматический верификатор** (тесты) — для большинства агентных задач хорошего верификатора нет, и это главная проблема оценки.
"""
)

md(
    """
#### ❓ **Вопрос**: почему агентные бенчи почти всегда про код, а не про «закажи пиццу» или «спланируй поездку»?

<details>

<summary><strong>Ответ</strong></summary>

Потому что у кода есть **дешёвый автоматический верификатор** — тесты: запустили, прошли/не прошли.</br>
Для «закажи пиццу» проверка успеха требует реального мира или человека-оценщика — это дорого, шумно и плохо масштабируется.</br>
Поэтому прогресс агентов измеряют там, где успех можно проверить программно.

</details>
"""
)

# ---------------------------------------------------------------------------
# Block 3: mini agent (openclaw)
# ---------------------------------------------------------------------------
md(
    """
## 3. Собираем свой мини-агент (openclaw)

Соберём настоящего (маленького) агента: LLM в цикле с файловыми инструментами. Это и есть упрощённый «кодовый агент» — назовём его **openclaw**.

Ключевые отличия от блока 1:
- инструменты **меняют мир** (пишут/читают файлы, исполняют код);
- модель сама решает **последовательность** и **момент остановки** (`finish`);
- есть **guard** на число шагов — иначе агент может зациклиться.
"""
)

code(
    '''
import os
import tempfile

# Песочница: все файловые операции — внутри временной директории.
WORKDIR = tempfile.mkdtemp(prefix="openclaw_")


def _safe_path(name: str) -> str:
    """Не дать агенту выйти за пределы песочницы."""
    path = os.path.realpath(os.path.join(WORKDIR, name))
    if not path.startswith(os.path.realpath(WORKDIR)):
        raise ValueError("Выход за пределы песочницы запрещён")
    return path


def write_file(filename: str, content: str) -> str:
    """Записать content в файл filename (внутри песочницы)."""
    with open(_safe_path(filename), "w") as f:
        f.write(content)
    return f"OK: записано {len(content)} символов в {filename}"


def read_file(filename: str) -> str:
    """Прочитать содержимое файла filename."""
    with open(_safe_path(filename)) as f:
        return f.read()


def list_files(_: str = "") -> str:
    """Показать список файлов в рабочей директории."""
    return ", ".join(sorted(os.listdir(WORKDIR))) or "(пусто)"


def python_eval(expression: str) -> str:
    """Вычислить python-выражение (для арифметики/проверок)."""
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "Ошибка: разрешена только арифметика"
    return str(eval(expression))  # noqa: S307 (демо)


AGENT_TOOLS = {
    "write_file": write_file,
    "read_file": read_file,
    "list_files": list_files,
    "python_eval": python_eval,
}

AGENT_SCHEMAS = [
    {"type": "function", "function": {"name": "write_file",
        "parameters": {"type": "object", "properties": {
            "filename": {"type": "string"}, "content": {"type": "string"}},
            "required": ["filename", "content"]}}},
    {"type": "function", "function": {"name": "read_file",
        "parameters": {"type": "object", "properties": {"filename": {"type": "string"}},
            "required": ["filename"]}}},
    {"type": "function", "function": {"name": "list_files",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "python_eval",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}},
            "required": ["expression"]}}},
]

print("Песочница:", WORKDIR)
print("Инструменты агента:", list(AGENT_TOOLS))
'''
)

code(
    '''
def run_agent(llm, task: str, tools: dict, schemas: list[dict], max_steps: int = 8,
              verbose: bool = True, messages: list[dict] | None = None):
    """Agent loop: модель в цикле дёргает тулзы, пока не вернёт финальный текст.

    Это и есть харнесс: system prompt + цикл + исполнение + лимит шагов.

    Если передать `messages` — продолжаем уже существующий диалог (с историей);
    список мутируется на месте, поэтому вызывающий код видит всю переписку.
    Без `messages` каждый вызов начинает разговор с нуля.
    """
    if messages is None:
        messages = []
    # system prompt (со списком тулзов) добавляем один раз — если его ещё нет.
    if not any(m["role"] == "system" for m in messages):
        system = render_system_prompt(schemas) + (
            "\\n\\nКогда задача выполнена — ответь пользователю обычным текстом без tool_call."
        )
        messages.insert(0, {"role": "system", "content": system})
    messages.append({"role": "user", "content": task})
    for step in range(max_steps):
        reply = llm.generate(messages)
        messages.append({"role": "assistant", "content": reply})
        calls = parse_tool_calls(reply)
        if verbose:
            print(f"[{step}] >>> {reply.strip()[:120]}")
        if not calls:
            return reply
        for call in calls:
            name, args = call.get("name"), call.get("arguments", {}) or {}
            fn = tools.get(name)
            result = fn(**args) if fn else f"Ошибка: нет инструмента {name}"
            if verbose:
                print(f"      tool {name} -> {str(result)[:80]}")
            messages.append({"role": "tool", "name": name, "content": str(result)})
    return "Достигнут лимит шагов"
'''
)

code(
    '''
llm = make_llm()
if llm is not None:
    result = run_agent(
        llm,
        task="Сохрани число 7 в файл n.txt, потом прочитай его и посчитай его квадрат.",
        tools=AGENT_TOOLS,
        schemas=AGENT_SCHEMAS,
    )
    print("\\nИТОГ:", result)
    print("Файлы в песочнице:", list_files())
'''
)

md(
    """
В `run_agent` ушёл тот же `CloudRuLLM` через `make_llm`, а харнесс (цикл, тулзы, лимиты) не изменился — поменялась только модель. Свой openclaw на реальной модели готов.
"""
)

md(
    """
### Поболтать с агентом вживую

Запустите ячейку ниже и давайте openclaw задачи в свободной форме — например:

- `создай файл hello.txt с приветствием и прочитай его обратно`
- `посчитай (123 + 456) * 2 и сохрани результат в answer.txt`
- `покажи, какие файлы лежат в песочнице`

История диалога сохраняется между задачами, поэтому агент помнит предыдущие шаги (можно сказать «а теперь удвой результат» — он поймёт, о чём речь). Песочница (`WORKDIR`) тоже общая, файлы не теряются. Пустая строка или `exit` — выход. Нужен реальный ключ Cloud.ru (см. ячейку setup).
"""
)

code(
    '''
def chat_with_agent(max_steps: int = 8) -> None:
    """Интерактивный диалог с нашим агентом openclaw на реальной модели Cloud.ru.

    Вводите задачи в свободной форме; пустая строка или 'exit' — выход.
    История диалога сохраняется между задачами (общий `messages`), поэтому агент
    помнит предыдущие шаги. Песочница (WORKDIR) тоже общая — файлы не теряются.
    """
    if not os.environ.get("API_KEY"):
        print("Нужен API_KEY Cloud.ru — задайте ключ в ячейке setup и перезапустите.")
        return

    llm = CloudRuLLM()
    messages: list[dict] = []  # одна история на весь диалог -> агент помнит контекст
    print("openclaw готов. Введите задачу (пустая строка или 'exit' — выход).\\n")
    while True:
        try:
            task = input("Вы > ").strip()
        except (EOFError, OSError):
            print("Неинтерактивный запуск — выходим из чата.")
            return
        if not task or task.lower() in {"exit", "quit", "выход"}:
            print("Выходим из чата.")
            return
        # messages мутируется на месте -> следующий запрос видит всю переписку.
        answer = run_agent(
            llm, task, tools=AGENT_TOOLS, schemas=AGENT_SCHEMAS,
            max_steps=max_steps, messages=messages,
        )
        print(f"\\nАгент > {answer}\\n")


chat_with_agent()
'''
)

md(
    """
#### ❓ **Вопрос**: наш `run_agent` — это workflow или агент? А `run_tool_loop` из блока 1?

<details>

<summary><strong>Ответ</strong></summary>

Оба — **агенты** (в маленьком масштабе): модель сама решает, какие тулы и в каком порядке звать и когда остановиться; число итераций заранее неизвестно.</br>
Workflow получился бы, если бы мы жёстко прописали «сначала write_file, потом read_file, потом python_eval» в коде, а модель использовали лишь для заполнения аргументов.

</details>
"""
)

# ---------------------------------------------------------------------------
# Block 4: code agents walkthrough
# ---------------------------------------------------------------------------
md(
    """
## 4. Кодовые агенты: харнессы и тулы (обзор)

Наш openclaw — это «ядро» любого кодового агента. Продакшн-агенты добавляют поверх много инженерии. Бегло пройдёмся по ландшафту и по тому, что внутри.

### 4.1 Харнессы для кодовых агентов

- **Claude Code / Claude Agent SDK** — харнесс Anthropic; SDK позволяет строить своих агентов на том же движке.
- **Codex / Codex SDK** — харнесс OpenAI.
- **opencode**, **pi** и др. — открытые харнессы, можно подключать разные модели.

Все они — вариации одного цикла из блока 3, обвешанного тулзами, памятью и UX. «Сделать своего openclaw» — значит собрать такой цикл под свою модель.
"""
)

md(
    """
#### ❓ **Вопрос**: наш openclaw умеет только `write_file` (перезапись файла **целиком**). В чём подвох на реальной кодовой базе?

<details>

<summary><strong>Ответ</strong></summary>

Перезапись целиком **затирает всё, чего нет в новом тексте**: модель хочет поменять одну строку, а сносит весь файл (включая чужие изменения).</br>
Поэтому настоящие кодовые агенты дают **Edit с точным совпадением** старой строки (`old_string -> new_string`) и требуют сначала **прочитать** файл перед записью — об этом ниже.

</details>
"""
)

md(
    """
Покажем этот подвох и «правильную» правку прямо на тулзах openclaw из блока 3.
"""
)

code(
    '''
# 1) Наивная перезапись целиком затирает всё, чего нет в новом тексте.
write_file("config.py", "DEBUG = True\\nAPI_KEY = 'secret-123'\\n")
print("До:    ", repr(read_file("config.py")))

# Модель хотела поменять только DEBUG, но Write перезаписал файл целиком:
write_file("config.py", "DEBUG = False\\n")
print("После: ", repr(read_file("config.py")), "  <- API_KEY потерян!")
'''
)

code(
    '''
def safe_edit(filename: str, old: str, new: str) -> str:
    """Заменить ровно одно вхождение old на new (как Edit в кодовых агентах).

    Требует, чтобы old встречалась в файле дословно и ровно один раз:
    это страхует от правки не того места и от «галлюцинированных» строк.
    """
    text = read_file(filename)
    if old not in text:
        return "Отклонено: old не найдена дословно (защита от галлюцинаций)"
    if text.count(old) > 1:
        return "Отклонено: old не уникальна — добавьте контекст"
    write_file(filename, text.replace(old, new))
    return "OK: заменено одно вхождение"


# Восстановим файл и поменяем ТОЛЬКО DEBUG через точное совпадение:
write_file("config.py", "DEBUG = True\\nAPI_KEY = 'secret-123'\\n")
print(safe_edit("config.py", "DEBUG = True", "DEBUG = False"))
print("Итог:  ", repr(read_file("config.py")), "  <- API_KEY на месте")

# Галлюцинированная правка (такой строки в файле нет дословно) — отклонена:
print(safe_edit("config.py", "DEBUG=True", "DEBUG = False"))
'''
)

md(
    """
### 4.2 Базовые тулзы: чтение и запись файлов

Здесь больше инженерии, чем кажется:

- **Read** обычно отдаёт файл **с номерами строк** и поддерживает чтение по диапазону (`offset`/`limit`) — чтобы не утопить контекст в огромных файлах.
- **Edit** чаще всего работает через **точное совпадение строки** (`old_string -> new_string`): модель обязана процитировать существующий фрагмент дословно. Это защита от «галлюцинированных» правок и от перезаписи не того места.
- **Write** перезаписывает файл целиком — поэтому хорошие харнессы требуют сначала *прочитать* файл, иначе легко затереть чужие изменения.

Эти ограничения загоняют модель в жёсткие рамки, где она ошибается реже.
"""
)

md(
    """
### 4.3 Управление вниманием: todos, напоминалки, нотификации

Длинные задачи не влезают в один проход, и модель «забывает» план. Решения:

- **TODO-листы**: агент ведёт явный список подзадач и вычёркивает их — это и память, и фокус.
- **Reminders / system-reminder**: харнесс периодически вставляет в контекст напоминание о текущей цели и правилах.
- **System notifications**: уведомления пользователю, когда нужен ввод или задача завершена (особенно для фоновых задач).
"""
)

md(
    """
### 4.4 Слои работы с памятью

Контекст модели конечен, поэтому память выносят наружу слоями:

- **Сессионная** — текущий диалог (самый свежий, но дорогой и теряется при компактификации).
- **Проектная** — файлы вроде `CLAUDE.md`/`AGENTS.md`: правила, команды, договорённости репозитория; подгружаются в начало каждой сессии.
- **Долговременная** — внешнее хранилище фактов (файлы памяти, векторная БД, wiki), куда агент пишет и откуда достаёт по релевантности.
"""
)

md(
    """
### 4.5 Plan mode

**Plan mode** — режим, где агенту запрещены изменяющие действия (правки, коммиты): он только исследует и **предлагает план**, который человек утверждает до исполнения. Это разделяет «думать» и «делать» и резко снижает цену ошибки на больших задачах.
"""
)

# ---------------------------------------------------------------------------
# Block 5: MCP
# ---------------------------------------------------------------------------
md(
    """
## 5. MCP — вызов тула по сети

**MCP (Model Context Protocol)** — идея простая:

> MCP — это **стандартный способ вызвать тул, который живёт в отдельном процессе/на сервере**, а не в твоём коде.

В блоках 1–3 тулзы были обычными python-функциями в том же процессе. MCP выносит их за границу процесса: **MCP-сервер** объявляет инструменты, **MCP-клиент** (агент) их обнаруживает и вызывает по стандартному протоколу (JSON-RPC поверх stdio или HTTP). Один и тот же сервер (например, «доступ к Postgres» или «GitHub») переиспользуется любым MCP-совместимым агентом.

Покажем «тул по сети» на самодельном мини-протоколе без зависимостей — тот же `execute_tool_calls`, но через сериализованный запрос/ответ.
"""
)

code(
    '''
import json


class MiniToolServer:
    """Игрушечный 'MCP-сервер': принимает JSON-запрос, возвращает JSON-ответ.

    В настоящем MCP это JSON-RPC поверх stdio/HTTP; суть — та же граница процесса.
    """

    def __init__(self, tools: dict):
        self._tools = tools

    def handle(self, request_json: str) -> str:
        req = json.loads(request_json)
        method = req.get("method")
        if method == "tools/list":
            return json.dumps({"tools": list(self._tools)})
        if method == "tools/call":
            name = req["params"]["name"]
            args = req["params"].get("arguments", {})
            try:
                result = self._tools[name](**args)
                return json.dumps({"result": result}, ensure_ascii=False)
            except Exception as exc:  # noqa: BLE001
                return json.dumps({"error": str(exc)}, ensure_ascii=False)
        return json.dumps({"error": "unknown method"})


class MiniToolClient:
    """Клиент, который 'по сети' (тут — через строки) зовёт сервер."""

    def __init__(self, server: MiniToolServer):
        self._server = server  # представьте, что это сокет/stdio

    def list_tools(self) -> list[str]:
        return json.loads(self._server.handle(json.dumps({"method": "tools/list"})))["tools"]

    def call(self, name: str, **arguments) -> str:
        req = json.dumps({"method": "tools/call",
                          "params": {"name": name, "arguments": arguments}})
        return self._server.handle(req)


server = MiniToolServer({"get_weather": get_weather, "calculator": calculator})
client = MiniToolClient(server)
print("tools/list ->", client.list_tools())
print("tools/call ->", client.call("get_weather", city="Tokyo"))
'''
)

md(
    """
В реальности вместо `MiniToolServer` — пакет `mcp`, а транспорт — stdio или HTTP. Минимальный настоящий MCP-сервер (запускать локально, не в Colab) выглядит так:

```python
# pip install mcp
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("weather")

@mcp.tool()
def get_weather(city: str) -> str:
    "Вернуть погоду в городе."
    return f"{city}: +18°C, ясно"

if __name__ == "__main__":
    mcp.run()  # общается по stdio; клиент (агент) подключается к процессу
```

Агент (Claude Code, Codex, наш openclaw…) добавляет этот сервер в конфиг — и его тулзы становятся доступны наравне со встроенными.
"""
)

md(
    """
### 5.1 Полезные MCP и готовые системы

Примеры полезных MCP-серверов:

- **ВкусВилл** — MCP для сборки продуктовой корзины: агент зовёт что-то вроде `add_to_cart(product, qty)`, а сама корзина собирается на стороне сервиса. Сценарий «собери корзину по рецепту/на неделю» решается агентом, а оформление заказа остаётся за человеком.
- **GitHub / GitLab** — issues, pull requests, поиск по коду.
- **БД (Postgres и др.)** — выполнить запрос, посмотреть схему таблиц.
- **Браузер / файловая система** — навигация по страницам, чтение и запись файлов.

Не всё нужно собирать руками — есть платформы «из коробки», где агенты, тулзы и MCP уже связаны:

- **[omo.dev](https://omo.dev/)** — пример такой системы.
- Экосистема MCP-серверов подключается к агенту как плагины — один сервер переиспользуется любым MCP-клиентом.
"""
)

md(
    """
#### ❓ **Вопрос**: чем MCP принципиально отличается от того, что мы делали в блоках 1–3 с обычными функциями?

<details>

<summary><strong>Ответ</strong></summary>

Принципиально — **ничем по смыслу**: это всё тот же tool call (имя + аргументы → результат).</br>
Отличие инженерное: тул живёт **за границей процесса** и общается по *стандартному протоколу*. За счёт стандарта один сервер переиспользуется любым MCP-клиентом, а тулзы можно деплоить и обновлять независимо от агента.</br>

По большому счёту **function call, MCP-тул и skill — это одно и то же** под капотом: «дай модели возможность вызвать что-то по имени с аргументами». Различаются лишь упаковкой и границей (функция в процессе → тул по сети → переиспользуемая процедура с инструкциями).

<img src="static/spiderman_function_mcp_skill.jpg" width=500 />

</details>
"""
)

# ---------------------------------------------------------------------------
# Block 6: harness optimization (GEPA)
# ---------------------------------------------------------------------------
md(
    """
## 6. Как оптимизировать харнесс?

Мы весь семинар повторяем: `Agent = LLM + Harness`, и качество = качество модели × качество харнесса. Модель мы (обычно) не трогаем — она дана. А вот харнесс — это куча **текстовых ручек**: system prompt, описания тулзов, инструкции в `CLAUDE.md`, формулировки шагов. Их обычно крутят руками: «поправил промпт → прогнал → вроде лучше». Можно ли это **автоматизировать**?

Два ключевых ингредиента:

- **Рефлексия.** Вместо скалярной награды (как в RL: «0.37, стало лучше или хуже?») даём модели прочитать **её собственные трейсы** — рассуждения, вызовы тулзов, ошибки, фидбэк проверки — и попросить **на естественном языке** объяснить, что пошло не так и как переписать инструкцию. Текст несёт куда больше сигнала, чем одно число.
- **Эволюция.** Держим **популяцию** кандидатов-промптов, мутируем их рефлексией, скрещиваем удачные находки и отбираем лучших — генетический поиск по пространству текстов.
"""
)

md(
    """
### GEPA — Genetic-Pareto

[**GEPA** (Genetic-Pareto, 2025)](https://arxiv.org/pdf/2507.19457) собирает обе идеи в один оптимизатор промптов для составных LLM-систем:

1. **прогнать** систему на нескольких задачах, собрать трейсы (что генерила, какие тулзы звала, где упала);
2. **отрефлексировать** трейсы на естественном языке → диагноз + предложенная правка инструкции;
3. **отобрать** кандидатов по **Парето-фронту**: храним не один «самый лучший» промпт, а фронт тех, кто выигрывает хоть на каких-то задачах — так не схлопываемся в локальный оптимум и комбинируем разные удачные стратегии.

Главный результат: рефлексия по тексту **обгоняет RL**. GEPA бьёт **GRPO** в среднем на ~6% (до +20%), при этом тратит **до 35× меньше прогонов (rollouts)**, и обходит ведущий промпт-оптимизатор **MIPROv2** на 10%+. Мораль семинара: **язык — более богатый обучающий сигнал, чем скалярная награда**, поэтому оптимизировать харнесс можно дёшево и без RL.
"""
)

md(
    """
#### ❓ **Вопрос**: почему рефлексия по трейсам обучает харнесс эффективнее (по числу прогонов), чем RL со скалярной наградой?

<details>

<summary><strong>Ответ</strong></summary>

В RL один прогон даёт **одно число** (reward) — «стало лучше/хуже», но не *почему*. Чтобы выжать сигнал, нужны тысячи прогонов.</br>
Рефлексия же читает **весь трейс** (рассуждения, вызовы тулзов, конкретную ошибку) и формулирует *причину и правку* на естественном языке — из одного прогона извлекается во много раз больше информации. Отсюда и сэмпл-эффективность: до ~35× меньше rollouts при лучшем качестве.</br>
Парето-фронт добавляет устойчивость: вместо одного «чемпиона» хранится набор кандидатов, каждый из которых хорош на своей части задач, — это спасает от схлопывания в локальный оптимум.

</details>
"""
)

# ---------------------------------------------------------------------------
# Quiz + blitz
# ---------------------------------------------------------------------------
md(
    """
## Блиц

Закрепляем материал семинара. Сначала — несколько сценариев на классификацию: **workflow** (поток задан кодом) или **agent** (модель решает в цикле)? Иногда ответ — «зависит». Дальше — короткие вопросы на термины.
"""
)

md(
    """
#### ❓ **Вопрос**: бот для заказа продуктов через чат — workflow или agent?

<details>

<summary><strong>Ответ</strong></summary>

**На самом деле — зависит от формулировки требований.**</br>
Если бот сам *делает* заказы, имеет доступ, например, к камере в холодильнике и сам отвечает за то, *когда* заказать продукты и *какие именно* — это уже полноценный **агент**: высокая неопределённость, решения принимает модель в цикле.</br>
Если же MCP используется лишь для добавления продуктов в корзину (как в примере с ВкусВилл выше), а сам заказ оформляет человек по фиксированному сценарию — это скорее **workflow**.

</details>
"""
)

md(
    """
#### ❓ **Вопрос**: бот для установки напоминалок — workflow или agent?

<details>

<summary><strong>Ответ</strong></summary>

**Workflow** (вырожденный): по сути один tool call — распарсить «напомни во вторник в 9» и создать напоминание. Цикла принятия решений нет, неопределённость минимальна.</br>
Один вызов LLM + один тул — это ещё **не агент**.

</details>
"""
)

md(
    """
#### ❓ **Вопрос**: авторесёрч (агент сам исследует тему и пишет отчёт) — workflow или agent?

<details>

<summary><strong>Ответ</strong></summary>

**Agent**: заранее неизвестно, сколько поисков и чтений понадобится, какие подвопросы возникнут и когда остановиться — модель решает это в цикле «искать → читать → проверять → синтезировать».</br>
Высокая степень неопределённости — характерный признак агентной задачи.

</details>
"""
)

md(
    """
#### ❓ **Вопрос**: `Agent = LLM + ?`

<details>

<summary><strong>Ответ</strong></summary>

`Harness` — обвязка: цикл, тулзы, system prompt, управление памятью/контекстом, лимиты и обработка ошибок.

</details>

#### ❓ **Вопрос**: почему Edit-тулзы требуют точного совпадения старой строки, а не «номер строки + новый текст»?

<details>

<summary><strong>Ответ</strong></summary>

Точное цитирование заставляет модель «видеть» реальный код и страхует от правки не того места и от галлюцинаций; номера строк легко устаревают после первой же правки.

</details>

#### ❓ **Вопрос**: что такое MCP в одном предложении?

<details>

<summary><strong>Ответ</strong></summary>

Стандартный протокол, чтобы вызывать тулзы, живущие в отдельном процессе/на сервере — «tool call по сети».

</details>

#### ❓ **Вопрос**: почему агентные бенчи — про код?

<details>

<summary><strong>Ответ</strong></summary>

Потому что тесты дают дешёвый автоматический верификатор успеха; для большинства реальных задач такого верификатора нет.

</details>

#### ❓ **Вопрос**: одна и та же модель даёт 30% resolved в «своём» харнессе и 18% в чужом. Это плохая модель или плохой харнесс?

<details>

<summary><strong>Ответ</strong></summary>

Ни то, ни другое по отдельности: метрика — это всегда **связка модель+харнесс**, и приписать просадку одному из них нельзя. Модели затачиваются под «свой» харнесс (GPT↔Codex, Claude↔Claude Code), поэтому чужой занижает результат именно из-за рассогласования пары, а не из-за «плохости» компонента.

</details>
"""
)

md(
    """
## Идеи и материалы

- **Загоняйте модель в жёсткие рамки.** Узкие тулзы, точные контракты, plan mode, лимиты шагов — чем меньше свободы у модели в опасных местах, тем надёжнее агент.
- Поиграть: дать `run_agent` (на реальной Cloud.ru-модели) задачу посложнее или добавить новый тул.

### Ссылки

- Cloud.ru Foundation Models — документация: https://cloud.ru/docs/foundation-models/ug/index
- SWE-bench: https://www.swebench.com/
- Anthropic — *Building effective agents*: https://www.anthropic.com/research/building-effective-agents
- Model Context Protocol: https://modelcontextprotocol.io/
- Готовые системы: https://omo.dev/
"""
)

# ---------------------------------------------------------------------------
# Демо: агентные паттерны вживую (live terminal demos)
# ---------------------------------------------------------------------------
md(
    """
## Демо: агентные паттерны вживую

Дальше — не код в ноутбуке, а **живые демо в терминале** (запускает лектор). Идея — показать, как обсуждённые паттерны выглядят в реальном кодовом агенте.
"""
)

md(
    """
### Рефлексия (reflection) — на примере скилла `tikz-diagrams`

**Reflection** — агентный паттерн, где модель **критикует и улучшает собственный вывод в цикле**: сгенерировал → посмотрел на результат → нашёл проблемы → переделал. Это ровно наш agent loop из блока 3, но «инструмент» — это *проверка собственной работы*.

Скилл `tikz-diagrams` так и устроен: рисует диаграмму → компилирует → **смотрит на картинку** → находит наложения/обрезку/ошибки → правит код → повторяет, пока не станет хорошо. Verifier здесь — компиляция + визуальная проверка отрендеренного изображения.

> 🖥️ Живое демо в терминале.
"""
)

md(
    """
### Авторесёрч (autoresearch)

**Autoresearch** — агент сам крутит цикл «гипотеза → эксперимент/поиск → вывод» к заданной цели: ведёт лог решений, проверяет результат заданным верификатором и останавливается по достижении цели или по лимиту времени. Классический пример высокой неопределённости — заранее неизвестно, сколько итераций и каких шагов понадобится (см. блок 2).

> 🖥️ Живое демо в терминале.
"""
)

# ---------------------------------------------------------------------------
# Как я использую Claude Code (relocated to the end)
# ---------------------------------------------------------------------------
md(
    """
## Как я использую Claude Code

Набор приёмов (на примере oh-my-claude-code), который превращает кодового агента из «автокомплита» в рабочий инструмент. Сгруппируем их по двум главным проблемам: **как объяснить задачу** и **как заставить агента довести её до конца**.
"""
)

md(
    """
### Как написать промт (снять неопределённость до кода)

Главная причина плохого результата — агент кодит не то, потому что задача размыта. Эти приёмы **сначала уточняют, что делать**, и только потом запускают исполнение:

| Приём | Что делает | Кто и кого спрашивает | Выход | Когда брать |
|---|---|---|---|---|
| **plan mode** | изменения запрещены: агент только исследует код и предлагает план | агент сам исследует репозиторий | план на утверждение человеком | большая задача, высокая цена ошибки |
| **grill-with-docs** | агент допрашивает тебя, сверяясь с документацией/кодом, и фиксирует ответы | агент → человека, опираясь на доки | уточнённая спека/документ | надо закрепить знание о системе |
| **deep interview** | агент задаёт уточняющие вопросы и гонит неопределённость вниз, пока ТЗ не станет однозначным | агент → человека | уточнённое ТЗ → код | размытая постановка (этот семинар собран так 🙂) |

Общая идея у всех трёх — **разделить «понять задачу» и «сделать задачу»**: дешёвый текстовый раунд до того, как агент начнёт менять файлы.
"""
)

md(
    """
### Как заставить агента доводить дело до конца (внешний цикл)

Вторая беда — агент останавливается на полпути: «вроде что-то сделал» и сдаётся. Лечится **внешним циклом с критерием завершения** — агента перезапускают/крутят, пока цель не достигнута:

- **ralph loop** — тупой, но упорный цикл: запускаем агента на ту же задачу снова и снова (буквально `while true`), и каждый прогон видит результат предыдущего. Берёт не умом, а повторением — хорошо заходит на «дожать почти готовое».
- **autoresearch** — агент сам крутит цикл «гипотеза → эксперимент/поиск → вывод» к цели: ведёт лог решений, проверяет результат **верификатором** и останавливается по достижении цели или лимиту.
- **goal mode** — задаёшь **цель + критерий готовности**; агент сам решает, сколько итераций нужно, и не выходит, пока критерий не выполнен (или не упёрся в лимит шагов/времени).

Ключ ко всем трём — **явный критерий «готово»** (тесты, верификатор, условие цели). Без него агент либо останавливается рано, либо крутится бесконечно — поэтому тут так важен дешёвый автоматический верификатор (как тесты в агентных бенчах, блок 2).
"""
)

md(
    """
### Эффективное использование лимитов и контекста

Контекст и лимиты — расходный ресурс. Несколько правил, чтобы не жечь его впустую:

- **Следи за KV-cache.** Префикс диалога кешируется на стороне сервера, но кеш живёт недолго — обычно **~5 минут с момента прошлого ответа**. Уложился в окно — повторный вызов дешевле и быстрее (префикс не пересчитывается); ушёл думать на полчаса — кеш протух, всё считается заново. Вывод: внутри одной задачи лучше работать пачкой, а не растягивать паузы.
- **`/compact` для длинных сессий.** Когда диалог разросся, сжимает историю в краткое резюме — освобождает контекст, но сохраняет суть. Делать до того, как упрёшься в потолок окна.
- **`/clear`, если сменилась задача.** Новая задача → старая история только мешает (засоряет контекст и путает модель). Чистый старт дешевле и точнее, чем тащить нерелевантный «хвост».
- **Не плоди 100500 скиллов.** Описание каждого скилла висит в контексте, чтобы агент знал, когда его звать. Десятки полузабытых скиллов — это постоянный налог на контекст и лишний шум при выборе. Держи только те, что реально используешь.
- **Не гонюсь за 1M-контекстными моделями.** Лично я их не беру: на длинном контексте качество **деградирует** (модель «теряется» в середине окна), стоят они **дороже**, а реальные задачи обычно спокойно влезают в **256k**. Дешевле и точнее держать контекст компактным (`/compact`, `/clear`), чем заливать в окно всё подряд и надеяться, что модель сама разберётся.

Общий принцип — **контекст должен содержать только релевантное текущей задаче**: меньше мусора → дешевле, быстрее и точнее.
"""
)

md(
    """
### Прочие приёмы

- **background tasks** — длинные задачи в фоне с нотификацией по завершении.
- **create skill** — упаковать повторяемый воркфлоу в переиспользуемый «скилл».
- **claude audit** — проверка актуальности документации против кода.
- **code review** — отдельный проход ревью диффа.
- **аналог openclaw** — свой минимальный агент под конкретную нишу.
"""
)

md(
    """
#### ❓ **Вопрос**: чем skill отличается от команды (slash-command), и что класть в skill, а что в `CLAUDE.md`?

<details>

<summary><strong>Ответ</strong></summary>

**Команда** — это разовый триггер: «сделай вот это сейчас» (по сути промпт с аргументами).</br>
**Skill** — переиспользуемая *процедура с инструкциями и контекстом*, которую агент подхватывает, когда задача ей соответствует (когда применять, шаги, ограничения).</br>
**`CLAUDE.md`** — всегда-актуальные правила и факты *проекта* (как собирать, где что лежит, стиль), которые грузятся в каждую сессию.</br>
Правило: «как делать конкретный воркфлоу» → skill; «что всегда верно про этот репозиторий» → `CLAUDE.md`; «сделай это однократно» → команда.

</details>
"""
)

md(
    """
Пример каркаса скилла — это просто markdown с фронтматтером (имя + когда применять) и инструкциями:
"""
)

code(
    '''
skill_example = """\\
---
name: clear-notebook-outputs
description: Очистить outputs во всех ноутбуках семинара перед коммитом
---

Когда пользователь просит подготовить ноутбук к коммиту:
1. Найти изменённые *.ipynb.
2. Выполнить nbconvert --ClearOutputPreprocessor.enabled=True --inplace.
3. Проверить, что размер файла < 100 KB.
"""
print(skill_example)
'''
)

# ---------------------------------------------------------------------------
# Write notebook
# ---------------------------------------------------------------------------
NB = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "12_agents.ipynb")
with open(out_path, "w") as f:
    json.dump(NB, f, ensure_ascii=False, indent=1)

print(f"Готово: {out_path} ({len(CELLS)} ячеек)")
