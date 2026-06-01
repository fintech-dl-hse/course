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
6. **Model ↔ Harness coupling** + квиз и блиц.

> **Главная мысль семинара:** агент — это не «умная модель», а **модель внутри харнесса**. Качество агента = качество модели × качество харнесса, и разделить их нельзя.
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

Ниже почти весь код работает **без GPU и без интернета** — на детерминированном «mock LLM», чтобы семинар был воспроизводим. Реальный Qwen и внешние API подключаются опциональными ячейками.
"""
)

code(
    """
# В Colab при необходимости раскомментируйте:
# !pip install -q transformers torch

import json
import re
import textwrap
from dataclasses import dataclass, field
from typing import Any, Callable

import random

random.seed(0)

print("Окружение готово. Тяжёлые зависимости (torch/transformers) — опциональны.")
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

![Цикл tool calling: модель решает, харнесс парсит и исполняет](static/tool_calling_cycle.png)
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
<div style="margin-top: 4px;">For each function call, return a json object with function name and arguments within <span style="color:#ea580c; font-weight:600;">&lt;tool_call&gt;&lt;/tool_call&gt;</span> XML tags.</div>

<div style="color: #b45309; font-weight: 700; margin-top: 12px;"># Tools</div>
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
### 1.3 Детерминированный mock LLM

Чтобы цикл tool calling был воспроизводим на семинаре (без GPU и сети), заменим модель на **сценарный mock**: он отдаёт заранее заданные ответы по шагам — ровно в том формате, который выдала бы настоящая Qwen. Парсер и исполнитель при этом — настоящие.
"""
)

code(
    '''
class MockLLM:
    """Игрушечная «модель»: отдаёт заранее заготовленные ответы по очереди.

    Реальная модель смотрит на messages; mock просто проигрывает script.
    """

    def __init__(self, script: list[str]):
        self.script = script
        self.step = 0

    def generate(self, messages: list[dict]) -> str:  # noqa: ARG002
        out = self.script[min(self.step, len(self.script) - 1)]
        self.step += 1
        return out


# Сценарий: сначала два tool call, потом финальный ответ.
weather_calc_script = [
    (
        '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>\\n'
        '<tool_call>{"name": "calculator", "arguments": {"expression": "2 + 2 * 10"}}</tool_call>'
    ),
    "В Париже +18°C и ясно, а 2 + 2 * 10 = 22.",
]
print("mock готов")
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


final = run_tool_loop(MockLLM(weather_calc_script), "Погода в Париже и сколько 2+2*10?", TOOL_SCHEMAS)
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
Если отдать сырой вывод функции пользователю — мы превратим агента обратно в жёсткий workflow и потеряем гибкость (см. блок 2).

</details>
"""
)

md(
    """
### 1.4 Реальная Qwen (опционально, нужен torch+transformers)

Здесь всё то же самое, но шаблон с инструментами строит сам `transformers`. Передаём `tools=` как список python-функций — схема собирается из docstring и type hints.
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
        print("transformers/torch не установлены — пропускаем (см. mock выше).")
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
### 1.5 Внешний API (опционально)

С хостед-моделями tool calling — это структурное поле в ответе (`tool_calls`), парсить регулярками не нужно. Любой OpenAI-совместимый эндпоинт (OpenAI, локальный vLLM с `--enable-auto-tool-choice`, и т.д.) работает одинаково.
"""
)

code(
    '''
# ОПЦИОНАЛЬНО: нужен ключ/эндпоинт и пакет openai.
def try_openai_tool_call(user_msg: str):
    try:
        from openai import OpenAI
    except ImportError:
        print("пакет openai не установлен — пропускаем.")
        return None
    # client = OpenAI()  # или base_url=локальный vLLM, api_key=...
    # resp = client.chat.completions.create(
    #     model="gpt-4o-mini",
    #     messages=[{"role": "user", "content": user_msg}],
    #     tools=TOOL_SCHEMAS,
    # )
    # tool_calls = resp.choices[0].message.tool_calls  # уже структурно!
    print("Раскомментируйте тело функции и задайте ключ/эндпоинт.")


print("Ячейка с API готова.")
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

![Agent = LLM + Harness: модель внутри обвязки](static/agent_equals_llm_harness.png)

Разница между workflow и агентом — в **степени неопределённости** и в том, **кто решает, что делать дальше**.

| | Workflow | Agent |
|---|---|---|
| Управление потоком | задано кодом заранее | модель решает на каждом шаге |
| Шаги известны? | да, фиксированный граф | нет, выясняются по ходу |
| Кол-во итераций | детерминированное | заранее неизвестно |
| Пример | «суммаризируй → переведи → отправь» | «разберись с этим багом» |
| Надёжность | высокая, предсказуемая | ниже, зато гибкая |

Эвристика: если вы можете нарисовать блок-схему заранее и она не меняется от входа — это **workflow**. Если модель сама выбирает следующий шаг в цикле «подумал → сделал → посмотрел на результат» — это **agent**.

![Workflow vs Agent: фиксированный конвейер против петли решений](static/workflow_vs_agent.png)

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
- результат — это связка **модель + харнесс**, а не «голая модель» (об этом блок 6);
- бенч на коде, потому что там есть **автоматический верификатор** (тесты) — для большинства агентских задач хорошего верификатора нет, и это главная проблема оценки.
"""
)

md(
    """
#### ❓ **Вопрос**: почему агентские бенчи почти всегда про код, а не про «закажи пиццу» или «спланируй поездку»?

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
def run_agent(llm, task: str, tools: dict, schemas: list[dict], max_steps: int = 8, verbose: bool = True):
    """Agent loop: модель в цикле дёргает тулзы, пока не вернёт финальный текст.

    Это и есть харнесс: system prompt + цикл + исполнение + лимит шагов.
    """
    system = render_system_prompt(schemas) + (
        "\\n\\nКогда задача выполнена — ответь пользователю обычным текстом без tool_call."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": task},
    ]
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
# Сценарий для mock-агента: записать число, прочитать, возвести в квадрат, ответить.
agent_script = [
    '<tool_call>{"name": "write_file", "arguments": {"filename": "n.txt", "content": "7"}}</tool_call>',
    '<tool_call>{"name": "read_file", "arguments": {"filename": "n.txt"}}</tool_call>',
    '<tool_call>{"name": "python_eval", "arguments": {"expression": "7 * 7"}}</tool_call>',
    "Готово: записал 7 в n.txt, квадрат числа равен 49.",
]

result = run_agent(
    MockLLM(agent_script),
    task="Сохрани число 7 в файл n.txt, потом посчитай его квадрат.",
    tools=AGENT_TOOLS,
    schemas=AGENT_SCHEMAS,
)
print("\\nИТОГ:", result)
print("Файлы в песочнице:", list_files())
'''
)

md(
    """
Чтобы подключить реальную модель — достаточно отдать в `run_agent` объект с методом `.generate(messages) -> str` поверх Qwen (как в 1.4) или внешнего API. Весь харнесс (цикл, тулзы, лимиты) остаётся тем же. **Это и есть «сделать своего openclaw».**
"""
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
### 4.2 Базовые тулзы: чтение и запись файлов

Кажется тривиальным, но именно тут много инженерии:

- **Read** обычно отдаёт файл **с номерами строк** и поддерживает чтение по диапазону (`offset`/`limit`) — чтобы не утопить контекст в огромных файлах.
- **Edit** чаще всего работает через **точное совпадение строки** (`old_string -> new_string`): модель обязана процитировать существующий фрагмент дословно. Это защита от «галлюцинированных» правок и от перезаписи не того места.
- **Write** перезаписывает файл целиком — поэтому хорошие харнессы требуют сначала *прочитать* файл, иначе легко затереть чужие изменения.

Эти ограничения — не каприз, а способ **загнать модель в жёсткие рамки**, где она ошибается реже.
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

**MCP (Model Context Protocol)** часто окружён хайпом, но идея простая:

> MCP — это **стандартный способ вызвать тул, который живёт в отдельном процессе/на сервере**, а не в твоём коде.

В блоках 1–3 тулзы были обычными python-функциями в том же процессе. MCP выносит их за границу процесса: **MCP-сервер** объявляет инструменты, **MCP-клиент** (агент) их обнаруживает и вызывает по стандартному протоколу (JSON-RPC поверх stdio или HTTP). Профит: один и тот же сервер (например, «доступ к Postgres» или «GitHub») переиспользуется любым MCP-совместимым агентом.

Покажем суть «тул по сети» на самодельном мини-протоколе **без зависимостей** — это буквально наш `execute_tool_calls`, но через сериализованный запрос/ответ (как было бы по сети).
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
Отличие инженерное: тул живёт **за границей процесса** и общается по *стандартному протоколу*. За счёт стандарта один сервер переиспользуется любым MCP-клиентом, а тулзы можно деплоить и обновлять независимо от агента.

</details>
"""
)

# ---------------------------------------------------------------------------
# Block 6: coupling + quiz + blitz
# ---------------------------------------------------------------------------
md(
    """
## 6. Model ↔ Harness coupling

Важный и часто упускаемый момент: **модель и харнесс обучаются вместе**.

- GPT-модели затачиваются под харнесс **Codex**.
- Claude (Opus и др.) — под **Claude Code**.

Поэтому нельзя честно сравнить «голый харнесс» без модели или «голую модель» без харнесса: одна и та же модель в чужом харнессе работает заметно хуже, а цифры из бенчей (блок 2) — это всегда **связка**. Когда читаете «модель X решает Y% SWE-bench» — спрашивайте, **в каком харнессе**.
"""
)

md(
    """
## Квиз: workflow или agent?

Для каждого сценария реши: это **workflow** (поток задан кодом) или **agent** (модель решает в цикле)? Иногда ответ — «зависит».
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
Высокая степень неопределённости — характерный признак агентской задачи.

</details>
"""
)

md(
    """
## Блиц

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

#### ❓ **Вопрос**: почему агентские бенчи — про код?

<details>

<summary><strong>Ответ</strong></summary>

Потому что тесты дают дешёвый автоматический верификатор успеха; для большинства реальных задач такого верификатора нет.

</details>

#### ❓ **Вопрос**: можно ли сравнить две модели «честно», прогнав их в одном харнессе?

<details>

<summary><strong>Ответ</strong></summary>

Не вполне: модели затачиваются под «свой» харнесс (GPT↔Codex, Claude↔Claude Code), поэтому чужой харнесс занижает результат. Сравнение всегда про связку модель+харнесс.

</details>
"""
)

md(
    """
## Идеи и материалы

- **Загоняйте модель в жёсткие рамки.** Узкие тулзы, точные контракты, plan mode, лимиты шагов — чем меньше свободы у модели в опасных местах, тем надёжнее агент.
- Поиграть: подключить к `run_agent` реальную Qwen из 1.4 и дать задачу посложнее.

### Ссылки

- SWE-bench: https://www.swebench.com/
- Anthropic — *Building effective agents*: https://www.anthropic.com/research/building-effective-agents
- Model Context Protocol: https://modelcontextprotocol.io/
- Готовые системы: https://omo.dev/
"""
)

# ---------------------------------------------------------------------------
# Демо: агентские паттерны вживую (live terminal demos)
# ---------------------------------------------------------------------------
md(
    """
## Демо: агентские паттерны вживую

Дальше — не код в ноутбуке, а **живые демо в терминале** (запускает лектор). Идея — показать, как обсуждённые паттерны выглядят в реальном кодовом агенте.
"""
)

md(
    """
### Рефлексия (reflection) — на примере скилла `tikz-diagrams`

**Reflection** — агентский паттерн, где модель **критикует и улучшает собственный вывод в цикле**: сгенерировал → посмотрел на результат → нашёл проблемы → переделал. Это ровно наш agent loop из блока 3, но «инструмент» — это *проверка собственной работы*.

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

Набор приёмов (на примере oh-my-claude-code), который превращает кодового агента из «автокомплита» в рабочий инструмент:

- **deep interview** — агент задаёт уточняющие вопросы и сам гонит неопределённость вниз, прежде чем кодить (этот семинар как раз собран через deep interview 🙂).
- **grill-with-docs** — то же интервью, но цель — *задокументировать* то, что выяснили.
- **background tasks** — длинные задачи в фоне с нотификацией по завершении.
- **create skill** — упаковать повторяемый воркфлоу в переиспользуемый «скилл».
- **claude audit** — проверка актуальности документации против кода.
- **code review** — отдельный проход ревью диффа.
- **autoresearch / goal mode** — агент сам крутит цикл «гипотеза → эксперимент → вывод» к заданной цели.
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
