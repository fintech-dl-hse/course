#!/usr/bin/env python3
"""
Пересоздание ноутбука с правильным форматированием кода.
Использует оригинальные ячейки из исходных ноутбуков.
"""

import json
from pathlib import Path

def load_notebook(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_notebook(notebook, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)

def create_markdown_cell(text):
    """Create markdown cell with proper line formatting."""
    lines = text.strip().split('\n')
    source = [line + '\n' for line in lines[:-1]]
    if lines:
        source.append(lines[-1])
    return {
        "cell_type": "markdown",
        "source": source,
        "metadata": {}
    }

def create_code_cell(code):
    """Create code cell with proper line formatting."""
    lines = code.strip().split('\n')
    source = [line + '\n' for line in lines[:-1]]
    if lines:
        source.append(lines[-1])
    return {
        "cell_type": "code",
        "execution_count": None,
        "source": source,
        "outputs": [],
        "metadata": {}
    }

def rebuild_notebook():
    """Rebuild notebook with correct formatting."""

    print("📖 Loading source notebooks...")
    nb1 = load_notebook("old_01_seminar_torch_mlp.ipynb")
    nb2 = load_notebook("old_02_seminar_autograd.ipynb")

    print("🔨 Building new notebook...")

    # Create base
    new_nb = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {
                "provenance": [],
                "toc_visible": True
            },
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "cells": []
    }

    cells = []

    # ===== ВВЕДЕНИЕ =====
    print("  Adding introduction...")
    cells.append(create_markdown_cell("""# Семинар 1: MLP на PyTorch и автоматическое дифференцирование

## План семинара

### Часть I: PyTorch MLP
* Работа с данными (make_moons)
* Определение модели MLP на PyTorch
* Функции потерь и обучение
* Роль нелинейностей
* Сравнение с SVM
* Батчинг и эффективность
* Блиц-вопросы

### Часть II: Autograd и Backpropagation
* Как работает автоматическое дифференцирование?
* Forward и backward pass
* Chain rule и backpropagation
* Примеры autograd в PyTorch
* Реализация собственного autograd
* Блиц-вопросы

### Часть III: Практические упражнения
* Задания для самостоятельной работы"""))

    cells.append(create_markdown_cell("---\n\n# Часть I: PyTorch MLP"))

    # ===== ЧАСТЬ I: из nb1 =====
    print("  Adding Part I from notebook 1...")

    # Берем ячейки 1-22 из notebook 1 (данные, MLP, обучение, визуализации)
    cells.extend(nb1['cells'][1:23])

    # Блиц-вопросы Часть I
    cells.append(create_markdown_cell("""# Блиц-вопросы Часть I

1. Как луны может разделить логистическая регрессия?

2. Чем наша реализация MLP отличается от LinearSVC?

3. Как `learning_rate` влияет на скорость обучения? Что будет с очень маленьким lr=1e-8? С очень большим lr=1e3?

4. Что будет, если убрать все нелинейности из нашей модели?

5. Что такое батч? Почему вычисления в нейросетях батчуются?

6. Чем тензор отличается от torch параметра?"""))

    # ===== ПЕРЕХОД =====
    print("  Adding transition...")
    cells.append(create_markdown_cell("""---

# Часть II: Как работает автоматическое дифференцирование?

На первой части мы использовали PyTorch как "черный ящик". Мы вызывали `loss.backward()` и магическим образом получали градиенты для всех параметров модели.

Но как это работает? Давайте разберемся!"""))

    # ===== ЧАСТЬ II: Autograd =====
    print("  Adding Part II...")

    cells.append(create_markdown_cell("""## Зачем мы пилим автоград? 🤖

Чтобы не считать градиенты вручную!

## Что мы запомнили на лекции? 🤷

* нейросеть -- это сложная функция (с параметрами), которая может быть представлена как композиция простых функций
* оптимизируем с помощью градиентного спуска

Чтобы эффективно обучать нейросети, нам нужно автоматически вычислять градиенты по всем параметрам."""))

    cells.append(create_markdown_cell("""## Как работать с автоградом? 🪄

От автограда нам нужно 2 вещи: **forward** и **backward pass**.

### **forward pass**
На этом этапе идет вычисление выхода сети: подаем вход, прогоняем через все слои, получаем предсказание.

### **backward pass**
На этом этапе вычисляются градиенты: начинаем с loss функции и идем назад по сети, вычисляя градиенты по всем параметрам с помощью chain rule."""))

    cells.append(create_markdown_cell("""# Backpropagation + Chain rule = ❤️

**Chain rule (правило дифференцирования сложной функции)**:

Если $F = f(g(x))$, то $\\frac{dF}{dx} = \\frac{dF}{dg} \\cdot \\frac{dg}{dx}$

Пример:
\\begin{align*}
F &= (a + b) c  \\\\
q &= a + b  \\\\
F &= q c
\\end{align*}

Тогда:
\\begin{align*}
\\frac{\\partial F}{\\partial a} &= \\frac{\\partial F}{\\partial q} \\cdot \\frac{\\partial q}{\\partial a} = c \\cdot 1 = c \\\\
\\frac{\\partial F}{\\partial b} &= \\frac{\\partial F}{\\partial q} \\cdot \\frac{\\partial q}{\\partial b} = c \\cdot 1 = c \\\\
\\frac{\\partial F}{\\partial c} &= q
\\end{align*}

**Backpropagation** - это просто применение chain rule для вычисления градиентов в нейросети!"""))

    cells.append(create_markdown_cell("# Рассмотрим пример, как работает autograd в PyTorch"))

    # Берем ячейки с примерами из nb2 (примерно 10-14)
    # Найдем ячейки с импортами и примерами
    for i, cell in enumerate(nb2['cells']):
        if cell['cell_type'] == 'code':
            source_text = ''.join(cell.get('source', []))
            # Imports
            if '%matplotlib inline' in source_text and 'import torch' in source_text:
                cells.append(cell)
                break

    cells.append(create_markdown_cell("""### Как на градиенты влияет сложение?

\\begin{align*}
c &= a + b \\\\
\\frac {\\partial c} {\\partial a} &= 1 \\\\
\\frac {\\partial c} {\\partial b} &= 1
\\end{align*}"""))

    # Найдем пример с сложением
    for cell in nb2['cells']:
        if cell['cell_type'] == 'code':
            source_text = ''.join(cell.get('source', []))
            if 'a = torch.Tensor([10.])' in source_text and 'a + b' in source_text:
                # Нужно убедиться что это пример сложения, а не умножения
                if 'с = a + b' in source_text or 'c = a + b' in source_text:
                    cells.append(cell)
                    break

    cells.append(create_markdown_cell("""### Как на градиенты влияет умножение?

\\begin{align*}
c &= a \\cdot b \\\\
\\frac {\\partial c} {\\partial a} &= b \\\\
\\frac {\\partial c} {\\partial b} &= a
\\end{align*}"""))

    # Найдем пример с умножением
    for cell in nb2['cells']:
        if cell['cell_type'] == 'code':
            source_text = ''.join(cell.get('source', []))
            if 'a = torch.Tensor([10.])' in source_text and 'a * b' in source_text:
                cells.append(cell)
                break

    cells.append(create_markdown_cell("# Мы готовы сделать свой автоград!"))

    cells.append(create_markdown_cell("""## ReLU (Rectified Linear Unit)

В семинаре мы будем использовать ReLU в качестве функции активации:

$$
\\text{ReLU}(x) = \\max(0, x)
$$

Производная ReLU:

$$
\\frac{d \\text{ReLU}}{dx} = \\begin{cases} 1, & x > 0 \\\\ 0, & x \\leq 0 \\end{cases}
$$"""))

    cells.append(create_markdown_cell("""### Python magic methods

Python позволяет переопределять операторы через magic methods:

```python
Value(1) + Value(2)
# превращается в
Value(1).__add__(Value(2))
```

Мы будем использовать это, чтобы автоматически строить computational graph!"""))

    cells.append(create_markdown_cell("""### Closures (замыкания)

Замыкание - это функция, которая "запоминает" переменные из внешней области видимости.

```python
def make_adder(x):
    def adder(y):
        return x + y  # x "запомнили" из внешней функции
    return adder

add_5 = make_adder(5)
print(add_5(10))  # 15
```

Мы будем использовать замыкания для хранения градиентных функций!"""))

    cells.append(create_markdown_cell("## Класс Value - наш автоград"))

    # Value class implementation
    cells.append(create_code_cell("""class Value:
    \"\"\"Класс для автоматического дифференцирования.\"\"\"

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad  # d(a+b)/da = 1
            other.grad += out.grad  # d(a+b)/db = 1
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad  # d(a*b)/da = b
            other.grad += self.data * out.grad  # d(a*b)/db = a
        out._backward = _backward

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad  # производная ReLU
        out._backward = _backward

        return out

    def backward(self):
        \"\"\"Запускает backpropagation от этого узла.\"\"\"
        # Топологическая сортировка
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Идем от выхода к входу и вычисляем градиенты
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()"""))

    cells.append(create_markdown_cell("### Пример использования нашего autograd"))

    cells.append(create_code_cell("""# Пример: вычислим f(x, y) = (x + y) * x и его градиенты
x = Value(2.0)
y = Value(3.0)

z = x + y  # z = 5
f = z * x  # f = 10

print(f"f = {f.data}")

# Вычисляем градиенты
f.backward()

print(f"df/dx = {x.grad}")  # Должно быть: df/dx = z + x = 5 + 2 = 7
print(f"df/dy = {y.grad}")  # Должно быть: df/dy = x = 2"""))

    # Блиц II
    cells.append(create_markdown_cell("""# Блиц-вопросы Часть II

1. Зачем нужны функции активации в нейросетях?

2. Зачем нужен autograd? Почему нельзя вычислять градиенты вручную?

3. Когда вычисляются градиенты - во время forward или backward pass?

4. Какие градиенты нам нужны для обучения нейросети и зачем?

5. Как computational graph, построенный во время forward pass, используется при backward pass?

6. Как сложение и умножение влияют на градиенты?

7. Что такое closure (замыкание) в Python?

8. Какой правильный порядок шагов при обучении нейросети?
   a) forward → backward → zero_grad → optimizer.step
   b) zero_grad → forward → backward → optimizer.step
   c) backward → forward → zero_grad → optimizer.step"""))

    # ===== ЧАСТЬ III =====
    print("  Adding Part III...")
    cells.append(create_markdown_cell("---\n\n# Часть III: Практические упражнения"))

    # Берем упражнения из nb1 (ячейки 34-42)
    cells.extend(nb1['cells'][34:42])

    # Добавляем новые упражнения
    cells.append(create_markdown_cell("""#### Задание 4: Реализуйте Sigmoid activation для класса Value

Сигмоида: $\\sigma(x) = \\frac{1}{1 + e^{-x}}$

Производная: $\\frac{d\\sigma}{dx} = \\sigma(x) \\cdot (1 - \\sigma(x))$

Добавьте метод `.sigmoid()` в класс Value, который:
1. Вычисляет сигмоиду от self.data
2. Корректно вычисляет градиент при backward pass

Подсказка: вам понадобится `import math` для функции `math.exp()`"""))

    cells.append(create_code_cell("""import math

# Ваша реализация здесь
# Добавьте метод sigmoid в класс Value выше

# Тест
x = Value(0.0)
y = x.sigmoid()
y.backward()

print(f"sigmoid(0) = {y.data}")  # Должно быть ~0.5
print(f"gradient = {x.grad}")     # Должно быть ~0.25"""))

    cells.append(create_markdown_cell("""#### Задание 5 (Бонус): Реализуйте простую линейную регрессию используя Value class

Используя класс Value, реализуйте и обучите простую линейную модель:

$$y = wx + b$$

1. Создайте несколько точек данных (например, $y = 2x + 1$ с шумом)
2. Инициализируйте параметры $w$ и $b$ как Value объекты
3. Реализуйте цикл обучения с MSE loss
4. Обновляйте параметры через градиентный спуск
5. Выведите финальные значения $w$ и $b$"""))

    cells.append(create_code_cell("""# Ваша реализация здесь

# Данные
# X = [1, 2, 3, 4, 5]
# y_true = [3, 5, 7, 9, 11]  # y = 2x + 1

# Инициализация параметров
# w = Value(...)
# b = Value(...)

# Цикл обучения
# for epoch in range(100):
#     loss = ...
#     loss.backward()
#     # обновить w и b
#     # обнулить градиенты"""))

    # Дополнительные материалы
    cells.append(create_markdown_cell("""---

## Дополнительные материалы

* [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
* [PyTorch Tutorials](https://pytorch.org/tutorials/)
* [Backpropagation Calculus](https://www.youtube.com/watch?v=tIeHLnjs5U8) - отличное видео от 3Blue1Brown
* [micrograd](https://github.com/karpathy/micrograd) - минималистичный autograd engine от Andrej Karpathy"""))

    # Собираем все
    new_nb['cells'] = cells

    print(f"\n💾 Saving rebuilt notebook ({len(cells)} cells)...")
    save_notebook(new_nb, "01_seminar_mlp_autograd.ipynb")

    print("✅ Done! Notebook rebuilt with correct formatting.")

    # Verify
    print("\n🔍 Verification:")
    nb_check = load_notebook("01_seminar_mlp_autograd.ipynb")
    code_cells = [c for c in nb_check['cells'] if c['cell_type'] == 'code']
    print(f"   Total cells: {len(nb_check['cells'])}")
    print(f"   Code cells: {len(code_cells)}")

    if code_cells:
        sample_cell = code_cells[5] if len(code_cells) > 5 else code_cells[0]
        num_lines = len(sample_cell.get('source', []))
        print(f"   Sample code cell has {num_lines} lines")
        if num_lines > 5:
            print("   ✓ Code is properly formatted!")
            # Show first few lines
            print("\n   First 3 lines of sample code:")
            for line in sample_cell['source'][:3]:
                print(f"     {line.rstrip()}")

if __name__ == "__main__":
    rebuild_notebook()
