#!/usr/bin/env python3
"""
Финальная версия ноутбука с правками:
1. Знакомство с PyTorch в начале Part I
2. Все блиц-вопросы в конце
3. Без практических упражнений
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

def rebuild_final():
    """Rebuild notebook with all requirements."""

    print("📖 Loading source notebooks...")
    nb1 = load_notebook("old_01_seminar_torch_mlp.ipynb")
    nb2 = load_notebook("old_02_seminar_autograd.ipynb")

    print("🔨 Building final notebook...")

    nb = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"}
        },
        "cells": []
    }

    cells = []

    # ===== ВВЕДЕНИЕ =====
    cells.append(create_markdown_cell("""# Семинар 1: MLP на PyTorch и автоматическое дифференцирование

## План семинара

### Часть I: PyTorch MLP
* Знакомство с PyTorch (базовые интерфейсы, broadcasting)
* Работа с данными (make_moons)
* Определение модели MLP на PyTorch
* Функции потерь и обучение
* Роль нелинейностей
* Сравнение с SVM
* Батчинг и эффективность

### Часть II: Autograd и Backpropagation
* Как работает автоматическое дифференцирование?
* Forward и backward pass
* Chain rule и backpropagation
* Примеры autograd в PyTorch
* Реализация собственного autograd

### Блиц-вопросы
* Проверка понимания материала"""))

    cells.append(create_markdown_cell("---\n\n# Часть I: PyTorch MLP"))

    # ===== ЗНАКОМСТВО С PYTORCH =====
    print("  Adding PyTorch basics...")
    cells.append(create_markdown_cell("""## Знакомство с PyTorch

PyTorch - библиотека для глубокого обучения, разработанная Meta (Facebook).

**Основные преимущества:**
* Интуитивный API (похож на NumPy)
* Динамический computational graph
* Удобные инструменты для GPU
* Автоматическое дифференцирование

### Аналогия с NumPy

PyTorch tensors работают очень похоже на NumPy arrays:"""))

    cells.append(create_code_cell("""import torch
import numpy as np

# NumPy
np_array = np.array([1, 2, 3, 4, 5])
print("NumPy array:", np_array)
print("Shape:", np_array.shape)
print("Mean:", np_array.mean())

print()

# PyTorch (очень похоже!)
torch_tensor = torch.tensor([1, 2, 3, 4, 5])
print("PyTorch tensor:", torch_tensor)
print("Shape:", torch_tensor.shape)
print("Mean:", torch_tensor.float().mean())  # PyTorch требует float для mean()"""))

    cells.append(create_markdown_cell("""### Основные операции с тензорами"""))

    cells.append(create_code_cell("""# Создание тензоров
a = torch.zeros(3, 4)        # Матрица 3x4 из нулей
b = torch.ones(3, 4)         # Матрица 3x4 из единиц
c = torch.rand(3, 4)         # Случайные числа [0, 1)
d = torch.randn(3, 4)        # Нормальное распределение N(0, 1)

print("Zeros:\\n", a)
print("\\nOnes:\\n", b)
print("\\nRandom uniform:\\n", c)
print("\\nRandom normal:\\n", d)"""))

    cells.append(create_code_cell("""# Арифметические операции
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])

print("x + y =", x + y)
print("x * y =", x * y)
print("x @ y =", x @ y)  # Скалярное произведение (dot product)"""))

    cells.append(create_markdown_cell("""### Broadcasting в PyTorch

**Broadcasting** - механизм, позволяющий производить операции между тензорами разных размеров.

PyTorch автоматически "растягивает" тензоры меньшего размера, чтобы они совпадали по размерности.

**Правила broadcasting:**
1. Если тензоры имеют разное количество измерений, форма тензора с меньшим количеством измерений дополняется единицами слева
2. Размеры считаются совместимыми, если они равны или один из них равен 1
3. Тензоры расширяются по измерениям размером 1

Подробнее: [PyTorch Broadcasting Semantics](https://pytorch.org/docs/stable/notes/broadcasting.html)"""))

    cells.append(create_code_cell("""# Пример 1: Вектор + скаляр
x = torch.tensor([1.0, 2.0, 3.0])  # shape: (3,)
scalar = 10.0                       # shape: ()

result = x + scalar
print("Вектор + скаляр:")
print(f"  {x.tolist()} + {scalar} = {result.tolist()}")
print(f"  Shapes: {x.shape} + () = {result.shape}")

print()

# Пример 2: Матрица + вектор
matrix = torch.tensor([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]])  # shape: (2, 3)
vector = torch.tensor([10.0, 20.0, 30.0])  # shape: (3,)

result = matrix + vector
print("Матрица + вектор:")
print("Matrix:\\n", matrix)
print("Vector:", vector)
print("Result:\\n", result)
print(f"Shapes: {matrix.shape} + {vector.shape} = {result.shape}")"""))

    cells.append(create_code_cell("""# Пример 3: Broadcasting в обе стороны
a = torch.tensor([[1.0],
                  [2.0],
                  [3.0]])  # shape: (3, 1)

b = torch.tensor([10.0, 20.0, 30.0])  # shape: (3,) → будет расширено до (1, 3)

result = a + b
print("Broadcasting в обе стороны:")
print("a (3, 1):\\n", a)
print("b (3,):", b)
print("Result (3, 3):\\n", result)
print(f"Shapes: {a.shape} + {b.shape} → {result.shape}")"""))

    cells.append(create_markdown_cell("""### PyTorch vs NumPy: ключевые отличия

| Аспект | NumPy | PyTorch |
|--------|-------|---------|
| Основная структура | `ndarray` | `Tensor` |
| GPU поддержка | ❌ Нет | ✅ Да (`.cuda()`, `.to('cuda')`) |
| Autograd | ❌ Нет | ✅ Да (`.backward()`) |
| Создание | `np.array([1,2,3])` | `torch.tensor([1,2,3])` |
| Случайные числа | `np.random.rand(3,4)` | `torch.rand(3,4)` |
| Broadcasting | ✅ Да | ✅ Да (те же правила) |

**Когда использовать PyTorch вместо NumPy:**
* Нужно обучать нейросети (autograd!)
* Нужны вычисления на GPU
* Работаете с глубоким обучением"""))

    # ===== ОСТАЛЬНАЯ ЧАСТЬ I (из nb1, ячейки 1-22, без блиц и упражнений) =====
    print("  Adding Part I content from nb1...")
    # Берем ячейки 1-22 (данные, MLP, обучение, визуализации)
    cells.extend(nb1['cells'][1:23])

    # ===== ПЕРЕХОД =====
    print("  Adding transition...")
    cells.append(create_markdown_cell("""---

# Часть II: Как работает автоматическое дифференцирование?

На первой части мы использовали PyTorch как "черный ящик". Мы вызывали `loss.backward()` и магическим образом получали градиенты для всех параметров модели.

Но как это работает? Давайте разберемся!"""))

    # ===== ЧАСТЬ II (autograd из nb2) =====
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

    # Ищем ячейки с примерами из nb2
    for i, cell in enumerate(nb2['cells']):
        if cell['cell_type'] == 'code':
            source_text = ''.join(cell.get('source', []))
            if '%matplotlib inline' in source_text and 'import torch' in source_text:
                cells.append(cell)
                break

    cells.append(create_markdown_cell("""### Как на градиенты влияет сложение?

\\begin{align*}
c &= a + b \\\\
\\frac {\\partial c} {\\partial a} &= 1 \\\\
\\frac {\\partial c} {\\partial b} &= 1
\\end{align*}"""))

    # Найти пример сложения
    for cell in nb2['cells']:
        if cell['cell_type'] == 'code':
            source_text = ''.join(cell.get('source', []))
            if 'a = torch.Tensor([10.])' in source_text and 'с = a + b' in source_text:
                cells.append(cell)
                break

    cells.append(create_markdown_cell("""### Как на градиенты влияет умножение?

\\begin{align*}
c &= a \\cdot b \\\\
\\frac {\\partial c} {\\partial a} &= b \\\\
\\frac {\\partial c} {\\partial b} &= a
\\end{align*}"""))

    # Найти пример умножения
    for cell in nb2['cells']:
        if cell['cell_type'] == 'code':
            source_text = ''.join(cell.get('source', []))
            if 'a = torch.Tensor([10.])' in source_text and 'с = a * b' in source_text:
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

    # ===== БЛИЦ-ВОПРОСЫ (ВСЕ В КОНЦЕ!) =====
    print("  Adding all quiz questions at the end...")
    cells.append(create_markdown_cell("""---

# Блиц-вопросы

## Часть I: PyTorch и MLP

1. В чем главное отличие PyTorch от NumPy?

2. Что такое broadcasting? Приведите пример.

3. Как луны может разделить логистическая регрессия?

4. Чем наша реализация MLP отличается от LinearSVC?

5. Как `learning_rate` влияет на скорость обучения? Что будет с очень маленьким lr=1e-8? С очень большим lr=1e3?

6. Что будет, если убрать все нелинейности из нашей модели?

7. Что такое батч? Почему вычисления в нейросетях батчуются?

8. Чем тензор отличается от torch параметра?

## Часть II: Autograd и Backpropagation

1. Зачем нужны функции активации в нейросетях?

2. Зачем нужен autograd? Почему нельзя вычислять градиенты вручную?

3. Когда вычисляются градиенты - во время forward или backward pass?

4. Какие градиенты нам нужны для обучения нейросети и зачем?

5. Как computational graph, построенный во время forward pass, используется при backward pass?

6. Как сложение и умножение влияют на градиенты?

7. Что такое closure (замыкание) в Python?

8. Какой правильный порядок шагов при обучении нейросети?
   - a) forward → backward → zero_grad → optimizer.step
   - b) zero_grad → forward → backward → optimizer.step
   - c) backward → forward → zero_grad → optimizer.step"""))

    # ===== ДОПОЛНИТЕЛЬНЫЕ МАТЕРИАЛЫ =====
    cells.append(create_markdown_cell("""---

## Дополнительные материалы

* [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
* [PyTorch Tutorials](https://pytorch.org/tutorials/)
* [PyTorch Broadcasting Semantics](https://pytorch.org/docs/stable/notes/broadcasting.html)
* [Backpropagation Calculus](https://www.youtube.com/watch?v=tIeHLnjs5U8) - отличное видео от 3Blue1Brown
* [micrograd](https://github.com/karpathy/micrograd) - минималистичный autograd engine от Andrej Karpathy"""))

    # Сохраняем
    nb['cells'] = cells

    print(f"\n💾 Saving final notebook ({len(cells)} cells)...")
    save_notebook(nb, "01_seminar_mlp_autograd.ipynb")

    print("✅ Done!")
    print(f"\n📊 Statistics:")
    print(f"   Total cells: {len(cells)}")
    print(f"   Code cells: {len([c for c in cells if c['cell_type'] == 'code'])}")
    print(f"   Markdown cells: {len([c for c in cells if c['cell_type'] == 'markdown'])}")

if __name__ == "__main__":
    rebuild_final()
