#!/usr/bin/env python3
"""
Скрипт для объединения двух семинарских ноутбуков в один.
Объединяет old_01_seminar_torch_mlp.ipynb и old_02_seminar_autograd.ipynb
в единый 01_seminar_mlp_autograd.ipynb
"""

import json
from pathlib import Path
from typing import List, Dict, Any

def load_notebook(path: Path) -> Dict:
    """Load a Jupyter notebook from file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_notebook(notebook: Dict, path: Path):
    """Save a Jupyter notebook to file."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)
    print(f"✅ Notebook saved to: {path}")

def create_markdown_cell(content: str, cell_id: str = None) -> Dict:
    """Create a markdown cell."""
    cell = {
        "cell_type": "markdown",
        "source": content if isinstance(content, list) else [content],
        "metadata": {}
    }
    if cell_id:
        cell["metadata"]["id"] = cell_id
    return cell

def create_code_cell(source: str, cell_id: str = None) -> Dict:
    """Create a code cell."""
    cell = {
        "cell_type": "code",
        "execution_count": None,
        "source": source if isinstance(source, list) else [source],
        "outputs": [],
        "metadata": {}
    }
    if cell_id:
        cell["metadata"]["id"] = cell_id
    return cell

def create_base_notebook() -> Dict:
    """Create base notebook structure with metadata."""
    return {
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

def merge_notebooks():
    """Main function to merge notebooks."""

    # Paths
    notebook1_path = Path("old_01_seminar_torch_mlp.ipynb")
    notebook2_path = Path("old_02_seminar_autograd.ipynb")
    output_path = Path("01_seminar_mlp_autograd.ipynb")

    print(f"📖 Loading {notebook1_path}...")
    nb1 = load_notebook(notebook1_path)
    print(f"   Found {len(nb1['cells'])} cells")

    print(f"📖 Loading {notebook2_path}...")
    nb2 = load_notebook(notebook2_path)
    print(f"   Found {len(nb2['cells'])} cells")

    # Create new notebook
    print("\n🔨 Creating merged notebook...")
    new_nb = create_base_notebook()
    cells = []

    # ========================================
    # ЧАСТЬ 0: Введение
    # ========================================
    print("  Adding Part 0: Introduction")

    cells.append(create_markdown_cell(
        "# Семинар 1: MLP на PyTorch и автоматическое дифференцирование\n\n"
        "## План семинара\n\n"
        "### Часть I: PyTorch MLP\n"
        "* Работа с данными (make_moons)\n"
        "* Определение модели MLP на PyTorch\n"
        "* Функции потерь и обучение\n"
        "* Роль нелинейностей\n"
        "* Сравнение с SVM\n"
        "* Батчинг и эффективность\n"
        "* Блиц-вопросы\n\n"
        "### Часть II: Autograd и Backpropagation\n"
        "* Как работает автоматическое дифференцирование?\n"
        "* Forward и backward pass\n"
        "* Chain rule и backpropagation\n"
        "* Примеры autograd в PyTorch\n"
        "* Реализация собственного autograd\n"
        "* Блиц-вопросы\n\n"
        "### Часть III: Практические упражнения\n"
        "* Задания для самостоятельной работы\n"
    ))

    cells.append(create_markdown_cell("---\n\n# Часть I: PyTorch MLP\n"))

    # ========================================
    # ЧАСТЬ I: PyTorch MLP (из notebook 1)
    # ========================================
    print("  Adding Part I: PyTorch MLP")

    # Cells 1-23 from notebook 1 (excluding blitz, keeping only core material)
    # Data section (cells 1-2)
    cells.extend(nb1['cells'][1:3])

    # MLP definition and training (cells 3-11)
    cells.extend(nb1['cells'][3:12])

    # Non-linearities (cells 12-14)
    cells.extend(nb1['cells'][12:15])

    # SVM comparison (cells 15-19)
    cells.extend(nb1['cells'][15:20])

    # Batching (cells 20-22)
    cells.extend(nb1['cells'][20:23])

    # Add Part I Quiz (markdown cell with curated questions)
    cells.append(create_markdown_cell(
        "# Блиц-вопросы Часть I\n\n"
        "1. Как луны может разделить логистическая регрессия?\n\n"
        "2. Чем наша реализация MLP отличается от LinearSVC?\n\n"
        "3. Как `learning_rate` влияет на скорость обучения? "
        "Что будет с очень маленьким lr=1e-8? С очень большим lr=1e3?\n\n"
        "4. Что будет, если убрать все нелинейности из нашей модели?\n\n"
        "5. Что такое батч? Почему вычисления в нейросетях батчуются?\n\n"
        "6. Чем тензор отличается от torch параметра?\n"
    ))

    # ========================================
    # ПЕРЕХОД между частями
    # ========================================
    print("  Adding transition section")

    cells.append(create_markdown_cell(
        "---\n\n"
        "# Часть II: Как работает автоматическое дифференцирование?\n\n"
        "На первой части мы использовали PyTorch как \"черный ящик\". "
        "Мы вызывали `loss.backward()` и магическим образом получали градиенты для всех параметров модели.\n\n"
        "Но как это работает? Давайте разберемся!\n"
    ))

    # ========================================
    # ЧАСТЬ II: Autograd (из notebook 2, пропускаем feedback/recap)
    # ========================================
    print("  Adding Part II: Autograd")

    # Skip cells 0-3 (feedback, expectations, poll, recap)
    # Start from cell 4 onwards, but we need to check content

    # Add motivation section
    cells.append(create_markdown_cell(
        "## Зачем мы пилим автоград? 🤖\n\n"
        "Чтобы не считать градиенты вручную!\n\n"
        "## Что мы запомнили на лекции? 🤷\n\n"
        "* нейросеть -- это сложная функция (с параметрами), которая может быть представлена как композиция простых функций\n"
        "* оптимизируем с помощью градиентного спуска\n\n"
        "Чтобы эффективно обучать нейросети, нам нужно автоматически вычислять градиенты по всем параметрам.\n"
    ))

    cells.append(create_markdown_cell(
        "## Как работать с автоградом? 🪄\n\n"
        "От автограда нам нужно 2 вещи: **forward** и **backward pass**.\n\n"
        "### **forward pass** \n"
        "На этом этапе идет вычисление выхода сети: подаем вход, прогоняем через все слои, получаем предсказание.\n\n"
        "### **backward pass**\n"
        "На этом этапе вычисляются градиенты: начинаем с loss функции и идем назад по сети, "
        "вычисляя градиенты по всем параметрам с помощью chain rule.\n"
    ))

    # Add chain rule explanation (find relevant cells from nb2)
    # We'll add cells starting from around cell 6-7 of nb2
    # But let's be more selective

    cells.append(create_markdown_cell(
        "# Backpropagation + Chain rule = ❤️\n\n"
        "**Chain rule (правило дифференцирования сложной функции)**:\n\n"
        "Если $F = f(g(x))$, то $\\frac{dF}{dx} = \\frac{dF}{dg} \\cdot \\frac{dg}{dx}$\n\n"
        "Пример:\n"
        "\\begin{align*}\n"
        "F &= (a + b) c  \\\\\n"
        "q &= a + b  \\\\\n"
        "F &= q c\n"
        "\\end{align*}\n\n"
        "Тогда:\n"
        "\\begin{align*}\n"
        "\\frac{\\partial F}{\\partial a} &= \\frac{\\partial F}{\\partial q} \\cdot \\frac{\\partial q}{\\partial a} = c \\cdot 1 = c \\\\\n"
        "\\frac{\\partial F}{\\partial b} &= \\frac{\\partial F}{\\partial q} \\cdot \\frac{\\partial q}{\\partial b} = c \\cdot 1 = c \\\\\n"
        "\\frac{\\partial F}{\\partial c} &= q\n"
        "\\end{align*}\n\n"
        "**Backpropagation** - это просто применение chain rule для вычисления градиентов в нейросети!\n"
    ))

    # Add PyTorch autograd examples (from nb2, cells ~10-14)
    cells.append(create_markdown_cell("# Рассмотрим пример, как работает autograd в PyTorch"))

    # Find and add the imports cell from nb2
    for cell in nb2['cells']:
        if cell['cell_type'] == 'code' and 'import torch' in str(cell.get('source', '')):
            cells.append(cell)
            break

    cells.append(create_markdown_cell(
        "### Как на градиенты влияет сложение?\n\n"
        "\\begin{align*}\n"
        "c &= a + b \\\\\n"
        "\\frac {\\partial c} {\\partial a} &= 1 \\\\\n"
        "\\frac {\\partial c} {\\partial b} &= 1\n"
        "\\end{align*}\n"
    ))

    # Add example code cells from nb2 (addition example)
    for i, cell in enumerate(nb2['cells']):
        if cell['cell_type'] == 'code' and 'a = torch.Tensor([10.])' in str(cell.get('source', '')):
            if 'a * b' in str(cell.get('source', '')):
                continue  # Skip multiplication example for now
            cells.append(cell)

    cells.append(create_markdown_cell(
        "### Как на градиенты влияет умножение?\n\n"
        "\\begin{align*}\n"
        "c &= a \\cdot b \\\\\n"
        "\\frac {\\partial c} {\\partial a} &= b \\\\\n"
        "\\frac {\\partial c} {\\partial b} &= a\n"
        "\\end{align*}\n"
    ))

    # Add multiplication example from nb2
    for cell in nb2['cells']:
        if cell['cell_type'] == 'code' and 'a = torch.Tensor([10.])' in str(cell.get('source', '')) and 'a * b' in str(cell.get('source', '')):
            cells.append(cell)
            break

    # Add custom autograd implementation section
    cells.append(create_markdown_cell("# Мы готовы сделать свой автоград!"))

    cells.append(create_markdown_cell(
        "## ReLU (Rectified Linear Unit)\n\n"
        "В семинаре мы будем использовать ReLU в качестве функции активации:\n\n"
        "$$\n"
        "\\text{ReLU}(x) = \\max(0, x)\n"
        "$$\n\n"
        "Производная ReLU:\n\n"
        "$$\n"
        "\\frac{d \\text{ReLU}}{dx} = \\begin{cases} 1, & x > 0 \\\\ 0, & x \\leq 0 \\end{cases}\n"
        "$$\n"
    ))

    cells.append(create_markdown_cell(
        "### Python magic methods\n\n"
        "Python позволяет переопределять операторы через magic methods:\n\n"
        "```python\n"
        "Value(1) + Value(2)\n"
        "# превращается в\n"
        "Value(1).__add__(Value(2))\n"
        "```\n\n"
        "Мы будем использовать это, чтобы автоматически строить computational graph!\n"
    ))

    cells.append(create_markdown_cell(
        "### Closures (замыкания)\n\n"
        "Замыкание - это функция, которая \"запоминает\" переменные из внешней области видимости.\n\n"
        "```python\n"
        "def make_adder(x):\n"
        "    def adder(y):\n"
        "        return x + y  # x \"запомнили\" из внешней функции\n"
        "    return adder\n\n"
        "add_5 = make_adder(5)\n"
        "print(add_5(10))  # 15\n"
        "```\n\n"
        "Мы будем использовать замыкания для хранения градиентных функций!\n"
    ))

    # Add simplified Value class implementation
    cells.append(create_markdown_cell("## Класс Value - наш автоград"))

    cells.append(create_code_cell(
        "class Value:\n"
        "    \"\"\"Класс для автоматического дифференцирования.\"\"\"\n"
        "    \n"
        "    def __init__(self, data, _children=(), _op=''):\n"
        "        self.data = data\n"
        "        self.grad = 0.0\n"
        "        self._backward = lambda: None\n"
        "        self._prev = set(_children)\n"
        "        self._op = _op\n"
        "    \n"
        "    def __repr__(self):\n"
        "        return f\"Value(data={self.data}, grad={self.grad})\"\n"
        "    \n"
        "    def __add__(self, other):\n"
        "        other = other if isinstance(other, Value) else Value(other)\n"
        "        out = Value(self.data + other.data, (self, other), '+')\n"
        "        \n"
        "        def _backward():\n"
        "            self.grad += out.grad  # d(a+b)/da = 1\n"
        "            other.grad += out.grad  # d(a+b)/db = 1\n"
        "        out._backward = _backward\n"
        "        \n"
        "        return out\n"
        "    \n"
        "    def __mul__(self, other):\n"
        "        other = other if isinstance(other, Value) else Value(other)\n"
        "        out = Value(self.data * other.data, (self, other), '*')\n"
        "        \n"
        "        def _backward():\n"
        "            self.grad += other.data * out.grad  # d(a*b)/da = b\n"
        "            other.grad += self.data * out.grad  # d(a*b)/db = a\n"
        "        out._backward = _backward\n"
        "        \n"
        "        return out\n"
        "    \n"
        "    def relu(self):\n"
        "        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')\n"
        "        \n"
        "        def _backward():\n"
        "            self.grad += (out.data > 0) * out.grad  # производная ReLU\n"
        "        out._backward = _backward\n"
        "        \n"
        "        return out\n"
        "    \n"
        "    def backward(self):\n"
        "        \"\"\"Запускает backpropagation от этого узла.\"\"\"\n"
        "        # Топологическая сортировка\n"
        "        topo = []\n"
        "        visited = set()\n"
        "        \n"
        "        def build_topo(v):\n"
        "            if v not in visited:\n"
        "                visited.add(v)\n"
        "                for child in v._prev:\n"
        "                    build_topo(child)\n"
        "                topo.append(v)\n"
        "        \n"
        "        build_topo(self)\n"
        "        \n"
        "        # Идем от выхода к входу и вычисляем градиенты\n"
        "        self.grad = 1.0\n"
        "        for node in reversed(topo):\n"
        "            node._backward()\n"
    ))

    # Add example usage
    cells.append(create_markdown_cell("### Пример использования нашего autograd"))

    cells.append(create_code_cell(
        "# Пример: вычислим f(x, y) = (x + y) * x и его градиенты\n"
        "x = Value(2.0)\n"
        "y = Value(3.0)\n"
        "\n"
        "z = x + y  # z = 5\n"
        "f = z * x  # f = 10\n"
        "\n"
        "print(f\"f = {f.data}\")\n"
        "\n"
        "# Вычисляем градиенты\n"
        "f.backward()\n"
        "\n"
        "print(f\"df/dx = {x.grad}\")  # Должно быть: df/dx = z + x = 5 + 2 = 7\n"
        "print(f\"df/dy = {y.grad}\")  # Должно быть: df/dy = x = 2\n"
    ))

    # Add Part II Quiz
    cells.append(create_markdown_cell(
        "# Блиц-вопросы Часть II\n\n"
        "1. Зачем нужны функции активации в нейросетях?\n\n"
        "2. Зачем нужен autograd? Почему нельзя вычислять градиенты вручную?\n\n"
        "3. Когда вычисляются градиенты - во время forward или backward pass?\n\n"
        "4. Какие градиенты нам нужны для обучения нейросети и зачем?\n\n"
        "5. Как computational graph, построенный во время forward pass, используется при backward pass?\n\n"
        "6. Как сложение и умножение влияют на градиенты?\n\n"
        "7. Что такое closure (замыкание) в Python?\n\n"
        "8. Какой правильный порядок шагов при обучении нейросети?\n"
        "   a) forward → backward → zero_grad → optimizer.step\n"
        "   b) zero_grad → forward → backward → optimizer.step\n"
        "   c) backward → forward → zero_grad → optimizer.step\n"
    ))

    # ========================================
    # ЧАСТЬ III: Практические упражнения
    # ========================================
    print("  Adding Part III: Practical exercises")

    cells.append(create_markdown_cell(
        "---\n\n"
        "# Часть III: Практические упражнения\n"
    ))

    # Add exercises from notebook 1 (cells 35-42)
    cells.extend(nb1['cells'][34:42])

    # Add new exercises
    cells.append(create_markdown_cell(
        "#### Задание 4: Реализуйте Sigmoid activation для класса Value\n\n"
        "Сигмоида: $\\sigma(x) = \\frac{1}{1 + e^{-x}}$\n\n"
        "Производная: $\\frac{d\\sigma}{dx} = \\sigma(x) \\cdot (1 - \\sigma(x))$\n\n"
        "Добавьте метод `.sigmoid()` в класс Value, который:\n"
        "1. Вычисляет сигмоиду от self.data\n"
        "2. Корректно вычисляет градиент при backward pass\n\n"
        "Подсказка: вам понадобится `import math` для функции `math.exp()`\n"
    ))

    cells.append(create_code_cell(
        "import math\n"
        "\n"
        "# Ваша реализация здесь\n"
        "# Добавьте метод sigmoid в класс Value выше\n"
        "\n"
        "# Тест\n"
        "x = Value(0.0)\n"
        "y = x.sigmoid()\n"
        "y.backward()\n"
        "\n"
        "print(f\"sigmoid(0) = {y.data}\")  # Должно быть ~0.5\n"
        "print(f\"gradient = {x.grad}\")     # Должно быть ~0.25\n"
    ))

    cells.append(create_markdown_cell(
        "#### Задание 5 (Бонус): Реализуйте простую линейную регрессию используя Value class\n\n"
        "Используя класс Value, реализуйте и обучите простую линейную модель:\n\n"
        "$$y = wx + b$$\n\n"
        "1. Создайте несколько точек данных (например, $y = 2x + 1$ с шумом)\n"
        "2. Инициализируйте параметры $w$ и $b$ как Value объекты\n"
        "3. Реализуйте цикл обучения с MSE loss\n"
        "4. Обновляйте параметры через градиентный спуск\n"
        "5. Выведите финальные значения $w$ и $b$\n"
    ))

    cells.append(create_code_cell(
        "# Ваша реализация здесь\n"
        "\n"
        "# Данные\n"
        "# X = [1, 2, 3, 4, 5]\n"
        "# y_true = [3, 5, 7, 9, 11]  # y = 2x + 1\n"
        "\n"
        "# Инициализация параметров\n"
        "# w = Value(...)\n"
        "# b = Value(...)\n"
        "\n"
        "# Цикл обучения\n"
        "# for epoch in range(100):\n"
        "#     loss = ...\n"
        "#     loss.backward()\n"
        "#     # обновить w и b\n"
        "#     # обнулить градиенты\n"
    ))

    # Final note
    cells.append(create_markdown_cell(
        "---\n\n"
        "## Дополнительные материалы\n\n"
        "* [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)\n"
        "* [PyTorch Tutorials](https://pytorch.org/tutorials/)\n"
        "* [Backpropagation Calculus](https://www.youtube.com/watch?v=tIeHLnjs5U8) - отличное видео от 3Blue1Brown\n"
        "* [micrograd](https://github.com/karpathy/micrograd) - минималистичный autograd engine от Andrej Karpathy\n"
    ))

    # Add all cells to notebook
    new_nb['cells'] = cells

    # Save
    print(f"\n💾 Saving merged notebook with {len(cells)} cells...")
    save_notebook(new_nb, output_path)

    print(f"\n✨ Done! Created {output_path}")
    print(f"   Total cells: {len(cells)}")
    print("\n📝 Next steps:")
    print("   1. Review the notebook")
    print("   2. Run all cells to check for errors")
    print("   3. Fine-tune visualizations and code")

if __name__ == "__main__":
    merge_notebooks()
