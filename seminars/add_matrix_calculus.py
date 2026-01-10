#!/usr/bin/env python3
"""
Добавление раздела о матричном дифференцировании в ноутбук.
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

def add_matrix_calculus_section():
    """Add matrix calculus section to the notebook."""

    print("📖 Loading notebook...")
    nb = load_notebook("01_seminar_mlp_autograd.ipynb")

    print("🔍 Finding insertion point...")
    # Find the cell after "Backpropagation + Chain rule"
    insertion_index = None
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'markdown':
            source_text = ''.join(cell.get('source', []))
            if 'Backpropagation + Chain rule' in source_text:
                # Find the next cell after the chain rule explanation
                # We'll insert after all related cells
                insertion_index = i + 1
                # Skip any immediately following cells that are part of the same section
                while insertion_index < len(nb['cells']):
                    next_cell = nb['cells'][insertion_index]
                    if next_cell['cell_type'] == 'markdown':
                        next_text = ''.join(next_cell.get('source', []))
                        # If we hit a new major section, stop
                        if next_text.strip().startswith('# '):
                            break
                    insertion_index += 1
                break

    if insertion_index is None:
        print("❌ Could not find insertion point!")
        return

    print(f"✅ Found insertion point at cell {insertion_index}")

    # Create new cells for matrix calculus section
    new_cells = []

    # Main section header
    new_cells.append(create_markdown_cell("""---

# Матричное дифференцирование

При работе с нейросетями мы постоянно имеем дело с векторами и матрицами. Чтобы эффективно вычислять градиенты, нам нужно понимать правила матричного дифференцирования.

## Обозначения

В этом разделе будем использовать следующие обозначения:

- **Скаляры**: $a, b, c$ (обычные числа)
- **Векторы**: $\\mathbf{x}, \\mathbf{y}, \\mathbf{w}$ (жирные строчные буквы)
  - $\\mathbf{x} \\in \\mathbb{R}^n$ — вектор-столбец размера $n$
- **Матрицы**: $W, X, A$ (заглавные буквы)
  - $W \\in \\mathbb{R}^{m \\times n}$ — матрица размера $m \\times n$
- **Транспонирование**: $\\mathbf{x}^T, W^T$"""))

    # Types of derivatives
    new_cells.append(create_markdown_cell("""## Типы производных

В зависимости от того, что по чему дифференцируем, получаем разные объекты:

| Числитель | Знаменатель | Результат | Размерность | Название |
|-----------|-------------|-----------|-------------|----------|
| Скаляр $y$ | Вектор $\\mathbf{x} \\in \\mathbb{R}^n$ | Вектор | $n \\times 1$ | Градиент |
| Вектор $\\mathbf{y} \\in \\mathbb{R}^m$ | Скаляр $x$ | Вектор | $m \\times 1$ | Тангенс |
| Вектор $\\mathbf{y} \\in \\mathbb{R}^m$ | Вектор $\\mathbf{x} \\in \\mathbb{R}^n$ | Матрица | $m \\times n$ | Якобиан |
| Скаляр $y$ | Матрица $W \\in \\mathbb{R}^{m \\times n}$ | Матрица | $m \\times n$ | Градиент |

**Важно**: В нейросетях чаще всего нас интересует производная **скаляра** (функции потерь) по **параметрам** (векторам или матрицам)."""))

    # Basic rules
    new_cells.append(create_markdown_cell("""## Основные правила матричного дифференцирования

### 1. Линейные операции

**Производная линейной формы:**
$$\\frac{\\partial (\\mathbf{a}^T \\mathbf{x})}{\\partial \\mathbf{x}} = \\mathbf{a}$$

где $\\mathbf{a}$ — константный вектор, $\\mathbf{x}$ — переменная.

**Производная квадратичной формы:**
$$\\frac{\\partial (\\mathbf{x}^T W \\mathbf{x})}{\\partial \\mathbf{x}} = (W + W^T) \\mathbf{x}$$

Если $W$ симметричная ($W = W^T$), то:
$$\\frac{\\partial (\\mathbf{x}^T W \\mathbf{x})}{\\partial \\mathbf{x}} = 2W\\mathbf{x}$$

**Производная нормы:**
$$\\frac{\\partial \\|\\mathbf{x}\\|^2}{\\partial \\mathbf{x}} = \\frac{\\partial (\\mathbf{x}^T \\mathbf{x})}{\\partial \\mathbf{x}} = 2\\mathbf{x}$$"""))

    new_cells.append(create_markdown_cell("""### 2. Матрично-векторное умножение

**Производная по вектору:**
$$\\frac{\\partial (W\\mathbf{x})}{\\partial \\mathbf{x}} = W^T$$

**Производная по матрице:**
$$\\frac{\\partial (\\mathbf{a}^T W \\mathbf{x})}{\\partial W} = \\mathbf{a} \\mathbf{x}^T$$

где результат — матрица размера $m \\times n$ (внешнее произведение векторов).

### 3. Chain Rule для матриц

Если $y = f(\\mathbf{z})$ и $\\mathbf{z} = g(\\mathbf{x})$, то:

$$\\frac{\\partial y}{\\partial \\mathbf{x}} = \\frac{\\partial y}{\\partial \\mathbf{z}} \\cdot \\frac{\\partial \\mathbf{z}}{\\partial \\mathbf{x}}$$

где $\\frac{\\partial \\mathbf{z}}{\\partial \\mathbf{x}}$ — это матрица Якоби размера $|\\mathbf{z}| \\times |\\mathbf{x}|$."""))

    # Examples for neural networks
    new_cells.append(create_markdown_cell("""## Примеры для нейросетей

### Пример 1: Линейный слой

Рассмотрим линейный слой: $\\mathbf{z} = W\\mathbf{x} + \\mathbf{b}$

**Дано**: градиент потерь по выходу $\\frac{\\partial L}{\\partial \\mathbf{z}}$

**Нужно найти**:
- $\\frac{\\partial L}{\\partial W}$ — градиент по весам
- $\\frac{\\partial L}{\\partial \\mathbf{x}}$ — градиент по входу (для backprop дальше)
- $\\frac{\\partial L}{\\partial \\mathbf{b}}$ — градиент по bias

**Решение**:

1. Градиент по весам:
$$\\frac{\\partial L}{\\partial W} = \\frac{\\partial L}{\\partial \\mathbf{z}} \\cdot \\mathbf{x}^T$$

2. Градиент по входу:
$$\\frac{\\partial L}{\\partial \\mathbf{x}} = W^T \\cdot \\frac{\\partial L}{\\partial \\mathbf{z}}$$

3. Градиент по bias:
$$\\frac{\\partial L}{\\partial \\mathbf{b}} = \\frac{\\partial L}{\\partial \\mathbf{z}}$$"""))

    # Code example
    new_cells.append(create_code_cell("""# Пример: градиенты линейного слоя в PyTorch
import torch

# Создаем данные
x = torch.randn(5, requires_grad=True)  # вход: вектор размера 5
W = torch.randn(3, 5, requires_grad=True)  # веса: матрица 3x5
b = torch.randn(3, requires_grad=True)  # bias: вектор размера 3

# Forward pass
z = W @ x + b  # z = Wx + b
loss = z.sum()  # простая функция потерь (сумма элементов)

# Backward pass
loss.backward()

print("Градиент по W:")
print(f"  Shape: {W.grad.shape}")
print(f"  Вычисляется как: dL/dz * x^T")
print()
print("Градиент по x:")
print(f"  Shape: {x.grad.shape}")
print(f"  Вычисляется как: W^T * dL/dz")
print()
print("Градиент по b:")
print(f"  Shape: {b.grad.shape}")
print(f"  Равен dL/dz")"""))

    # Example 2: MSE loss
    new_cells.append(create_markdown_cell("""### Пример 2: Mean Squared Error (MSE)

Функция потерь: $L = \\frac{1}{n} \\|\\mathbf{y}_{pred} - \\mathbf{y}_{true}\\|^2 = \\frac{1}{n} \\sum_{i=1}^n (y_{pred}^{(i)} - y_{true}^{(i)})^2$

**Градиент по предсказаниям:**
$$\\frac{\\partial L}{\\partial \\mathbf{y}_{pred}} = \\frac{2}{n}(\\mathbf{y}_{pred} - \\mathbf{y}_{true})$$

Это показывает, что градиент направлен в сторону увеличения ошибки, и при обучении мы движемся в противоположную сторону (gradient descent)."""))

    new_cells.append(create_code_cell("""# Проверяем градиент MSE
y_pred = torch.randn(10, requires_grad=True)
y_true = torch.randn(10)

# MSE loss
loss = ((y_pred - y_true) ** 2).mean()
loss.backward()

print("Предсказания:", y_pred.data[:5])
print("Истинные значения:", y_true[:5])
print("Градиент:", y_pred.grad[:5])
print()
print("Формула: dL/dy_pred = 2/n * (y_pred - y_true)")
manual_grad = 2 * (y_pred.data - y_true) / len(y_pred)
print("Ручной расчет:", manual_grad[:5])
print("Совпадает с PyTorch:", torch.allclose(y_pred.grad, manual_grad))"""))

    # Summary
    new_cells.append(create_markdown_cell("""## Зачем это нужно?

1. **Понимание backpropagation**: Все градиенты в нейросетях вычисляются по этим правилам
2. **Оптимизация**: Знание формул помогает писать эффективный код
3. **Отладка**: Можно проверить правильность градиентов вручную
4. **Создание своих слоев**: При реализации custom layers нужно знать, как вычислять градиенты

## Полезные ресурсы

- [The Matrix Calculus You Need For Deep Learning](https://explained.ai/matrix-calculus/) — подробный туториал
- [Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) — справочник формул
- [CS231n: Backpropagation](https://cs231n.github.io/optimization-2/) — объяснение backprop с примерами
- [PyTorch Autograd](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html) — как PyTorch вычисляет градиенты"""))

    # Insert new cells
    print(f"📝 Inserting {len(new_cells)} new cells...")
    nb['cells'] = nb['cells'][:insertion_index] + new_cells + nb['cells'][insertion_index:]

    # Save
    print(f"💾 Saving updated notebook ({len(nb['cells'])} cells)...")
    save_notebook(nb, "01_seminar_mlp_autograd.ipynb")

    print("✅ Done!")
    print(f"\n📊 Statistics:")
    print(f"   Total cells: {len(nb['cells'])}")
    print(f"   Added cells: {len(new_cells)}")
    print(f"   Code cells: {len([c for c in nb['cells'] if c['cell_type'] == 'code'])}")
    print(f"   Markdown cells: {len([c for c in nb['cells'] if c['cell_type'] == 'markdown'])}")

if __name__ == "__main__":
    add_matrix_calculus_section()
