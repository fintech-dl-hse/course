#!/usr/bin/env python3
"""
Обновление ноутбука согласно новым требованиям:
1. Все блиц-вопросы в конце
2. Удалить практические упражнения
3. Добавить введение в PyTorch в начало
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

def update_notebook():
    """Update notebook according to new requirements."""

    print("📖 Loading notebook...")
    nb = load_notebook("01_seminar_mlp_autograd.ipynb")

    print("🔨 Restructuring notebook...")

    # Создаем новую структуру
    new_cells = []

    # ===== ВВЕДЕНИЕ =====
    print("  Adding introduction...")
    new_cells.append(create_markdown_cell("""# Семинар 1: MLP на PyTorch и автоматическое дифференцирование

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

    new_cells.append(create_markdown_cell("---\n\n# Часть I: PyTorch MLP"))

    # ===== НОВАЯ СЕКЦИЯ: Знакомство с PyTorch =====
    print("  Adding PyTorch basics section...")
    new_cells.append(create_markdown_cell("""## Знакомство с PyTorch

PyTorch - библиотека для глубокого обучения, разработанная Meta (Facebook).

**Основные преимущества:**
* Интуитивный API (похож на NumPy)
* Динамический computational graph
* Удобные инструменты для GPU
* Автоматическое дифференцирование

### Аналогия с NumPy

PyTorch tensors работают очень похоже на NumPy arrays:"""))

    new_cells.append(create_code_cell("""import torch
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

    new_cells.append(create_markdown_cell("""### Основные операции с тензорами"""))

    new_cells.append(create_code_cell("""# Создание тензоров
a = torch.zeros(3, 4)        # Матрица 3x4 из нулей
b = torch.ones(3, 4)         # Матрица 3x4 из единиц
c = torch.rand(3, 4)         # Случайные числа [0, 1)
d = torch.randn(3, 4)        # Нормальное распределение N(0, 1)

print("Zeros:\\n", a)
print("\\nOnes:\\n", b)
print("\\nRandom uniform:\\n", c)
print("\\nRandom normal:\\n", d)"""))

    new_cells.append(create_code_cell("""# Арифметические операции
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])

print("x + y =", x + y)
print("x * y =", x * y)
print("x @ y =", x @ y)  # Скалярное произведение (dot product)"""))

    new_cells.append(create_markdown_cell("""### Broadcasting в PyTorch

**Broadcasting** - механизм, позволяющий производить операции между тензорами разных размеров.

PyTorch автоматически "растягивает" тензоры меньшего размера, чтобы они совпадали по размерности.

**Правила broadcasting:**
1. Если тензоры имеют разное количество измерений, форма тензора с меньшим количеством измерений дополняется единицами слева
2. Размеры считаются совместимыми, если они равны или один из них равен 1
3. Тензоры расширяются по измерениям размером 1

Подробнее: [PyTorch Broadcasting Semantics](https://pytorch.org/docs/stable/notes/broadcasting.html)"""))

    new_cells.append(create_code_cell("""# Пример 1: Вектор + скаляр
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

    new_cells.append(create_code_cell("""# Пример 3: Broadcasting в обе стороны
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

    new_cells.append(create_markdown_cell("""### PyTorch vs NumPy: ключевые отличия

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

    # ===== ОСТАЛЬНАЯ ЧАСТЬ I =====
    print("  Adding rest of Part I...")

    # Берем ячейки из исходного ноутбука (пропускаем блиц-вопросы и упражнения)
    # Находим ячейки для Part I (от "Данные" до "батчинг")
    in_part_1 = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'markdown':
            source_text = ''.join(cell.get('source', []))

            # Начало данных
            if '#  Данные' in source_text or '# Данные' in source_text:
                in_part_1 = True
                new_cells.append(cell)
                continue

            # Пропускаем блиц I
            if 'Блиц-вопросы Часть I' in source_text or 'блиц' in source_text.lower():
                in_part_1 = False
                continue

            # Начало Part II
            if 'Часть II:' in source_text or 'автоматическое дифференцирование' in source_text:
                in_part_1 = False
                # Не добавляем эту ячейку, добавим позже

        if in_part_1:
            new_cells.append(cell)

    # ===== ПЕРЕХОД =====
    print("  Adding transition...")
    new_cells.append(create_markdown_cell("""---

# Часть II: Как работает автоматическое дифференцирование?

На первой части мы использовали PyTorch как "черный ящик". Мы вызывали `loss.backward()` и магическим образом получали градиенты для всех параметров модели.

Но как это работает? Давайте разберемся!"""))

    # ===== ЧАСТЬ II =====
    print("  Adding Part II...")

    # Берем ячейки Part II (пропускаем блиц-вопросы и упражнения)
    in_part_2 = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'markdown':
            source_text = ''.join(cell.get('source', []))

            # Начало Part II
            if 'Зачем мы пилим автоград' in source_text:
                in_part_2 = True
                new_cells.append(cell)
                continue

            # Пропускаем блиц II
            if 'Блиц-вопросы Часть II' in source_text:
                in_part_2 = False
                continue

            # Пропускаем упражнения
            if 'Часть III' in source_text or 'Практические упражнения' in source_text:
                in_part_2 = False
                continue

            # Пропускаем дополнительные материалы (они будут в конце)
            if 'Дополнительные материалы' in source_text:
                in_part_2 = False
                continue

        if in_part_2:
            new_cells.append(cell)

    # ===== БЛИЦ-ВОПРОСЫ (все в конце) =====
    print("  Adding quiz section at the end...")
    new_cells.append(create_markdown_cell("""---

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
    new_cells.append(create_markdown_cell("""---

## Дополнительные материалы

* [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
* [PyTorch Tutorials](https://pytorch.org/tutorials/)
* [PyTorch Broadcasting Semantics](https://pytorch.org/docs/stable/notes/broadcasting.html)
* [Backpropagation Calculus](https://www.youtube.com/watch?v=tIeHLnjs5U8) - отличное видео от 3Blue1Brown
* [micrograd](https://github.com/karpathy/micrograd) - минималистичный autograd engine от Andrej Karpathy"""))

    # Сохраняем
    nb['cells'] = new_cells

    print(f"\n💾 Saving updated notebook ({len(new_cells)} cells)...")
    save_notebook(nb, "01_seminar_mlp_autograd.ipynb")

    print("✅ Done! Notebook updated.")
    print(f"\n📊 Statistics:")
    print(f"   Total cells: {len(new_cells)}")
    print(f"   Code cells: {len([c for c in new_cells if c['cell_type'] == 'code'])}")
    print(f"   Markdown cells: {len([c for c in new_cells if c['cell_type'] == 'markdown'])}")

if __name__ == "__main__":
    update_notebook()
