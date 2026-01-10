# Семинары

## Семинар 1: MLP на PyTorch и автоматическое дифференцирование

**Файл:** `01_seminar_mlp_autograd.ipynb`

**Дата:** 19 января 2026

**Темы:**
- Введение в глубинное обучение
- Обучение нейросетей
- Алгоритм обратного распространения ошибки

### Содержание

#### Часть I: PyTorch MLP
- **Знакомство с PyTorch:** введение в библиотеку глубокого обучения
  - Аналогия с NumPy (tensors vs arrays)
  - Основные операции с тензорами
  - Broadcasting и его правила ([PyTorch Broadcasting](https://pytorch.org/docs/stable/notes/broadcasting.html))
  - Сравнение PyTorch и NumPy
- **Работа с данными:** генерация и визуализация датасета make_moons
- **Определение модели MLP:** создание многослойного перцептрона на PyTorch
- **Функции потерь:** SVM max-margin loss с L2 регуляризацией
- **Обучение модели:** ручная реализация SGD
- **Роль нелинейностей:** сравнение ReLU, Sigmoid и Identity
- **Сравнение с SVM:** LinearSVC, polynomial и RBF kernels
- **Батчинг и эффективность:** демонстрация ~80-90x ускорения

#### Часть II: Autograd и Backpropagation
- **Мотивация:** зачем нужно автоматическое дифференцирование
- **Forward и backward pass:** объяснение двух ключевых этапов
- **Chain rule:** математическая основа backpropagation
- **Матричное дифференцирование:** основы matrix calculus для deep learning
  - Обозначения и типы производных (градиент, якобиан)
  - Основные правила дифференцирования (линейные формы, квадратичные формы, нормы)
  - Производные матрично-векторных операций
  - Примеры для нейросетей (линейный слой, MSE loss)
  - Полезные ресурсы (Matrix Calculus, Matrix Cookbook)
- **Примеры в PyTorch:** как сложение и умножение влияют на градиенты
- **Собственный autograd:** реализация класса Value с нуля
  - Python magic methods (`__add__`, `__mul__`)
  - Closures для хранения градиентных функций
  - Топологическая сортировка для backward pass
- **Блиц-вопросы:** закрепление знаний об autograd (все вопросы собраны в конце ноутбука)

### Технические детали

- **Язык:** Python 3.10+
- **Основные библиотеки:**
  - PyTorch 2.0+
  - NumPy 1.24+
  - Matplotlib 3.7+
  - scikit-learn 1.3+

- **Особенности кода:**
  - Type hints для всех функций
  - Подробные docstrings
  - Современный Python стиль

### Запуск семинара

```bash
# Открыть в Jupyter
~/miniconda3/envs/audio/bin/jupyter notebook 01_seminar_mlp_autograd.ipynb

# Или конвертировать в Python скрипт
~/miniconda3/envs/audio/bin/jupyter nbconvert --to script 01_seminar_mlp_autograd.ipynb
```

### Связанные домашние задания

- `mlp` (1 балл)
- `weight-init` (1 балл)
- `activations` (1 балл)

### Дополнительные материалы

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Backpropagation Calculus](https://www.youtube.com/watch?v=tIeHLnjs5U8) - 3Blue1Brown
- [micrograd](https://github.com/karpathy/micrograd) - Andrej Karpathy

---

## Структура репозитория семинаров

```
seminars/
├── 01_seminar_mlp_autograd.ipynb  # Семинар 1 (финальная версия)
├── old_01_seminar_torch_mlp.ipynb # Архив: старый семинар 1
├── old_02_seminar_autograd.ipynb  # Архив: старый семинар 2
├── merge_notebooks.py             # Скрипт объединения ноутбуков
├── rebuild_notebook.py            # Скрипт пересоздания с правильным форматированием
├── modernize_code.py              # Скрипт модернизации кода
├── test_notebook.py               # Автоматические тесты
└── rm_widgets.sh                  # Очистка widget metadata
```

## Педагогическая логика

Семинары следуют принципу "**снаружи внутрь**":
1. Сначала учимся **использовать** инструменты (PyTorch, нейросети)
2. Потом понимаем **как они работают** изнутри (autograd, backpropagation)
3. Наконец, **практикуемся** самостоятельно (упражнения)

Оценка времени на семинар: ~2 часа (стандартная "пара")
