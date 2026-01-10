#!/usr/bin/env python3
"""
Скрипт для модернизации кода в объединенном ноутбуке.
Добавляет type hints, docstrings и улучшает комментарии.
"""

import json
from pathlib import Path
from typing import Dict

def load_notebook(path: Path) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_notebook(notebook: Dict, path: Path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)
    print(f"✅ Notebook saved to: {path}")

# Modernized MLP class with type hints and docstrings
MODERN_MLP_CODE = """from typing import Optional, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    \"\"\"
    Multi-Layer Perceptron для бинарной классификации.

    Архитектура:
        input (2) -> hidden (100) -> hidden (100) -> output (1)

    Args:
        activation_cls: Класс функции активации (по умолчанию nn.Sigmoid)
    \"\"\"

    def __init__(self, activation_cls: Callable = nn.Sigmoid):
        super().__init__()

        # Слои сети
        self.input_layer = nn.Linear(2, 100)    # [batch, 2] -> [batch, 100]
        self.hidden_layer = nn.Linear(100, 100)  # [batch, 100] -> [batch, 100]
        self.output_layer = nn.Linear(100, 1)    # [batch, 100] -> [batch, 1]
        self.activation = activation_cls()

    def forward(self, x_coordinates: torch.Tensor) -> torch.Tensor:
        \"\"\"
        Forward pass через сеть.

        Args:
            x_coordinates: Входные координаты точек, shape [batch_size, 2]

        Returns:
            scores: Предсказанные скоры, shape [batch_size]
        \"\"\"
        latents = self.activation(self.input_layer(x_coordinates))  # [batch_size, 100]
        latents = self.activation(self.hidden_layer(latents))       # [batch_size, 100]
        scores = self.output_layer(latents)                         # [batch_size, 1]
        scores = scores[:, 0]                                       # [batch_size]

        return scores


# Создаем модель
model = MLP()
print(model)
print("Количество параметров:", sum(p.numel() for p in model.parameters()))
"""

# Modernized loss function
MODERN_LOSS_CODE = """def loss(model: nn.Module,
         Xbatch: np.ndarray,
         ybatch: np.ndarray,
         alpha: float = 1e-4) -> tuple[torch.Tensor, torch.Tensor]:
    \"\"\"
    Вычисляет SVM-style max-margin loss с L2 регуляризацией.

    Args:
        model: Модель PyTorch
        Xbatch: Батч входных данных, shape [batch_size, 2]
        ybatch: Батч меток (0 или 1), shape [batch_size]
        alpha: Коэффициент L2 регуляризации

    Returns:
        total_loss: Суммарный loss (data loss + regularization)
        accuracy: Точность на батче
    \"\"\"
    # Конвертируем numpy -> torch
    Xbatch = torch.tensor(Xbatch, dtype=torch.float32)  # [batch_size, 2]
    ybatch = torch.tensor(ybatch, dtype=torch.float32).unsqueeze(-1)  # [batch_size, 1]

    # Forward pass для получения предсказаний
    model_prediction = model.forward(Xbatch)  # [batch_size]

    # SVM max-margin loss: max(0, 1 - y_true * y_pred)
    losses = F.relu(1 - ybatch * model_prediction.unsqueeze(-1))  # [batch_size, 1]
    data_loss = losses.mean()

    # L2 regularization: alpha * sum(w^2)
    reg_loss = alpha * sum((p * p).sum() for p in model.parameters())
    total_loss = data_loss + reg_loss

    # Accuracy
    accuracy = ((ybatch > 0) == (model_prediction.unsqueeze(-1) > 0)).float().mean()

    return total_loss, accuracy

# Пример использования
total_loss, acc = loss(model, X, y)
print(f"Loss: {total_loss:.4f}, Accuracy: {acc:.2%}")
"""

# Modernized train function
MODERN_TRAIN_CODE = """def train(model: nn.Module,
          learning_rate: float = 0.1,
          num_steps: int = 300):
    \"\"\"
    Обучает модель используя ручной SGD.

    Args:
        model: Модель для обучения
        learning_rate: Скорость обучения (learning rate)
        num_steps: Количество шагов оптимизации
    \"\"\"
    # Генерируем тренировочные данные
    Xbatch, ybatch = make_moons(n_samples=100, noise=0.1, random_state=1)
    ybatch = ybatch * 2 - 1  # Конвертируем метки: 0,1 -> -1,1

    for k in range(num_steps):
        # 1. Обнуляем градиенты с предыдущего шага
        model.zero_grad()

        # 2. Forward pass: вычисляем loss
        total_loss, acc = loss(model, Xbatch, ybatch)

        # 3. Backward pass: вычисляем градиенты
        # Здесь происходит "магия" - PyTorch автоматически вычисляет
        # градиенты по всем параметрам используя backpropagation!
        total_loss.backward()

        # 4. Шаг оптимизации: обновляем веса
        # Градиентный спуск: w_new = w_old - lr * gradient
        with torch.no_grad():  # Отключаем autograd для обновления весов
            for p in model.parameters():
                p.data = p.data - learning_rate * p.grad

        # Логирование
        if k % 50 == 0:
            print(f"Step {k:3d}: loss = {total_loss.item():.4f}, accuracy = {acc.item():.2%}")


# Обучаем модель
model = MLP()
train(model, learning_rate=0.1)
"""

def modernize_notebook():
    """Modernize code in the merged notebook."""

    notebook_path = Path("01_seminar_mlp_autograd.ipynb")

    print(f"📖 Loading {notebook_path}...")
    nb = load_notebook(notebook_path)

    print("🔧 Modernizing code...")

    # Find and replace MLP class definition
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell.get('source', []))

            # Replace MLP class
            if 'class MLP(nn.Module):' in source and 'def __init__(self' in source:
                print(f"  Updating MLP class at cell {i}")
                cell['source'] = MODERN_MLP_CODE.split('\n')

            # Replace loss function
            elif 'def loss(model, Xbatch, ybatch):' in source:
                print(f"  Updating loss function at cell {i}")
                cell['source'] = MODERN_LOSS_CODE.split('\n')

            # Replace train function
            elif 'def train(model, learning_rate' in source:
                print(f"  Updating train function at cell {i}")
                cell['source'] = MODERN_TRAIN_CODE.split('\n')

    # Save
    print(f"\n💾 Saving modernized notebook...")
    save_notebook(nb, notebook_path)

    print("\n✨ Done! Code has been modernized.")
    print("\n📝 Changes made:")
    print("   - Added type hints to MLP class, loss, and train functions")
    print("   - Added comprehensive docstrings")
    print("   - Improved code comments")
    print("   - Better variable names and structure")

if __name__ == "__main__":
    modernize_notebook()
