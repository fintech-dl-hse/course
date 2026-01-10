#!/usr/bin/env python3
"""
Тестирование ключевых частей объединенного ноутбука.
"""

import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("ТЕСТИРОВАНИЕ СЕМИНАРСКОГО НОУТБУКА")
print("="*60)

# Test Part I: PyTorch MLP
print("\n📦 Part I: PyTorch MLP")
print("-" * 60)

print("✓ Importing libraries...")
from typing import Optional, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_moons

print("✓ Generating data...")
X, y = make_moons(n_samples=100, noise=0.1, random_state=1)
print(f"  Data shape: X={X.shape}, y={y.shape}")

print("✓ Defining MLP model...")
class MLP(nn.Module):
    def __init__(self, activation_cls: Callable = nn.Sigmoid):
        super().__init__()
        self.input_layer = nn.Linear(2, 100)
        self.hidden_layer = nn.Linear(100, 100)
        self.output_layer = nn.Linear(100, 1)
        self.activation = activation_cls()

    def forward(self, x_coordinates: torch.Tensor) -> torch.Tensor:
        latents = self.activation(self.input_layer(x_coordinates))
        latents = self.activation(self.hidden_layer(latents))
        scores = self.output_layer(latents)
        scores = scores[:, 0]
        return scores

model = MLP()
num_params = sum(p.numel() for p in model.parameters())
print(f"  Model parameters: {num_params}")
assert num_params == 10501, f"Expected 10501 parameters, got {num_params}"

print("✓ Defining loss function...")
def loss(model: nn.Module, Xbatch: np.ndarray, ybatch: np.ndarray, alpha: float = 1e-4):
    Xbatch = torch.tensor(Xbatch, dtype=torch.float32)
    ybatch = torch.tensor(ybatch, dtype=torch.float32).unsqueeze(-1)
    model_prediction = model.forward(Xbatch)
    losses = F.relu(1 - ybatch * model_prediction.unsqueeze(-1))
    data_loss = losses.mean()
    reg_loss = alpha * sum((p * p).sum() for p in model.parameters())
    total_loss = data_loss + reg_loss
    accuracy = ((ybatch > 0) == (model_prediction.unsqueeze(-1) > 0)).float().mean()
    return total_loss, accuracy

y_test = y * 2 - 1  # Convert to -1, 1
total_loss, acc = loss(model, X, y_test)
print(f"  Initial loss: {total_loss.item():.4f}, accuracy: {acc.item():.2%}")

print("✓ Testing training function...")
def train(model: nn.Module, learning_rate: float = 0.1, num_steps: int = 50):
    Xbatch, ybatch = make_moons(n_samples=100, noise=0.1, random_state=1)
    ybatch = ybatch * 2 - 1
    for k in range(num_steps):
        model.zero_grad()
        total_loss, acc = loss(model, Xbatch, ybatch)
        total_loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                p.data = p.data - learning_rate * p.grad
        if k == 0 or k == num_steps - 1:
            print(f"  Step {k:3d}: loss = {total_loss.item():.4f}, accuracy = {acc.item():.2%}")

train(model, learning_rate=0.1, num_steps=50)

# Test Part II: Autograd
print("\n🔧 Part II: Custom Autograd")
print("-" * 60)

print("✓ Defining Value class...")
class Value:
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
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

print("✓ Testing custom autograd...")
x = Value(2.0)
y = Value(3.0)
z = x + y  # z = 5
f = z * x  # f = 10

print(f"  f = {f.data}")
assert f.data == 10.0, f"Expected f=10, got {f.data}"

f.backward()
print(f"  df/dx = {x.grad} (expected: 7.0)")
print(f"  df/dy = {y.grad} (expected: 2.0)")
assert abs(x.grad - 7.0) < 1e-6, f"Expected df/dx=7, got {x.grad}"
assert abs(y.grad - 2.0) < 1e-6, f"Expected df/dy=2, got {y.grad}"

# Test ReLU
print("✓ Testing ReLU...")
x_neg = Value(-1.0)
x_pos = Value(2.0)
y_neg = x_neg.relu()
y_pos = x_pos.relu()
print(f"  ReLU(-1.0) = {y_neg.data} (expected: 0.0)")
print(f"  ReLU(2.0) = {y_pos.data} (expected: 2.0)")
assert y_neg.data == 0.0, f"Expected ReLU(-1)=0, got {y_neg.data}"
assert y_pos.data == 2.0, f"Expected ReLU(2)=2, got {y_pos.data}"

print("\n" + "="*60)
print("✅ ВСЕ ТЕСТЫ ПРОШЛИ УСПЕШНО!")
print("="*60)
print("\n📝 Следующие шаги:")
print("  1. Открыть ноутбук в Jupyter:")
print("     ~/miniconda3/envs/audio/bin/jupyter notebook 01_seminar_mlp_autograd.ipynb")
print("  2. Запустить все ячейки (Restart & Run All)")
print("  3. Проверить визуализации")
print("  4. При необходимости доработать детали")
