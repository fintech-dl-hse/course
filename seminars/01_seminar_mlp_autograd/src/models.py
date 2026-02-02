"""
Neural network model definitions for Manim visualizations.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_moons


# ============================================================================
# Model Directory
# ============================================================================

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================================
# Model Definitions
# ============================================================================

class MLP1Hidden(nn.Module):
    """Simple 1-hidden-layer MLP for classification."""

    def __init__(self, hidden_dim=16, activation_cls=nn.ReLU):
        super().__init__()
        self.fc1 = nn.Linear(2, int(hidden_dim))
        self.fc2 = nn.Linear(int(hidden_dim), 1)
        self.activation = activation_cls()

    def forward(self, x_coordinates):
        h = self.activation(self.fc1(x_coordinates))
        scores = self.fc2(h)
        return scores[:, 0]


class MLP2Hidden(nn.Module):
    """2-hidden-layer MLP for classification."""

    def __init__(self, hidden_dim=100, activation_cls=nn.ReLU):
        super().__init__()
        self.fc1 = nn.Linear(2, int(hidden_dim))
        self.fc2 = nn.Linear(int(hidden_dim), int(hidden_dim))
        self.fc3 = nn.Linear(int(hidden_dim), 1)
        self.activation = activation_cls()

    def forward(self, x_coordinates):
        h1 = self.activation(self.fc1(x_coordinates))
        h2 = self.activation(self.fc2(h1))
        scores = self.fc3(h2)
        return scores[:, 0]


# ============================================================================
# Loss Functions
# ============================================================================

def loss_function(model, Xbatch, ybatch):
    """Compute SVM-style hinge loss for the model."""
    Xbatch = torch.tensor(Xbatch).float()
    ybatch = torch.tensor(ybatch).float().unsqueeze(-1)
    model_prediction = model.forward(Xbatch).unsqueeze(-1)
    losses = F.relu(1 - ybatch * model_prediction)
    loss = losses.mean()
    alpha = 1e-4
    reg_loss = alpha * sum((p * p).sum() for p in model.parameters())
    total_loss = loss + reg_loss
    accuracy = ((ybatch > 0) == (model_prediction > 0)).float().mean()
    return total_loss, accuracy


# ============================================================================
# Training Functions
# ============================================================================

def train_model(model, learning_rate=0.05, n_steps=500):
    """Train an MLP model using manual SGD on moons dataset."""
    Xbatch, ybatch = make_moons(n_samples=100, noise=0.1, random_state=1)
    ybatch = ybatch * 2 - 1  # make y be -1 or 1

    for k in range(n_steps):
        model.zero_grad()
        total_loss, acc = loss_function(model, Xbatch, ybatch)
        total_loss.backward()
        for p in model.parameters():
            p.data = p.data - learning_rate * p.grad

    return model


def train_width_model(model, learning_rate=0.02, n_steps=2500):
    """Train a width-model on moons dataset using BCE loss and Adam."""
    X, y = make_moons(n_samples=100, noise=0.1, random_state=1)
    X_t = torch.tensor(X).float()
    y_t = torch.tensor(y).float()
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(n_steps):
        opt.zero_grad()
        logits = model.forward(X_t)
        loss = F.binary_cross_entropy_with_logits(logits, y_t)
        alpha = 1e-4
        reg = alpha * sum((p * p).sum() for p in model.parameters())
        (loss + reg).backward()
        opt.step()
    model.eval()
    return model


# ============================================================================
# Model Loading/Saving
# ============================================================================

WIDTH_MODEL_PATHS = {
    2: os.path.join(MODEL_DIR, "mlp_relu_width_2.pth"),
    4: os.path.join(MODEL_DIR, "mlp_relu_width_4.pth"),
    8: os.path.join(MODEL_DIR, "mlp_relu_width_8.pth"),
    16: os.path.join(MODEL_DIR, "mlp_relu_width_16.pth"),
}


def get_or_train_width_model(hidden_dim, activation_cls=nn.ReLU):
    """Lazily load or train and save MLP1Hidden model for a given hidden width."""
    if hidden_dim not in WIDTH_MODEL_PATHS:
        raise ValueError(
            f"Unsupported hidden_dim={hidden_dim}, "
            f"expected one of {sorted(WIDTH_MODEL_PATHS.keys())}"
        )
    model_path = WIDTH_MODEL_PATHS[hidden_dim]
    if os.path.exists(model_path):
        print(f"Loading width-model from {model_path}")
        model = MLP1Hidden(hidden_dim=hidden_dim, activation_cls=activation_cls)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        return model

    print(f"Training width-model (not found at {model_path})...")
    model = MLP1Hidden(hidden_dim=hidden_dim, activation_cls=activation_cls)
    model = train_width_model(model, learning_rate=0.02, n_steps=2500)
    torch.save(model.state_dict(), model_path)
    print(f"Width-model saved to {model_path}")
    model.eval()
    return model
