# MLP Visualization Plan

## Overview
Create a Manim animation demonstrating the evolution from linear transformations to Multi-Layer Perceptrons (MLPs), using real trained models on the moons dataset. The animation shows how nonlinearity enables MLPs to learn complex decision boundaries.

## Structure

### Introduction Scene
- Title: "From Linear Transformations to MLPs"
- Subtitle: "Understanding Multi-Layer Perceptrons"
- Sets the context for the educational animation

### Scene 1: Linear Transformation on 2D Plane
- **Visualization**: Show 2D data points (moons dataset) on a plane
- **Transformation**: Apply a 2x2 linear transformation matrix
- **Effect**: Show how points move/transform but remain linearly separable
- **Text**: "Linear transformation: f(x) = Wx + b"
- **Key insight**: Linear transformations preserve linear separability

### Scene 2: Dimensionality Expansion
- **Visualization**: Transform 2D points to higher dimension (2D → 3D)
- **Show**: How adding dimensions can help separate data
- **Text**: "Expanding dimensions: 2D → higher dimensional space"
- **Key insight**: More dimensions provide more flexibility

### Scene 3: Introducing Nonlinearity - MLP
- **Visualization**: Show full MLP architecture (2 → 100 → 100 → 1)
- **Animate**: Data flow through layers with ReLU activations
- **Show**: Decision boundary learned by trained MLP
- **Text**: "Adding nonlinearity: MLP with ReLU activations"
- **Key insight**: Nonlinearity enables learning complex boundaries

### Scene 4: Without Nonlinearity
- **Visualization**: Show same MLP but with Identity activation (linear)
- **Compare**: Decision boundaries side-by-side (MLP with ReLU vs MLP without)
- **Text**: "Without nonlinearity: Composition of linear = linear"
- **Key insight**: Multiple linear layers collapse to single linear transformation

### Conclusion Scene
- Summary of key takeaways:
  1. Linear transformations preserve linear separability
  2. Adding dimensions provides more flexibility
  3. Nonlinearity enables learning complex boundaries
  4. Composition of linear layers = single linear layer

## Implementation Details

### Code Structure
- **File**: `course/.manim/01_mlp_visualization.py`
- **Base class**: `InteractiveScene` from manimlib
- **Imports**: Use `manim_imports_ext` and helper classes from `_2024.transformers.helpers`

### Model Training
- Extract MLP class and training code from `course/seminars/01_seminar_mlp_autograd.ipynb`
- Train two models:
  1. MLP with ReLU activation (full model)
  2. MLP with Identity activation (linear baseline)
- Use `make_moons(n_samples=100, noise=0.1, random_state=1)` dataset
- Models are trained during animation initialization

### Visualization Components

1. **2D Data Visualization**
   - Use `Axes` for coordinate system
   - Scatter plot of moons dataset with color coding (two classes)
   - Decision boundary as filled regions using polygons

2. **Network Architecture**
   - Use `NeuralNetwork` class from helpers
   - Show layers as groups of dots/neurons
   - Animate data flow through connections

3. **Transformation Visualization**
   - For linear transformation: show matrix multiplication effect on points
   - For dimensionality expansion: show projection to higher space (3D view)
   - Use `ThreeDAxes` for 3D visualizations

4. **Decision Boundaries**
   - Generate mesh grid for visualization
   - Create filled regions using polygons from mesh grid
   - Overlay data points on decision boundaries

### Animation Flow

1. **Introduction** (3-5 seconds)
   - Title and subtitle
   - Sets context

2. **Scene 1: Linear Transformation** (15-20 seconds)
   - Show 2D points
   - Apply linear transformation
   - Show transformed points
   - Display insight about linear separability

3. **Scene 2: Dimensionality Expansion** (15-20 seconds)
   - Show 2D → 3D transformation
   - Visualize in 3D space
   - Show how separation becomes easier

4. **Scene 3: MLP with Nonlinearity** (20-30 seconds)
   - Show MLP architecture
   - Animate forward pass
   - Show learned decision boundary
   - Highlight nonlinear regions

5. **Scene 4: Comparison** (15-20 seconds)
   - Side-by-side comparison
   - MLP with ReLU vs MLP without
   - Emphasize that linear composition = linear

6. **Conclusion** (5-8 seconds)
   - Summary of key takeaways

## Technical Requirements

### Dependencies
- PyTorch (for model training)
- scikit-learn (for make_moons)
- numpy (for data manipulation)
- manimlib (for animation)

### Model Architecture
```python
class MLP(nn.Module):
    def __init__(self, activation_cls=nn.ReLU):
        self.input_layer = nn.Linear(2, 100)
        self.hidden_layer = nn.Linear(100, 100)
        self.output_layer = nn.Linear(100, 1)
        self.activation = activation_cls()
```

### Training
- Use training code from notebook
- Train both ReLU and Identity versions
- Models converge during initialization
- Training happens before animation starts

### Visualization Functions
- `get_decision_boundary()`: Generate mesh grid and compute predictions
- `create_decision_boundary_mobject()`: Create Manim polygons from mesh grid
- `create_data_points()`: Create Manim dots for data points

## File Organization

- **Plan**: `course/.manim/plans/01_mlp_visualization.md`
- **Code**: `course/.manim/01_mlp_visualization.py`

## Key Features

- Real trained MLPs showing actual learned boundaries
- Smooth transitions between scenes
- Educational text labels in English
- Visual demonstration of key concepts
- Comparison of linear vs nonlinear models

## Success Criteria

- Clear progression from linear → dimensional expansion → nonlinear MLP
- Visual demonstration that linear composition = linear
- Real trained models showing actual learned boundaries
- Smooth, educational animations
- All text in English

