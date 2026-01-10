# MLP Visualization Scenes

Educational Manim animations for teaching Multi-Layer Perceptron concepts in the Deep Learning course.

## File Location
`.manim/mlp_visualization.py`

## Available Scenes

### 1. MLPDemo
**Quick demo scene combining key concepts**
- Network architecture visualization
- Mathematical equation
- Key message about deep learning

```bash
~/miniconda3/envs/manim/bin/manimgl .manim/mlp_visualization.py MLPDemo
```

### 2. LinearTransformationScene
**Linear transformations (Wx + b) basics**
- 2D coordinate plane with data points
- Weight matrix and bias visualization
- Shows limitation: only straight decision boundaries
- Demonstrates linearly separable data

```bash
~/miniconda3/envs/manim/bin/manimgl .manim/mlp_visualization.py LinearTransformationScene
```

### 3. ActivationFunctionScene
**Non-linearity through activation functions**
- XOR pattern (non-linearly separable data)
- ReLU and Sigmoid function graphs
- Shows how activations enable curved boundaries
- Visualizes "folding" of space

```bash
~/miniconda3/envs/manim/bin/manimgl .manim/mlp_visualization.py ActivationFunctionScene
```

### 4. SingleLayerMLPScene
**Complete 2-layer network demonstration**
- Network architecture (2 → 6 → 2)
- Layer labels (Input, Hidden, Output)
- Animated forward pass
- Mathematical equation display

```bash
~/miniconda3/envs/manim/bin/manimgl .manim/mlp_visualization.py SingleLayerMLPScene
```

### 5. DeepMLPEvolutionScene
**Comparing networks of different depths**
- Side-by-side comparison (1, 2, 3, 4 layers)
- Spiral dataset visualization
- Shows progression: deeper → more complex boundaries
- Warning about overfitting

```bash
~/miniconda3/envs/manim/bin/manimgl .manim/mlp_visualization.py DeepMLPEvolutionScene
```

### 6. UniversalApproximationScene
**Universal Approximation Theorem demonstration**
- Target function: sin(x) + 0.5*sin(3x)
- Progressive approximation with 2, 5, 10, 20 neurons
- Shows increasing accuracy with more neurons
- Educational message about function approximation

```bash
~/miniconda3/envs/manim/bin/manimgl .manim/mlp_visualization.py UniversalApproximationScene
```

### 7. MLPComponentsScene
**Educational breakdown of MLP building blocks**
- Component breakdown: x, W, b, σ, y
- Layer stacking visualization
- Flow from input through multiple layers to output
- Composition insight

```bash
~/miniconda3/envs/manim/bin/manimgl .manim/mlp_visualization.py MLPComponentsScene
```

## Interactive Controls

When a scene is running:
- **d** - Toggle visibility of elements
- **f** - Fast forward
- **z** - Rewind
- **Cmd+Q** or **ESC** - Quit

## Technical Details

### Dependencies
- Uses 3b1b's Manim version (manimgl)
- Imports from `/Users/d.tarasov/workspace/hse/fintech-dl-hse/videos`
- Leverages helper classes: `NeuralNetwork`, `WeightMatrix`, `value_to_color`

### Helper Classes Created
- `DataPoints` - Visualize 2D data points with colors
- `DecisionBoundary` - Decision boundary for binary classification
- Helper functions:
  - `generate_spiral_data()` - Spiral dataset for classification
  - `generate_xor_data()` - XOR pattern dataset

### Color Scheme
- Blue: Positive weights / Class 1
- Red: Negative weights / Class 2
- Green: Target functions / ReLU
- Yellow: Decision boundaries / Highlights
- Purple/Orange: Variations for comparisons

## Output Location

Videos are rendered to:
`.manim/media/videos/mlp_visualization/`

## Educational Goals

These scenes teach students:
1. Linear transformations create straight boundaries
2. Activation functions enable non-linearity
3. Hidden layers build complex representations
4. Depth increases boundary complexity
5. MLPs approximate arbitrary functions
6. Network = composition of simple operations

## Notes

- All scenes use `InteractiveScene` for exploration
- Animations are 2-5 minutes each
- Mathematical notation included throughout
- Follows repository's color conventions
- Tested with Python 3.10 and manimgl v1.7.2
