# Autograd Visualization Plan

## Overview

Create a Manim animation demonstrating how automatic differentiation (Autograd) works in deep learning. The visualization explains the Chain Rule, computational graph construction during forward pass, the distinction between parameters and activations, and gradient computation during backward pass.

## Target Audience

Students learning deep learning fundamentals who have seen basic neural networks but haven't understood how gradients are computed automatically.

## Key Concepts to Visualize

1. **Chain Rule** - Why Autograd works mathematically
2. **Forward Pass** - Building and saving the computation graph
3. **Parameters vs Activations** - What gets saved and why
4. **Gradient Propagation** - Computing gradients for both parameters AND layer inputs
5. **Backward Pass** - Computing gradients in reverse topological order

## Structure

### Scene 1: ChainRuleScene
**Purpose**: Establish the mathematical foundation for why automatic differentiation works.

**Content**:
- Title: "The Chain Rule: Foundation of Backpropagation"
- Show the chain rule formula: $\frac{dF}{dx} = \frac{dF}{dg} \cdot \frac{dg}{dx}$
- Visual example with simple composition: $F = f(g(x))$
- Concrete numerical example: $F = (x + 2)^2$
  - Let $g = x + 2$, $F = g^2$
  - $\frac{dF}{dx} = \frac{dF}{dg} \cdot \frac{dg}{dx} = 2g \cdot 1 = 2(x+2)$
- Animate the "chain" of derivatives connecting
- Extend to multiple compositions (3-4 layers)
- Key insight: "We can compute derivatives of complex functions by breaking them into simple parts"

**Animations**:
- Fade in formula
- Transform simple function into composite form
- Show arrows connecting derivatives
- Highlight each derivative term as it's computed
- Show the multiplication chain

**Duration**: ~30 seconds

---

### Scene 2: ComputationGraphScene
**Purpose**: Show how the computation graph is built during forward pass.

**Content**:
- Title: "Building the Computational Graph"
- Start with a simple expression: $L = (x \cdot w + b)^2$ (like a loss)
- Show input values as nodes (circles)
- Show operations as nodes (squares/diamonds)
- Build the graph step by step:
  1. Input nodes: $x$, $w$, $b$
  2. Multiply: $h_1 = x \cdot w$
  3. Add: $h_2 = h_1 + b$
  4. Square: $L = h_2^2$
- Show edges connecting nodes (data flow)
- Highlight that each node stores:
  - Its output value (forward)
  - A "backward function" for gradient computation
- Animate data flowing forward through the graph

**Visual Design**:
- Input nodes: Blue circles
- Operation nodes: Green diamonds
- Output node: Red circle
- Edges: Arrows showing data flow
- Node labels show current value

**Animations**:
- Create nodes one by one as operations happen
- Draw edges to show dependencies
- Pulse/highlight values as they're computed
- Show stored backward functions appearing

**Duration**: ~45 seconds

---

### Scene 3: ParametersVsActivationsScene
**Purpose**: Clarify the difference between parameters and activations, and why activations are saved.

**Content**:
- Title: "Parameters vs Activations"
- Side-by-side layout:
  - Left: "Parameters" (weights, biases)
  - Right: "Activations" (intermediate values)

**Parameters Panel**:
- Show: $W$, $b$ as matrices/vectors
- Labels: "Learned values", "Updated during training"
- Color: Blue (to match weight visualization convention)
- Note: "Require gradients for optimization"

**Activations Panel**:
- Show: $h_1$, $h_2$, $a_1$, $a_2$ (layer outputs)
- Labels: "Computed during forward pass", "Saved for backward pass"
- Color: Green/Orange
- Note: "Needed to compute parameter gradients"

**Visual Demonstration**:
- Show a 2-layer network: Input → FC1 → ReLU → FC2 → Output
- Animate forward pass with values flowing
- Show activations being "stored" (saved to cache boxes)
- Explain: "Without saved activations, we can't compute $\frac{\partial L}{\partial W}$"

**Formula Section**:
- For linear layer: $h = Wx + b$
- Gradient w.r.t. weights: $\frac{\partial L}{\partial W} = \frac{\partial L}{\partial h} \cdot x^T$
- Highlight: "$x^T$ is the INPUT to this layer - we MUST save it!"

**Animations**:
- Split screen reveal
- Forward pass animation with value storage
- Flash saved values when explaining their necessity

**Duration**: ~50 seconds

---

### Scene 4: GradientPropagationScene
**Purpose**: Show that gradients are computed for BOTH parameters AND layer inputs.

**Content**:
- Title: "Gradient Propagation: Parameters and Inputs"
- Show a simple 2-layer network
- For each layer, show two gradient computations:

**Layer Visualization**:
```
[Input x] → [Layer 1: W1, b1] → [h1] → [Layer 2: W2, b2] → [output]
```

**For Layer 2**:
- Gradient w.r.t. W2: $\frac{\partial L}{\partial W_2} = \frac{\partial L}{\partial out} \cdot h_1^T$ → Used for optimization
- Gradient w.r.t. input (h1): $\frac{\partial L}{\partial h_1} = W_2^T \cdot \frac{\partial L}{\partial out}$ → Propagated backward

**For Layer 1**:
- Receives $\frac{\partial L}{\partial h_1}$ from Layer 2
- Gradient w.r.t. W1: $\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial h_1} \cdot x^T$ → Used for optimization
- Gradient w.r.t. input (x): $\frac{\partial L}{\partial x} = W_1^T \cdot \frac{\partial L}{\partial h_1}$ → Could be propagated further

**Key Insight**:
- "Each layer computes TWO gradients"
- "One for its parameters (to update weights)"
- "One for its input (to propagate backward)"

**Animations**:
- Show gradient flow arrows going backward
- Split each layer into two gradient computations
- Color-code: Blue for parameter gradients, Red for input gradients
- Animate the chain of propagation

**Duration**: ~50 seconds

---

### Scene 5: BackwardPassScene
**Purpose**: Demonstrate the complete backward pass with gradient computation.

**Content**:
- Title: "Backward Pass: Computing Gradients"
- Use the same computation graph from Scene 2
- Start from the loss node
- Process nodes in reverse topological order

**Step-by-step Backward**:
1. Start at loss L: $\frac{\partial L}{\partial L} = 1$
2. Back through square: $\frac{\partial L}{\partial h_2} = 2h_2$
3. Back through add: $\frac{\partial L}{\partial h_1} = 1$, $\frac{\partial L}{\partial b} = 1$
4. Back through multiply: $\frac{\partial L}{\partial x} = w$, $\frac{\partial L}{\partial w} = x$

**Visual Effects**:
- Gradient values appear next to each node
- Edges pulse in reverse order
- Accumulated gradients shown (for nodes with multiple outputs)

**Code Parallel** (optional small panel):
```python
# Forward
h1 = x * w
h2 = h1 + b
L = h2 ** 2

# Backward
dL_dL = 1
dL_dh2 = 2 * h2 * dL_dL
dL_dh1 = 1 * dL_dh2
dL_db = 1 * dL_dh2
dL_dx = w * dL_dh1
dL_dw = x * dL_dh1
```

**Animations**:
- Red glow moving backward through graph
- Gradient values appearing and accumulating
- Flash final gradients for parameters (w, b)

**Duration**: ~45 seconds

---

### Scene 6: MLPAutogradScene
**Purpose**: Apply autograd concepts to a real MLP on the moons dataset.

**Content**:
- Title: "Autograd in Action: Training an MLP"
- Left panel: Moons dataset with decision boundary
- Right panel: Network architecture (2 → 4 → 1)

**Training Loop Visualization**:
1. **Forward Pass**:
   - Show data point flowing through network
   - Highlight activations being saved
   - Show prediction and loss computation

2. **Backward Pass**:
   - Show gradient flowing back
   - Update parameter gradient accumulators
   - Visualize gradient magnitude (color intensity)

3. **Parameter Update**:
   - Show SGD step: $w = w - \eta \cdot \nabla_w L$
   - Decision boundary shifts slightly

4. **Repeat** (show 3-4 iterations)

**Final State**:
- Show learned decision boundary
- Summary: "Autograd enables efficient training by automatically computing all gradients"

**Animations**:
- Forward/backward flow animations
- Decision boundary morphing
- Loss value decreasing

**Duration**: ~60 seconds

---

### Scene 7: SummaryScene
**Purpose**: Recap the key insights.

**Content**:
- Title: "Automatic Differentiation: Key Takeaways"

**Summary Points** (appear one by one):
1. "Chain Rule enables computing derivatives of complex functions"
2. "Forward pass builds the computation graph and saves activations"
3. "Activations are saved because they're needed for gradient computation"
4. "Each layer computes gradients for BOTH parameters AND inputs"
5. "Backward pass processes nodes in reverse topological order"

**Final Visualization**:
- Small diagram showing forward and backward arrows through a network
- "This is how PyTorch's autograd works!"

**Animations**:
- Points fade in sequentially
- Final diagram assembles

**Duration**: ~20 seconds

---

## Technical Implementation

### File Structure
```
.manim/
├── src/
│   ├── __init__.py
│   ├── models.py          # MLP model definitions
│   ├── data.py            # Data utilities
│   ├── visualization.py   # Manim visualization helpers
│   └── autograd.py        # Simple autograd implementation
├── plans/
│   └── 01_autograd_visualization.md  # This file
└── 01_autograd_visualization.py      # Main animation code
```

### Classes (all inherit from InteractiveScene)
1. `ChainRuleScene`
2. `ComputationGraphScene`
3. `ParametersVsActivationsScene`
4. `GradientPropagationScene`
5. `BackwardPassScene`
6. `MLPAutogradScene`
7. `SummaryScene`

### Shared Components
- `ComputationNode`: VGroup representing a node in the computation graph
- `ComputationEdge`: Arrow with optional gradient label
- `ComputationGraph`: VGroup managing nodes and edges
- `GradientFlow`: Animation for visualizing gradient propagation

### Color Scheme
- **Input values**: BLUE
- **Parameters**: BLUE_E
- **Operations**: GREEN
- **Activations**: ORANGE
- **Gradients**: RED
- **Loss**: RED_E
- **Positive values**: BLUE gradient
- **Negative values**: RED gradient

### Animation Timing
- Each scene should be self-contained
- Use consistent animation speeds (run_time=1.5 for transforms, 0.8 for fades)
- Add appropriate waits between major steps

## Running the Animation

```bash
# Activate conda environment
conda activate manim

# Run specific scene
~/miniconda3/envs/manim/bin/manimgl /Users/d.tarasov/workspace/hse/fintech-dl-hse/course/.manim/01_autograd_visualization.py ChainRuleScene

# Run all scenes
~/miniconda3/envs/manim/bin/manimgl /Users/d.tarasov/workspace/hse/fintech-dl-hse/course/.manim/01_autograd_visualization.py
```

## Success Criteria

1. Chain rule is clearly explained with visual examples
2. Computation graph construction is intuitive to follow
3. Distinction between parameters and activations is clear
4. Gradient propagation to both parameters and inputs is demonstrated
5. Backward pass order (reverse topological) is visualized
6. Real MLP training shows autograd in action
7. All text is in English
8. Animations are smooth and educational

## References

- [micrograd by Andrej Karpathy](https://github.com/karpathy/micrograd)
- [PyTorch Autograd Tutorial](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
- [3Blue1Brown Backpropagation video](https://www.youtube.com/watch?v=tIeHLnjs5U8)
- Course seminar: `01_seminar_mlp_autograd.ipynb`
