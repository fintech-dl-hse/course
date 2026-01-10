"""
MLP (Multi-Layer Perceptron) Visualization for Deep Learning Course
Educational animations demonstrating MLP concepts from basic linear transformations
to deep networks and universal approximation.
"""

import sys
sys.path.append('/Users/d.tarasov/workspace/hse/fintech-dl-hse/videos')

from manim_imports_ext import *
from _2024.transformers.helpers import NeuralNetwork, WeightMatrix, value_to_color

# ============================================================================
# Helper Classes
# ============================================================================

class DataPoints(VGroup):
    """Visualize 2D data points with colors"""
    def __init__(self, points, colors=None, radius=0.08):
        super().__init__()
        if colors is None:
            colors = [BLUE] * len(points)
        for point, color in zip(points, colors):
            dot = Dot(point, radius=radius)
            dot.set_fill(color, opacity=0.8)
            dot.set_stroke(WHITE, 1)
            self.add(dot)


class DecisionBoundary(VMobject):
    """Visualize decision boundary for binary classification"""
    def __init__(self, func, x_range=(-3, 3), y_range=(-3, 3), resolution=50):
        super().__init__()
        self.func = func
        self.x_range = x_range
        self.y_range = y_range
        self.resolution = resolution

    def generate_boundary(self):
        """Generate contour line for decision boundary"""
        x = np.linspace(self.x_range[0], self.x_range[1], self.resolution)
        y = np.linspace(self.y_range[0], self.y_range[1], self.resolution)
        X, Y = np.meshgrid(x, y)

        Z = np.zeros_like(X)
        for i in range(self.resolution):
            for j in range(self.resolution):
                Z[i, j] = self.func(np.array([X[i, j], Y[i, j]]))

        return X, Y, Z


def generate_spiral_data(n_points=100, noise=0.1):
    """Generate spiral dataset for classification"""
    theta = np.linspace(0, 4 * np.pi, n_points // 2)
    r = np.linspace(0.5, 2, n_points // 2)

    # First spiral
    x1 = r * np.cos(theta) + np.random.randn(n_points // 2) * noise
    y1 = r * np.sin(theta) + np.random.randn(n_points // 2) * noise

    # Second spiral (offset by pi)
    x2 = r * np.cos(theta + np.pi) + np.random.randn(n_points // 2) * noise
    y2 = r * np.sin(theta + np.pi) + np.random.randn(n_points // 2) * noise

    points1 = [[x1[i], y1[i], 0] for i in range(len(x1))]
    points2 = [[x2[i], y2[i], 0] for i in range(len(x2))]

    return points1, points2


def generate_xor_data(n_points=50):
    """Generate XOR-like dataset"""
    points_per_quadrant = n_points // 4

    # Class 1: top-right and bottom-left
    class1 = []
    for _ in range(points_per_quadrant):
        class1.append([np.random.uniform(0.5, 2), np.random.uniform(0.5, 2), 0])
        class1.append([np.random.uniform(-2, -0.5), np.random.uniform(-2, -0.5), 0])

    # Class 2: top-left and bottom-right
    class2 = []
    for _ in range(points_per_quadrant):
        class2.append([np.random.uniform(-2, -0.5), np.random.uniform(0.5, 2), 0])
        class2.append([np.random.uniform(0.5, 2), np.random.uniform(-2, -0.5), 0])

    return class1, class2


# ============================================================================
# Scene 1: Linear Transformation
# ============================================================================

class LinearTransformationScene(InteractiveScene):
    """Show how a single linear layer transforms 2D input space"""

    def construct(self):
        # Title
        title = Text("Linear Transformation: Wx + b", font_size=60)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()

        # Create coordinate plane
        plane = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            background_line_style={
                "stroke_color": GREY_C,
                "stroke_width": 1,
            }
        )
        plane.scale(0.8).shift(LEFT * 3)

        # Generate two classes of data (linearly separable)
        class1_points = [[np.random.uniform(-2, 0), np.random.uniform(-2, 2), 0]
                        for _ in range(30)]
        class2_points = [[np.random.uniform(0, 2), np.random.uniform(-2, 2), 0]
                        for _ in range(30)]

        class1_colors = [BLUE] * len(class1_points)
        class2_colors = [RED] * len(class2_points)

        data_points = DataPoints(class1_points + class2_points,
                                class1_colors + class2_colors)
        data_points.move_to(plane.get_center())

        self.play(ShowCreation(plane), FadeIn(data_points))
        self.wait()

        # Show weight matrix and bias
        matrix_label = Text("Weight Matrix W:", font_size=36)
        W = WeightMatrix(
            values=np.array([[0.8, -0.5], [0.3, 0.9]]),
            shape=(2, 2),
            num_decimal_places=1,
        )
        W.scale(0.6)

        bias_label = Text("Bias b:", font_size=36)
        b = WeightMatrix(
            values=np.array([[0.2], [-0.3]]),
            shape=(2, 1),
            num_decimal_places=1,
        )
        b.scale(0.6)

        matrix_group = VGroup(matrix_label, W, bias_label, b)
        matrix_group.arrange(DOWN, buff=0.3)
        matrix_group.to_edge(RIGHT, buff=1)

        self.play(
            Write(matrix_label),
            FadeIn(W),
            Write(bias_label),
            FadeIn(b)
        )
        self.wait()

        # Add explanation
        explanation = Text(
            "Linear transformation can only create\nstraight decision boundaries",
            font_size=32,
            color=YELLOW
        )
        explanation.to_edge(DOWN)
        self.play(Write(explanation))
        self.wait(2)

        # Show decision boundary (a straight line)
        boundary_line = Line(
            plane.c2p(-1, -2),
            plane.c2p(-1, 2),
            color=YELLOW,
            stroke_width=4
        )

        boundary_label = Text("Decision Boundary", font_size=28, color=YELLOW)
        boundary_label.next_to(boundary_line, RIGHT, buff=0.2)

        self.play(
            ShowCreation(boundary_line),
            Write(boundary_label)
        )
        self.wait(2)

        # Fade out
        self.play(
            *[FadeOut(mob) for mob in [title, plane, data_points, matrix_group,
                                       explanation, boundary_line, boundary_label]]
        )


# ============================================================================
# Scene 2: Activation Functions
# ============================================================================

class ActivationFunctionScene(InteractiveScene):
    """Demonstrate how activation functions enable non-linear decision boundaries"""

    def construct(self):
        # Title
        title = Text("Adding Non-linearity: Activation Functions", font_size=56)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()

        # Show the problem: XOR pattern
        plane = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            background_line_style={
                "stroke_color": GREY_C,
                "stroke_width": 1,
            }
        )
        plane.scale(0.6).shift(LEFT * 4 + UP * 0.5)

        class1, class2 = generate_xor_data(n_points=40)
        data_points = DataPoints(class1 + class2,
                                [BLUE] * len(class1) + [RED] * len(class2))
        data_points.move_to(plane.get_center())

        problem_label = Text("Non-linearly Separable Data (XOR)", font_size=32)
        problem_label.next_to(plane, DOWN, buff=0.3)

        self.play(
            ShowCreation(plane),
            FadeIn(data_points),
            Write(problem_label)
        )
        self.wait()

        # Show activation function graphs
        ax_relu = Axes(
            (-2, 2, 1),
            (-0.5, 2, 1),
            width=3,
            height=2
        )
        ax_relu.to_edge(RIGHT).shift(UP * 1.5)

        relu_graph = ax_relu.get_graph(
            lambda x: max(0, x),
            color=GREEN
        )
        relu_label = Text("ReLU: max(0, x)", font_size=28)
        relu_label.next_to(ax_relu, UP, buff=0.2)

        ax_sigmoid = Axes(
            (-2, 2, 1),
            (0, 1, 0.5),
            width=3,
            height=2
        )
        ax_sigmoid.next_to(ax_relu, DOWN, buff=0.8)

        sigmoid_graph = ax_sigmoid.get_graph(
            lambda x: 1 / (1 + np.exp(-x)),
            color=BLUE
        )
        sigmoid_label = Text("Sigmoid: 1/(1+e^-x)", font_size=28)
        sigmoid_label.next_to(ax_sigmoid, UP, buff=0.2)

        self.play(
            ShowCreation(ax_relu),
            ShowCreation(relu_graph),
            Write(relu_label)
        )
        self.wait()

        self.play(
            ShowCreation(ax_sigmoid),
            ShowCreation(sigmoid_graph),
            Write(sigmoid_label)
        )
        self.wait()

        # Show the key insight
        insight = Text(
            "Activation functions allow networks\nto learn curved decision boundaries",
            font_size=32,
            color=YELLOW
        )
        insight.to_edge(DOWN)
        self.play(Write(insight))
        self.wait(2)

        # Visualize how ReLU "folds" the space
        fold_text = Text("ReLU creates 'folds' in the space", font_size=28)
        fold_text.next_to(plane, UP, buff=0.2)
        self.play(Write(fold_text))
        self.wait()

        # Highlight regions
        negative_region = Rectangle(
            width=2.4, height=2.4,
            fill_color=RED, fill_opacity=0.2,
            stroke_width=0
        )
        negative_region.move_to(plane.c2p(-1.5, -1.5))

        positive_region = Rectangle(
            width=2.4, height=2.4,
            fill_color=GREEN, fill_opacity=0.2,
            stroke_width=0
        )
        positive_region.move_to(plane.c2p(1.5, 1.5))

        region_label = Text("Negative → 0", font_size=24, color=RED)
        region_label.next_to(negative_region, LEFT, buff=0.1)

        self.play(FadeIn(negative_region), Write(region_label))
        self.wait()
        self.play(FadeIn(positive_region))
        self.wait(2)

        # Clean up
        self.play(*[FadeOut(mob) for mob in self.mobjects])


# ============================================================================
# Scene 3: Single Hidden Layer MLP
# ============================================================================

class SingleLayerMLPScene(InteractiveScene):
    """Complete 2-layer network with decision boundary visualization"""

    def construct(self):
        # Title
        title = Text("Single Hidden Layer MLP", font_size=60)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()

        # Network architecture
        network = NeuralNetwork(
            layer_sizes=[2, 6, 2],
            neuron_radius=0.12,
            h_buff_ratio=5.0
        )
        network.scale(0.8).shift(UP * 0.5)

        # Layer labels
        input_label = Text("Input\n(2)", font_size=28)
        input_label.next_to(network.layers[0], DOWN, buff=0.3)

        hidden_label = Text("Hidden\n(6)", font_size=28)
        hidden_label.next_to(network.layers[1], DOWN, buff=0.3)

        output_label = Text("Output\n(2)", font_size=28)
        output_label.next_to(network.layers[2], DOWN, buff=0.3)

        labels = VGroup(input_label, hidden_label, output_label)

        self.play(
            FadeIn(network),
            Write(labels)
        )
        self.wait()

        # Show equation
        equation = Tex(
            r"y = \text{softmax}(W_2 \cdot \text{ReLU}(W_1 \cdot x + b_1) + b_2)",
            font_size=36
        )
        equation.to_edge(DOWN, buff=0.5)
        self.play(Write(equation))
        self.wait()

        # Animate forward pass
        # Create a "data particle" flowing through the network
        particle = Dot(color=YELLOW, radius=0.15)
        particle.move_to(network.layers[0][0].get_center())

        forward_label = Text("Forward Pass", font_size=32, color=YELLOW)
        forward_label.next_to(network, UP, buff=0.3)
        self.play(Write(forward_label))

        self.play(FadeIn(particle))

        # Flow through hidden layer
        for neuron in network.layers[1]:
            self.play(
                particle.animate.move_to(neuron.get_center()),
                run_time=0.2
            )

        # Flow to output
        self.play(
            particle.animate.move_to(network.layers[2][0].get_center()),
            run_time=0.3
        )
        self.wait()

        self.play(FadeOut(particle), FadeOut(forward_label))

        # Show key insight
        insight = Text(
            "Each hidden neuron learns a linear boundary.\n"
            "Combined, they create complex decision regions.",
            font_size=30,
            color=YELLOW
        )
        insight.to_edge(DOWN, buff=1.5)
        self.play(
            FadeOut(equation),
            Write(insight)
        )
        self.wait(3)

        # Clean up
        self.play(*[FadeOut(mob) for mob in self.mobjects])


# ============================================================================
# Scene 4: Deep MLP Evolution
# ============================================================================

class DeepMLPEvolutionScene(InteractiveScene):
    """Show how predictions/boundaries change with increasing depth"""

    def construct(self):
        # Title
        title = Text("Deep MLP: Evolution with Depth", font_size=60)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()

        # Generate spiral dataset
        class1, class2 = generate_spiral_data(n_points=80, noise=0.15)

        # Create multiple network configurations
        networks = []
        layer_configs = [
            ([2, 2], "1 Layer"),
            ([2, 4, 2], "2 Layers"),
            ([2, 6, 4, 2], "3 Layers"),
            ([2, 8, 6, 4, 2], "4 Layers"),
        ]

        # Position networks in a 2x2 grid
        positions = [
            UP * 2 + LEFT * 3.5,
            UP * 2 + RIGHT * 3.5,
            DOWN * 1.5 + LEFT * 3.5,
            DOWN * 1.5 + RIGHT * 3.5,
        ]

        for (layer_sizes, label_text), pos in zip(layer_configs, positions):
            # Create network
            net = NeuralNetwork(
                layer_sizes=layer_sizes,
                neuron_radius=0.08,
                h_buff_ratio=3.0
            )
            net.scale(0.5)
            net.move_to(pos)

            # Add label
            label = Text(label_text, font_size=28)
            label.next_to(net, UP, buff=0.2)

            # Add data visualization (simplified)
            plane = NumberPlane(
                x_range=[-3, 3, 1],
                y_range=[-3, 3, 1],
                background_line_style={
                    "stroke_color": GREY_D,
                    "stroke_width": 0.5,
                }
            )
            plane.scale(0.3)
            plane.next_to(net, DOWN, buff=0.2)

            # Add simplified data points
            points_visual = DataPoints(
                [[p[0]*0.2, p[1]*0.2, 0] for p in class1[:10]] +
                [[p[0]*0.2, p[1]*0.2, 0] for p in class2[:10]],
                [BLUE]*10 + [RED]*10,
                radius=0.04
            )
            points_visual.move_to(plane.get_center())

            networks.append(VGroup(net, label, plane, points_visual))

        # Animate appearance
        for net_group in networks:
            self.play(FadeIn(net_group), run_time=0.7)
        self.wait()

        # Add complexity annotation
        complexity_text = Text(
            "Deeper networks → More complex boundaries",
            font_size=36,
            color=YELLOW
        )
        complexity_text.to_edge(DOWN)
        self.play(Write(complexity_text))
        self.wait()

        # Highlight progression
        arrow = Arrow(
            networks[0].get_right(),
            networks[1].get_left(),
            color=YELLOW,
            buff=0.2
        )
        self.play(ShowCreation(arrow))
        self.wait(0.5)

        arrow2 = Arrow(
            networks[1].get_bottom(),
            networks[3].get_top(),
            color=YELLOW,
            buff=0.2
        )
        self.play(Transform(arrow, arrow2))
        self.wait()

        # Warning about overfitting
        warning = Text(
            "But beware: Too deep can lead to overfitting!",
            font_size=32,
            color=RED
        )
        warning.next_to(complexity_text, UP, buff=0.3)
        self.play(Write(warning))
        self.wait(3)

        # Clean up
        self.play(*[FadeOut(mob) for mob in self.mobjects])


# ============================================================================
# Scene 5: Universal Approximation
# ============================================================================

class UniversalApproximationScene(InteractiveScene):
    """Illustrate universal approximation theorem"""

    def construct(self):
        # Title
        title = Text("Universal Approximation Theorem", font_size=56)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()

        # Subtitle
        subtitle = Text(
            "MLPs can approximate any continuous function",
            font_size=36,
            color=YELLOW
        )
        subtitle.next_to(title, DOWN, buff=0.3)
        self.play(Write(subtitle))
        self.wait()

        # Create axes for function approximation
        axes = Axes(
            (-3, 3, 1),
            (-2, 2, 1),
            width=10,
            height=4
        )
        axes.shift(DOWN * 0.5)

        # Target function
        target_func = axes.get_graph(
            lambda x: np.sin(x) + 0.5 * np.sin(3 * x),
            color=GREEN
        )
        target_label = Text("Target Function", font_size=28, color=GREEN)
        target_label.next_to(axes, UP, buff=0.2).shift(LEFT * 3)

        self.play(
            ShowCreation(axes),
            ShowCreation(target_func),
            Write(target_label)
        )
        self.wait()

        # Approximations with increasing neurons
        neuron_counts = [2, 5, 10, 20]
        colors = [RED, ORANGE, BLUE, PURPLE]

        approximations = []
        for n_neurons, color in zip(neuron_counts, colors):
            # Simplified approximation (in reality would need actual MLP)
            # Here we just show progressively better fits
            noise_scale = 1.0 / (n_neurons * 0.5)
            approx_func = axes.get_graph(
                lambda x, ns=noise_scale: (
                    np.sin(x) + 0.5 * np.sin(3 * x) +
                    np.sin(5*x) * ns * 0.3
                ),
                color=color
            )
            approx_func.set_stroke(width=2, opacity=0.7)
            approximations.append((approx_func, n_neurons, color))

        # Show progression
        current_approx = None
        for approx_func, n_neurons, color in approximations:
            neuron_label = Text(
                f"MLP with {n_neurons} hidden neurons",
                font_size=32,
                color=color
            )
            neuron_label.next_to(axes, UP, buff=0.2).shift(RIGHT * 2.5)

            if current_approx is None:
                self.play(
                    ShowCreation(approx_func),
                    Write(neuron_label)
                )
                current_approx = approx_func
                current_label = neuron_label
            else:
                self.play(
                    Transform(current_approx, approx_func),
                    Transform(current_label, neuron_label)
                )
            self.wait(1.5)

        # Final message
        final_text = Text(
            "With enough neurons, MLPs can fit arbitrarily complex functions",
            font_size=32,
            color=YELLOW
        )
        final_text.to_edge(DOWN)
        self.play(Write(final_text))
        self.wait(3)

        # Clean up
        self.play(*[FadeOut(mob) for mob in self.mobjects])


# ============================================================================
# Scene 6: MLP Components Summary
# ============================================================================

class MLPComponentsScene(InteractiveScene):
    """Educational summary of MLP components"""

    def construct(self):
        # Title
        title = Text("MLP Building Blocks", font_size=60)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()

        # Main equation
        equation = Tex(
            r"y = \sigma(W \cdot x + b)",
            font_size=72
        )
        equation.shift(UP * 1.5)
        self.play(Write(equation))
        self.wait()

        # Break down components
        components = [
            ("x", "Input vector", LEFT * 5 + UP * 0.5),
            ("W", "Weight matrix", LEFT * 5 + DOWN * 1),
            ("b", "Bias vector", RIGHT * 0.5 + UP * 0.5),
            ("sigma", "Activation function", RIGHT * 0.5 + DOWN * 1),
            ("y", "Output vector", LEFT * 5 + DOWN * 2.5),
        ]

        component_texts = VGroup()
        for symbol, description, pos in components:
            symbol_text = Tex(f"\\textbf{{{symbol}}}:", font_size=48)
            desc_text = Text(description, font_size=32)
            component_group = VGroup(symbol_text, desc_text)
            component_group.arrange(RIGHT, buff=0.3)
            component_group.move_to(pos)
            component_texts.add(component_group)

        for comp in component_texts:
            self.play(Write(comp), run_time=0.7)
        self.wait()

        # Show layer stacking
        self.play(
            *[FadeOut(mob) for mob in [equation, component_texts]]
        )

        stack_title = Text("Stacking Layers", font_size=48)
        stack_title.shift(UP * 2.5)
        self.play(Write(stack_title))

        # Layer stack visualization
        layers = [
            "Input: x",
            "Linear: W₁x + b₁",
            "ReLU: max(0, z₁)",
            "Linear: W₂h₁ + b₂",
            "ReLU: max(0, z₂)",
            "Linear: W₃h₂ + b₃",
            "Output: y"
        ]

        layer_mobs = VGroup()
        for i, layer_text in enumerate(layers):
            layer = Text(layer_text, font_size=28)
            if i == 0:
                color = BLUE
            elif i == len(layers) - 1:
                color = GREEN
            else:
                color = WHITE
            layer.set_color(color)
            layer_mobs.add(layer)

        layer_mobs.arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        layer_mobs.shift(DOWN * 0.5)

        # Animate layer by layer
        arrows = VGroup()
        for i, layer in enumerate(layer_mobs):
            self.play(Write(layer), run_time=0.5)
            if i < len(layer_mobs) - 1:
                arrow = Arrow(
                    layer.get_bottom(),
                    layer_mobs[i+1].get_top(),
                    buff=0.1,
                    stroke_width=2,
                    max_tip_length_to_length_ratio=0.15
                )
                arrows.add(arrow)
                self.play(ShowCreation(arrow), run_time=0.3)

        self.wait(2)

        # Final insight
        final_insight = Text(
            "Composition of simple operations\ncreates powerful function approximators",
            font_size=32,
            color=YELLOW
        )
        final_insight.to_edge(DOWN)
        self.play(Write(final_insight))
        self.wait(3)

        # Clean up
        self.play(*[FadeOut(mob) for mob in self.mobjects])


# ============================================================================
# Demo Scene for Quick Testing
# ============================================================================

class MLPDemo(InteractiveScene):
    """Quick demo combining key concepts"""

    def construct(self):
        title = Text("MLP: From Linear to Deep", font_size=60)
        self.play(Write(title))
        self.wait()

        # Simple network
        network = NeuralNetwork(
            layer_sizes=[2, 4, 4, 2],
            neuron_radius=0.15,
            h_buff_ratio=4.0
        )
        network.shift(DOWN * 0.5)

        self.play(FadeOut(title))
        self.play(FadeIn(network))
        self.wait()

        # Show equation
        eq = Tex(
            r"y = W_3 \sigma(W_2 \sigma(W_1 x + b_1) + b_2) + b_3",
            font_size=40
        )
        eq.to_edge(UP)
        self.play(Write(eq))
        self.wait(2)

        # Highlight message
        message = Text(
            "Deep Learning = Stacking simple transformations\nto learn complex patterns",
            font_size=36,
            color=YELLOW
        )
        message.to_edge(DOWN)
        self.play(Write(message))
        self.wait(3)

        self.play(*[FadeOut(mob) for mob in self.mobjects])
