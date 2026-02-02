"""
Autograd Visualization for Deep Learning Course
Educational animations demonstrating automatic differentiation concepts:
- Chain Rule
- Computation Graph
- Parameters vs Activations
- Gradient Propagation
- Backward Pass
"""

import sys
import os

sys.path.append('/Users/d.tarasov/workspace/hse/fintech-dl-hse/videos')

from manim_imports_ext import *
from _2024.transformers.helpers import NeuralNetwork, value_to_color

import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import make_moons

# Import shared utilities
sys.path.append(os.path.dirname(__file__))
from src.autograd import Value
from src.data import get_moons_data
from src.models import MLP1Hidden, get_or_train_width_model


# ============================================================================
# Computation Graph Visualization Components
# ============================================================================

class ComputationNode(VGroup):
    """A node in the computational graph visualization."""

    def __init__(
        self,
        label: str,
        value: float = None,
        node_type: str = "value",  # "value", "operation", "parameter", "loss"
        radius: float = 0.4,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.label_text = label
        self.value = value
        self.node_type = node_type
        self.grad = None

        # Create node shape based on type
        if node_type == "operation":
            shape = Square(side_length=radius * 1.6)
            shape.set_stroke(GREEN, width=3)
            shape.set_fill(GREEN_E, opacity=0.3)
        elif node_type == "parameter":
            shape = Circle(radius=radius)
            shape.set_stroke(BLUE, width=3)
            shape.set_fill(BLUE_E, opacity=0.3)
        elif node_type == "loss":
            shape = Circle(radius=radius)
            shape.set_stroke(RED, width=3)
            shape.set_fill(RED_E, opacity=0.3)
        else:  # "value" or "activation"
            shape = Circle(radius=radius)
            shape.set_stroke(ORANGE, width=3)
            shape.set_fill(ORANGE, opacity=0.2)

        self.shape = shape
        self.add(shape)

        # Add label
        label_mob = Text(label, font_size=24)
        label_mob.move_to(shape.get_center())
        self.label_mob = label_mob
        self.add(label_mob)

        # Value display (below node)
        if value is not None:
            value_text = Text(f"= {value:.2f}", font_size=18, color=GREY_B)
            value_text.next_to(shape, DOWN, buff=0.1)
            self.value_mob = value_text
            self.add(value_text)

        # Gradient display (will be added during backward)
        self.grad_mob = None

    def show_gradient(self, grad_value: float, scene=None):
        """Display gradient value on the node."""
        self.grad = grad_value
        grad_text = Text(f"grad={grad_value:.2f}", font_size=16, color=RED)
        grad_text.next_to(self.shape, UP, buff=0.1)
        self.grad_mob = grad_text

        if scene:
            scene.play(FadeIn(grad_text), self.shape.animate.set_fill(RED_E, opacity=0.4))
        else:
            self.add(grad_text)

        return grad_text


# ============================================================================
# Scene 1: Chain Rule
# ============================================================================

class ChainRuleScene(InteractiveScene):
    """Scene demonstrating the chain rule - foundation of backpropagation."""

    def construct(self):
        # Title
        title = Text("The Chain Rule: Foundation of Backpropagation", font_size=44)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title))
        self.wait(0.5)

        # Chain rule formula
        chain_rule = Tex(
            r"\text{If } F = f(g(x)), \text{ then } \frac{dF}{dx} = \frac{dF}{dg} \cdot \frac{dg}{dx}",
            font_size=40
        )
        chain_rule.next_to(title, DOWN, buff=0.6)
        self.play(Write(chain_rule))
        self.wait(1.0)

        # Visual representation - chain of functions
        chain_visual = VGroup()

        # Create boxes for functions
        x_box = Square(side_length=0.8).set_stroke(BLUE, width=2)
        x_label = Text("x", font_size=28).move_to(x_box)
        x_group = VGroup(x_box, x_label)

        g_box = RoundedRectangle(width=1.2, height=0.8, corner_radius=0.1)
        g_box.set_stroke(GREEN, width=2).set_fill(GREEN_E, opacity=0.3)
        g_label = Text("g", font_size=28).move_to(g_box)
        g_group = VGroup(g_box, g_label)

        f_box = RoundedRectangle(width=1.2, height=0.8, corner_radius=0.1)
        f_box.set_stroke(ORANGE, width=2).set_fill(ORANGE, opacity=0.2)
        f_label = Text("f", font_size=28).move_to(f_box)
        f_group = VGroup(f_box, f_label)

        F_box = Square(side_length=0.8).set_stroke(RED, width=2)
        F_label = Text("F", font_size=28).move_to(F_box)
        F_group = VGroup(F_box, F_label)

        # Arrange horizontally
        chain_visual = VGroup(x_group, g_group, f_group, F_group)
        chain_visual.arrange(RIGHT, buff=1.0)
        chain_visual.move_to(ORIGIN)

        # Add arrows
        arrow1 = Arrow(x_group.get_right(), g_group.get_left(), buff=0.1, stroke_width=3)
        arrow2 = Arrow(g_group.get_right(), f_group.get_left(), buff=0.1, stroke_width=3)
        arrow3 = Arrow(f_group.get_right(), F_group.get_left(), buff=0.1, stroke_width=3)
        arrows = VGroup(arrow1, arrow2, arrow3)

        self.play(
            LaggedStartMap(FadeIn, [x_group, g_group, f_group, F_group], lag_ratio=0.3),
            LaggedStartMap(GrowArrow, [arrow1, arrow2, arrow3], lag_ratio=0.3),
        )
        self.wait(0.5)

        # Forward labels
        forward_label = Text("Forward Pass", font_size=24, color=BLUE)
        forward_label.next_to(chain_visual, DOWN, buff=0.4)
        forward_arrow = Arrow(LEFT * 2, RIGHT * 2, color=BLUE, stroke_width=2)
        forward_arrow.next_to(forward_label, DOWN, buff=0.15)

        self.play(FadeIn(forward_label), GrowArrow(forward_arrow))
        self.wait(0.5)

        # Now show backward pass
        backward_label = Text("Backward Pass (Gradients)", font_size=24, color=RED)
        backward_label.next_to(forward_arrow, DOWN, buff=0.4)
        backward_arrow = Arrow(RIGHT * 2, LEFT * 2, color=RED, stroke_width=2)
        backward_arrow.next_to(backward_label, DOWN, buff=0.15)

        self.play(FadeIn(backward_label), GrowArrow(backward_arrow))
        self.wait(0.5)

        # Show derivative labels on backward arrows
        df_dg = Tex(r"\frac{dF}{dg}", font_size=24, color=RED)
        df_dg.next_to(arrow3, UP, buff=0.1)

        dg_dx = Tex(r"\frac{dg}{dx}", font_size=24, color=RED)
        dg_dx.next_to(arrow1, UP, buff=0.1)

        self.play(FadeIn(df_dg), FadeIn(dg_dx))
        self.wait(0.5)

        # Highlight the multiplication
        multiply_eq = Tex(
            r"\frac{dF}{dx} = \frac{dF}{dg} \times \frac{dg}{dx}",
            font_size=36,
            color=YELLOW
        )
        multiply_eq.to_edge(DOWN, buff=0.5)

        self.play(Write(multiply_eq))
        self.wait(1.0)

        # Concrete example
        self.play(
            FadeOut(chain_visual),
            FadeOut(arrows),
            FadeOut(forward_label),
            FadeOut(forward_arrow),
            FadeOut(backward_label),
            FadeOut(backward_arrow),
            FadeOut(df_dg),
            FadeOut(dg_dx),
            FadeOut(multiply_eq),
        )

        # Example: F = (x + 2)^2
        example_title = Text("Example: F = (x + 2)^2", font_size=36)
        example_title.next_to(chain_rule, DOWN, buff=0.6)
        self.play(Write(example_title))

        # Let g = x + 2, F = g^2
        decomposition = Tex(
            r"\text{Let } g = x + 2, \quad F = g^2",
            font_size=32
        )
        decomposition.next_to(example_title, DOWN, buff=0.4)
        self.play(Write(decomposition))

        # Derivatives
        derivatives = Tex(
            r"\frac{dg}{dx} = 1, \quad \frac{dF}{dg} = 2g",
            font_size=32
        )
        derivatives.next_to(decomposition, DOWN, buff=0.3)
        self.play(Write(derivatives))

        # Chain rule application
        chain_app = Tex(
            r"\frac{dF}{dx} = \frac{dF}{dg} \cdot \frac{dg}{dx} = 2g \cdot 1 = 2(x+2)",
            font_size=32
        )
        chain_app.next_to(derivatives, DOWN, buff=0.3)
        self.play(Write(chain_app))
        self.wait(1.0)

        # Verification with x = 3
        verify = Tex(
            r"\text{Verify: At } x=3: \quad F = 25, \quad \frac{dF}{dx} = 2(5) = 10",
            font_size=28,
            color=GREEN
        )
        verify.next_to(chain_app, DOWN, buff=0.4)
        self.play(Write(verify))
        self.wait(1.5)

        # Key insight
        insight = Text(
            "Chain rule lets us compute derivatives of complex functions\n"
            "by breaking them into simple parts!",
            font_size=28
        )
        insight.to_edge(DOWN, buff=0.4)
        insight_bg = BackgroundRectangle(insight, color=BLACK, fill_opacity=0.8, buff=0.2)
        self.play(FadeIn(insight_bg), Write(insight))
        self.wait(2.0)


# ============================================================================
# Scene 2: Computation Graph
# ============================================================================

class ComputationGraphScene(InteractiveScene):
    """Scene showing how computation graph is built during forward pass."""

    def construct(self):
        # Title
        title = Text("Building the Computational Graph", font_size=44)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title))
        self.wait(0.5)

        # Expression we'll compute
        expression = Tex(r"L = (x \cdot w + b)^2", font_size=36)
        expression.next_to(title, DOWN, buff=0.4)
        self.play(Write(expression))
        self.wait(0.5)

        # Create nodes
        x_node = ComputationNode("x", value=2.0, node_type="value")
        w_node = ComputationNode("w", value=3.0, node_type="parameter")
        b_node = ComputationNode("b", value=1.0, node_type="parameter")

        mult_node = ComputationNode("*", node_type="operation", radius=0.35)
        add_node = ComputationNode("+", node_type="operation", radius=0.35)
        square_node = ComputationNode("^2", node_type="operation", radius=0.35)

        h1_node = ComputationNode("h1", value=6.0, node_type="value")
        h2_node = ComputationNode("h2", value=7.0, node_type="value")
        L_node = ComputationNode("L", value=49.0, node_type="loss")

        # Position nodes
        x_node.move_to(LEFT * 5 + UP * 1)
        w_node.move_to(LEFT * 5 + DOWN * 1)
        b_node.move_to(LEFT * 1.5 + DOWN * 2)

        mult_node.move_to(LEFT * 3 + ORIGIN)
        h1_node.move_to(LEFT * 1 + ORIGIN)

        add_node.move_to(RIGHT * 1 + DOWN * 0.5)
        h2_node.move_to(RIGHT * 3 + DOWN * 0.5)

        square_node.move_to(RIGHT * 5 + DOWN * 0.5)
        L_node.move_to(RIGHT * 5 + UP * 1.5)

        # Build graph step by step
        self.play(FadeIn(x_node), FadeIn(w_node))
        self.wait(0.3)

        # Step 1: x * w = h1
        step1_label = Tex(r"h_1 = x \cdot w = 2 \cdot 3 = 6", font_size=28)
        step1_label.to_edge(DOWN, buff=0.8)

        arrow_x_mult = Arrow(x_node.get_right(), mult_node.get_left() + UP * 0.15, buff=0.1)
        arrow_w_mult = Arrow(w_node.get_right(), mult_node.get_left() + DOWN * 0.15, buff=0.1)
        arrow_mult_h1 = Arrow(mult_node.get_right(), h1_node.get_left(), buff=0.1)

        self.play(
            GrowArrow(arrow_x_mult),
            GrowArrow(arrow_w_mult),
            FadeIn(mult_node),
            Write(step1_label),
        )
        self.play(GrowArrow(arrow_mult_h1), FadeIn(h1_node))
        self.wait(0.5)

        # Step 2: h1 + b = h2
        self.play(FadeOut(step1_label))
        step2_label = Tex(r"h_2 = h_1 + b = 6 + 1 = 7", font_size=28)
        step2_label.to_edge(DOWN, buff=0.8)

        self.play(FadeIn(b_node))

        arrow_h1_add = Arrow(h1_node.get_right(), add_node.get_left() + UP * 0.15, buff=0.1)
        arrow_b_add = Arrow(b_node.get_right(), add_node.get_left() + DOWN * 0.15, buff=0.1)
        arrow_add_h2 = Arrow(add_node.get_right(), h2_node.get_left(), buff=0.1)

        self.play(
            GrowArrow(arrow_h1_add),
            GrowArrow(arrow_b_add),
            FadeIn(add_node),
            Write(step2_label),
        )
        self.play(GrowArrow(arrow_add_h2), FadeIn(h2_node))
        self.wait(0.5)

        # Step 3: h2^2 = L
        self.play(FadeOut(step2_label))
        step3_label = Tex(r"L = h_2^2 = 7^2 = 49", font_size=28)
        step3_label.to_edge(DOWN, buff=0.8)

        arrow_h2_sq = Arrow(h2_node.get_right(), square_node.get_left(), buff=0.1)
        arrow_sq_L = Arrow(square_node.get_top(), L_node.get_bottom(), buff=0.1)

        self.play(
            GrowArrow(arrow_h2_sq),
            FadeIn(square_node),
            Write(step3_label),
        )
        self.play(GrowArrow(arrow_sq_L), FadeIn(L_node))
        self.wait(0.8)

        self.play(FadeOut(step3_label))

        # Highlight forward pass
        forward_text = Text("Forward Pass: Values computed left to right", font_size=28, color=BLUE)
        forward_text.to_edge(DOWN, buff=0.5)
        self.play(Write(forward_text))

        # Flash nodes in order
        nodes_order = [x_node, w_node, mult_node, h1_node, b_node, add_node, h2_node, square_node, L_node]
        for node in nodes_order:
            self.play(
                node.shape.animate.set_stroke(YELLOW, width=5),
                run_time=0.2
            )
            self.play(
                node.shape.animate.set_stroke(node.shape.get_stroke_color(), width=3),
                run_time=0.15
            )

        self.wait(0.5)
        self.play(FadeOut(forward_text))

        # Key insight
        insight = Text(
            "Each operation saves its inputs for backward pass!",
            font_size=28
        )
        insight.to_edge(DOWN, buff=0.5)
        insight_bg = BackgroundRectangle(insight, color=BLACK, fill_opacity=0.8, buff=0.15)
        self.play(FadeIn(insight_bg), Write(insight))
        self.wait(2.0)


# ============================================================================
# Scene 3: Parameters vs Activations
# ============================================================================

class ParametersVsActivationsScene(InteractiveScene):
    """Scene clarifying the distinction between parameters and activations."""

    def construct(self):
        # Title
        title = Text("Parameters vs Activations", font_size=48)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title))
        self.wait(0.5)

        # Create two panels
        left_panel = Rectangle(width=5.5, height=5, color=BLUE)
        left_panel.to_edge(LEFT, buff=0.5).shift(DOWN * 0.3)
        left_title = Text("Parameters", font_size=32, color=BLUE)
        left_title.next_to(left_panel, UP, buff=0.2)

        right_panel = Rectangle(width=5.5, height=5, color=ORANGE)
        right_panel.to_edge(RIGHT, buff=0.5).shift(DOWN * 0.3)
        right_title = Text("Activations", font_size=32, color=ORANGE)
        right_title.next_to(right_panel, UP, buff=0.2)

        self.play(
            ShowCreation(left_panel),
            ShowCreation(right_panel),
            Write(left_title),
            Write(right_title),
        )
        self.wait(0.3)

        # Parameters content
        param_items = VGroup(
            Text("W, b (weights, biases)", font_size=24),
            Text("Learned during training", font_size=22, color=GREY_B),
            Text("Updated via gradient descent", font_size=22, color=GREY_B),
            Text("Stored permanently", font_size=22, color=GREY_B),
        )
        param_items.arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        param_items.move_to(left_panel.get_center())

        # Activations content
        activ_items = VGroup(
            Text("h1, h2, a1, a2 (layer outputs)", font_size=24),
            Text("Computed during forward pass", font_size=22, color=GREY_B),
            Text("Saved for backward pass", font_size=22, color=GREY_B),
            Text("Discarded after backward", font_size=22, color=GREY_B),
        )
        activ_items.arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        activ_items.move_to(right_panel.get_center())

        self.play(
            LaggedStartMap(FadeIn, param_items, lag_ratio=0.2),
            LaggedStartMap(FadeIn, activ_items, lag_ratio=0.2),
        )
        self.wait(1.0)

        # Clear and show formula explanation
        self.play(
            FadeOut(param_items),
            FadeOut(activ_items),
        )

        # Linear layer formula
        formula_title = Text("Linear Layer: h = Wx + b", font_size=28)
        formula_title.move_to(UP * 0.5)

        grad_w = Tex(
            r"\frac{\partial L}{\partial W} = \frac{\partial L}{\partial h} \cdot x^T",
            font_size=32
        )
        grad_w.next_to(formula_title, DOWN, buff=0.5)

        # Highlight the x^T term
        explanation = Text(
            "x is the INPUT to this layer - we MUST save it!",
            font_size=26,
            color=YELLOW
        )
        explanation.next_to(grad_w, DOWN, buff=0.4)

        self.play(Write(formula_title))
        self.play(Write(grad_w))
        self.play(Write(explanation))
        self.wait(1.5)

        # Show memory diagram
        self.play(
            FadeOut(formula_title),
            FadeOut(grad_w),
            FadeOut(explanation),
        )

        memory_title = Text("Memory During Training", font_size=28)
        memory_title.move_to(UP * 1.5)

        # Forward pass memory
        fwd_mem = VGroup(
            Rectangle(width=2, height=0.6, color=BLUE).set_fill(BLUE_E, opacity=0.5),
            Text("W1", font_size=20),
        )
        fwd_mem[1].move_to(fwd_mem[0])

        fwd_mem2 = VGroup(
            Rectangle(width=2, height=0.6, color=BLUE).set_fill(BLUE_E, opacity=0.5),
            Text("W2", font_size=20),
        )
        fwd_mem2[1].move_to(fwd_mem2[0])

        activ_mem1 = VGroup(
            Rectangle(width=1.5, height=0.6, color=ORANGE).set_fill(ORANGE, opacity=0.3),
            Text("x", font_size=20),
        )
        activ_mem1[1].move_to(activ_mem1[0])

        activ_mem2 = VGroup(
            Rectangle(width=1.5, height=0.6, color=ORANGE).set_fill(ORANGE, opacity=0.3),
            Text("h1", font_size=20),
        )
        activ_mem2[1].move_to(activ_mem2[0])

        memory_blocks = VGroup(fwd_mem, activ_mem1, fwd_mem2, activ_mem2)
        memory_blocks.arrange(RIGHT, buff=0.3)
        memory_blocks.move_to(ORIGIN)

        memory_label = Text("Stored in memory during forward pass", font_size=22, color=GREY_B)
        memory_label.next_to(memory_blocks, DOWN, buff=0.3)

        self.play(Write(memory_title))
        self.play(LaggedStartMap(FadeIn, memory_blocks, lag_ratio=0.2))
        self.play(FadeIn(memory_label))
        self.wait(1.0)

        # Key insight
        insight = Text(
            "Activations use significant memory!\n"
            "This is why batch size affects memory usage.",
            font_size=26
        )
        insight.to_edge(DOWN, buff=0.4)
        insight_bg = BackgroundRectangle(insight, color=BLACK, fill_opacity=0.8, buff=0.15)
        self.play(FadeIn(insight_bg), Write(insight))
        self.wait(2.0)


# ============================================================================
# Scene 4: Gradient Propagation
# ============================================================================

class GradientPropagationScene(InteractiveScene):
    """Scene showing gradients are computed for both parameters AND inputs."""

    def construct(self):
        # Title
        title = Text("Gradient Propagation: Parameters AND Inputs", font_size=40)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title))
        self.wait(0.5)

        # Simple 2-layer network diagram
        input_node = Circle(radius=0.35, color=BLUE)
        input_node.set_fill(BLUE_E, opacity=0.3)
        input_label = Text("x", font_size=24)
        input_group = VGroup(input_node, input_label)
        input_label.move_to(input_node)

        layer1_box = RoundedRectangle(width=2, height=1.2, corner_radius=0.1, color=GREEN)
        layer1_box.set_fill(GREEN_E, opacity=0.3)
        layer1_label = Text("Layer 1\nW1, b1", font_size=18)
        layer1_group = VGroup(layer1_box, layer1_label)
        layer1_label.move_to(layer1_box)

        h1_node = Circle(radius=0.35, color=ORANGE)
        h1_node.set_fill(ORANGE, opacity=0.2)
        h1_label = Text("h1", font_size=24)
        h1_group = VGroup(h1_node, h1_label)
        h1_label.move_to(h1_node)

        layer2_box = RoundedRectangle(width=2, height=1.2, corner_radius=0.1, color=GREEN)
        layer2_box.set_fill(GREEN_E, opacity=0.3)
        layer2_label = Text("Layer 2\nW2, b2", font_size=18)
        layer2_group = VGroup(layer2_box, layer2_label)
        layer2_label.move_to(layer2_box)

        out_node = Circle(radius=0.35, color=RED)
        out_node.set_fill(RED_E, opacity=0.3)
        out_label = Text("out", font_size=24)
        out_group = VGroup(out_node, out_label)
        out_label.move_to(out_node)

        # Arrange
        network = VGroup(input_group, layer1_group, h1_group, layer2_group, out_group)
        network.arrange(RIGHT, buff=0.8)
        network.move_to(UP * 0.5)

        # Arrows
        arrows = VGroup(
            Arrow(input_group.get_right(), layer1_group.get_left(), buff=0.1),
            Arrow(layer1_group.get_right(), h1_group.get_left(), buff=0.1),
            Arrow(h1_group.get_right(), layer2_group.get_left(), buff=0.1),
            Arrow(layer2_group.get_right(), out_group.get_left(), buff=0.1),
        )

        self.play(
            LaggedStartMap(FadeIn, network, lag_ratio=0.15),
            LaggedStartMap(GrowArrow, arrows, lag_ratio=0.15),
        )
        self.wait(0.5)

        # Show gradient computation for Layer 2
        layer2_panel = Rectangle(width=5.5, height=2.2, color=WHITE)
        layer2_panel.to_edge(DOWN, buff=0.3).shift(LEFT * 2.5)
        layer2_panel.set_stroke(WHITE, width=1)

        layer2_title = Text("Layer 2 Gradients:", font_size=22, color=YELLOW)
        layer2_title.next_to(layer2_panel, UP, buff=0.1).align_to(layer2_panel, LEFT)

        grad_w2 = Tex(
            r"\frac{\partial L}{\partial W_2} = \frac{\partial L}{\partial \text{out}} \cdot h_1^T",
            font_size=26
        )
        grad_w2_label = Text("(for optimization)", font_size=18, color=BLUE)

        grad_h1 = Tex(
            r"\frac{\partial L}{\partial h_1} = W_2^T \cdot \frac{\partial L}{\partial \text{out}}",
            font_size=26
        )
        grad_h1_label = Text("(propagate backward)", font_size=18, color=RED)

        grad_w2_group = VGroup(grad_w2, grad_w2_label)
        grad_w2_label.next_to(grad_w2, RIGHT, buff=0.2)

        grad_h1_group = VGroup(grad_h1, grad_h1_label)
        grad_h1_label.next_to(grad_h1, RIGHT, buff=0.2)

        grads_layer2 = VGroup(grad_w2_group, grad_h1_group)
        grads_layer2.arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        grads_layer2.move_to(layer2_panel.get_center())

        self.play(
            ShowCreation(layer2_panel),
            Write(layer2_title),
        )
        self.play(FadeIn(grads_layer2))

        # Highlight layer 2
        self.play(
            layer2_box.animate.set_stroke(YELLOW, width=4),
        )
        self.wait(1.0)

        # Show gradient computation for Layer 1
        layer1_panel = Rectangle(width=5.5, height=2.2, color=WHITE)
        layer1_panel.to_edge(DOWN, buff=0.3).shift(RIGHT * 2.5)
        layer1_panel.set_stroke(WHITE, width=1)

        layer1_title = Text("Layer 1 Gradients:", font_size=22, color=YELLOW)
        layer1_title.next_to(layer1_panel, UP, buff=0.1).align_to(layer1_panel, LEFT)

        grad_w1 = Tex(
            r"\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial h_1} \cdot x^T",
            font_size=26
        )
        grad_w1_label = Text("(for optimization)", font_size=18, color=BLUE)

        grad_x = Tex(
            r"\frac{\partial L}{\partial x} = W_1^T \cdot \frac{\partial L}{\partial h_1}",
            font_size=26
        )
        grad_x_label = Text("(if needed)", font_size=18, color=GREY_B)

        grad_w1_group = VGroup(grad_w1, grad_w1_label)
        grad_w1_label.next_to(grad_w1, RIGHT, buff=0.2)

        grad_x_group = VGroup(grad_x, grad_x_label)
        grad_x_label.next_to(grad_x, RIGHT, buff=0.2)

        grads_layer1 = VGroup(grad_w1_group, grad_x_group)
        grads_layer1.arrange(DOWN, buff=0.25, aligned_edge=LEFT)
        grads_layer1.move_to(layer1_panel.get_center())

        self.play(
            layer2_box.animate.set_stroke(GREEN, width=2),
            ShowCreation(layer1_panel),
            Write(layer1_title),
        )
        self.play(FadeIn(grads_layer1))

        self.play(
            layer1_box.animate.set_stroke(YELLOW, width=4),
        )
        self.wait(1.0)

        # Backward flow arrows
        backward_arrows = VGroup(
            Arrow(out_group.get_left() + UP * 0.3, layer2_group.get_right() + UP * 0.3,
                  buff=0.1, color=RED, stroke_width=4),
            Arrow(layer2_group.get_left() + UP * 0.3, h1_group.get_right() + UP * 0.3,
                  buff=0.1, color=RED, stroke_width=4),
            Arrow(h1_group.get_left() + UP * 0.3, layer1_group.get_right() + UP * 0.3,
                  buff=0.1, color=RED, stroke_width=4),
        )

        self.play(
            layer1_box.animate.set_stroke(GREEN, width=2),
            LaggedStartMap(GrowArrow, backward_arrows, lag_ratio=0.3),
        )

        backward_label = Text("Gradient Flow", font_size=20, color=RED)
        backward_label.next_to(backward_arrows, UP, buff=0.1)
        self.play(FadeIn(backward_label))

        self.wait(2.0)


# ============================================================================
# Scene 5: Backward Pass
# ============================================================================

class BackwardPassScene(InteractiveScene):
    """Scene demonstrating complete backward pass with gradient computation."""

    def construct(self):
        # Title
        title = Text("Backward Pass: Computing Gradients", font_size=44)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title))
        self.wait(0.5)

        # Create simple computation graph
        # L = (x * w + b)^2 with x=2, w=3, b=1
        # h1 = x * w = 6
        # h2 = h1 + b = 7
        # L = h2^2 = 49

        nodes = {}

        nodes['x'] = VGroup(
            Circle(radius=0.4, color=BLUE).set_fill(BLUE_E, opacity=0.3),
            Text("x=2", font_size=20)
        )
        nodes['x'][1].move_to(nodes['x'][0])

        nodes['w'] = VGroup(
            Circle(radius=0.4, color=BLUE).set_fill(BLUE_E, opacity=0.3),
            Text("w=3", font_size=20)
        )
        nodes['w'][1].move_to(nodes['w'][0])

        nodes['b'] = VGroup(
            Circle(radius=0.4, color=BLUE).set_fill(BLUE_E, opacity=0.3),
            Text("b=1", font_size=20)
        )
        nodes['b'][1].move_to(nodes['b'][0])

        nodes['h1'] = VGroup(
            Circle(radius=0.4, color=ORANGE).set_fill(ORANGE, opacity=0.2),
            Text("h1=6", font_size=20)
        )
        nodes['h1'][1].move_to(nodes['h1'][0])

        nodes['h2'] = VGroup(
            Circle(radius=0.4, color=ORANGE).set_fill(ORANGE, opacity=0.2),
            Text("h2=7", font_size=20)
        )
        nodes['h2'][1].move_to(nodes['h2'][0])

        nodes['L'] = VGroup(
            Circle(radius=0.4, color=RED).set_fill(RED_E, opacity=0.3),
            Text("L=49", font_size=20)
        )
        nodes['L'][1].move_to(nodes['L'][0])

        # Position nodes
        nodes['x'].move_to(LEFT * 5 + UP * 0.8)
        nodes['w'].move_to(LEFT * 5 + DOWN * 0.8)
        nodes['b'].move_to(LEFT * 1 + DOWN * 1.5)
        nodes['h1'].move_to(LEFT * 2.5 + ORIGIN)
        nodes['h2'].move_to(RIGHT * 1 + ORIGIN)
        nodes['L'].move_to(RIGHT * 4 + ORIGIN)

        # Arrows
        arrows = VGroup(
            Arrow(nodes['x'].get_right(), nodes['h1'].get_left() + UP * 0.1, buff=0.1),
            Arrow(nodes['w'].get_right(), nodes['h1'].get_left() + DOWN * 0.1, buff=0.1),
            Arrow(nodes['h1'].get_right(), nodes['h2'].get_left() + UP * 0.1, buff=0.1),
            Arrow(nodes['b'].get_right(), nodes['h2'].get_left() + DOWN * 0.1, buff=0.1),
            Arrow(nodes['h2'].get_right(), nodes['L'].get_left(), buff=0.1),
        )

        all_nodes = VGroup(*nodes.values())
        all_nodes.shift(UP * 0.5)
        arrows.shift(UP * 0.5)

        self.play(
            LaggedStartMap(FadeIn, list(nodes.values()), lag_ratio=0.1),
            LaggedStartMap(GrowArrow, arrows, lag_ratio=0.1),
        )
        self.wait(0.5)

        # Gradient computation steps
        step_panel = Rectangle(width=11, height=1.2, color=WHITE)
        step_panel.to_edge(DOWN, buff=0.4)
        step_panel.set_stroke(WHITE, width=1)
        self.play(ShowCreation(step_panel))

        # Step 1: Start at L
        step1 = Tex(r"\text{Step 1: } \frac{\partial L}{\partial L} = 1", font_size=28)
        step1.move_to(step_panel.get_center())
        self.play(Write(step1))

        grad_L = Text("grad=1", font_size=16, color=RED)
        grad_L.next_to(nodes['L'], UP, buff=0.1)
        self.play(
            nodes['L'][0].animate.set_stroke(RED, width=4),
            FadeIn(grad_L)
        )
        self.wait(0.5)

        # Step 2: Back through square
        self.play(FadeOut(step1))
        step2 = Tex(
            r"\text{Step 2: } \frac{\partial L}{\partial h_2} = 2 h_2 = 2 \times 7 = 14",
            font_size=28
        )
        step2.move_to(step_panel.get_center())
        self.play(Write(step2))

        grad_h2 = Text("grad=14", font_size=16, color=RED)
        grad_h2.next_to(nodes['h2'], UP, buff=0.1)
        self.play(
            nodes['L'][0].animate.set_stroke(RED, width=2),
            nodes['h2'][0].animate.set_stroke(RED, width=4),
            FadeIn(grad_h2)
        )
        self.wait(0.5)

        # Step 3: Back through add
        self.play(FadeOut(step2))
        step3 = Tex(
            r"\text{Step 3: } \frac{\partial L}{\partial h_1} = 14, \quad \frac{\partial L}{\partial b} = 14",
            font_size=28
        )
        step3.move_to(step_panel.get_center())
        self.play(Write(step3))

        grad_h1 = Text("grad=14", font_size=16, color=RED)
        grad_h1.next_to(nodes['h1'], UP, buff=0.1)
        grad_b = Text("grad=14", font_size=16, color=RED)
        grad_b.next_to(nodes['b'], UP, buff=0.1)

        self.play(
            nodes['h2'][0].animate.set_stroke(ORANGE, width=2),
            nodes['h1'][0].animate.set_stroke(RED, width=4),
            nodes['b'][0].animate.set_stroke(RED, width=4),
            FadeIn(grad_h1),
            FadeIn(grad_b)
        )
        self.wait(0.5)

        # Step 4: Back through multiply
        self.play(FadeOut(step3))
        step4 = Tex(
            r"\text{Step 4: } \frac{\partial L}{\partial x} = w \cdot 14 = 42, \quad \frac{\partial L}{\partial w} = x \cdot 14 = 28",
            font_size=28
        )
        step4.move_to(step_panel.get_center())
        self.play(Write(step4))

        grad_x = Text("grad=42", font_size=16, color=RED)
        grad_x.next_to(nodes['x'], UP, buff=0.1)
        grad_w = Text("grad=28", font_size=16, color=RED)
        grad_w.next_to(nodes['w'], UP, buff=0.1)

        self.play(
            nodes['h1'][0].animate.set_stroke(ORANGE, width=2),
            nodes['x'][0].animate.set_stroke(RED, width=4),
            nodes['w'][0].animate.set_stroke(RED, width=4),
            FadeIn(grad_x),
            FadeIn(grad_w)
        )
        self.wait(1.0)

        # Summary
        self.play(FadeOut(step4))
        summary = Text(
            "Gradients computed in reverse topological order!",
            font_size=26,
            color=YELLOW
        )
        summary.move_to(step_panel.get_center())
        self.play(Write(summary))
        self.wait(2.0)


# ============================================================================
# Scene 6: MLP Autograd in Action
# ============================================================================

class MLPAutogradScene(InteractiveScene):
    """Scene showing autograd in action on real MLP training."""

    def construct(self):
        # Title
        title = Text("Autograd in Action: Training an MLP", font_size=44)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title))
        self.wait(0.5)

        # Get data
        X, y = get_moons_data()

        # Create axes for decision boundary (left side)
        axes = Axes(
            x_range=[-2, 3, 0.5],
            y_range=[-2, 2.5, 0.5],
            width=5,
            height=4.5,
            axis_config={"include_tip": True}
        )
        axes.to_edge(LEFT, buff=0.6).shift(DOWN * 0.3)

        # Data points
        dots = VGroup()
        for point, label in zip(X, y):
            dot = Dot(axes.c2p(point[0], point[1]), radius=0.06)
            dot.set_color(BLUE if label == 0 else RED)
            dots.add(dot)

        self.play(FadeIn(axes))
        self.play(LaggedStartMap(FadeIn, dots, lag_ratio=0.02))
        self.wait(0.3)

        # Network diagram (right side)
        network = NeuralNetwork([2, 4, 1])
        network.scale(0.55)
        network.to_edge(RIGHT, buff=1.0).shift(UP * 0.8)

        net_label = Text("MLP: 2 → 4 → 1", font_size=24)
        net_label.next_to(network, UP, buff=0.2)

        self.play(
            LaggedStartMap(FadeIn, network.layers, lag_ratio=0.1),
            FadeIn(net_label)
        )
        self.play(LaggedStartMap(ShowCreation, network.lines, lag_ratio=0.02))
        self.wait(0.3)

        # Training loop visualization
        training_panel = Rectangle(width=5, height=2.5, color=WHITE)
        training_panel.to_edge(RIGHT, buff=0.8).shift(DOWN * 1.8)
        training_panel.set_stroke(WHITE, width=1)

        panel_title = Text("Training Step", font_size=22)
        panel_title.next_to(training_panel, UP, buff=0.1)

        self.play(ShowCreation(training_panel), Write(panel_title))

        # Training steps
        steps = [
            ("1. Forward Pass", "Compute prediction", BLUE),
            ("2. Compute Loss", "Compare with target", ORANGE),
            ("3. Backward Pass", "Compute gradients", RED),
            ("4. Update Weights", "Gradient descent step", GREEN),
        ]

        step_text = None
        for i, (step_name, step_desc, color) in enumerate(steps):
            if step_text:
                self.play(FadeOut(step_text))

            step_text = VGroup(
                Text(step_name, font_size=24, color=color),
                Text(step_desc, font_size=18, color=GREY_B),
            )
            step_text.arrange(DOWN, buff=0.1)
            step_text.move_to(training_panel.get_center())

            self.play(FadeIn(step_text))

            # Animate corresponding network behavior
            if i == 0:  # Forward
                for layer in network.layers:
                    self.play(
                        layer.animate.set_fill(BLUE, opacity=0.5),
                        run_time=0.3
                    )
                    self.play(
                        layer.animate.set_fill(WHITE, opacity=0.2),
                        run_time=0.15
                    )
            elif i == 2:  # Backward
                for layer in reversed(network.layers):
                    self.play(
                        layer.animate.set_fill(RED, opacity=0.5),
                        run_time=0.3
                    )
                    self.play(
                        layer.animate.set_fill(WHITE, opacity=0.2),
                        run_time=0.15
                    )

            self.wait(0.5)

        self.play(FadeOut(step_text))

        # Show trained decision boundary
        try:
            model = get_or_train_width_model(4)
            from src.data import get_decision_boundary
            from src.visualization import create_decision_boundary_mobject

            x_range_axes = [float(axes.x_range[0]), float(axes.x_range[1])]
            y_range_axes = [float(axes.y_range[0]), float(axes.y_range[1])]
            xx, yy, Z = get_decision_boundary(model, X, h=0.1, x_range=x_range_axes, y_range=y_range_axes)
            boundary = create_decision_boundary_mobject(axes, xx, yy, Z, opacity=0.3, step=2)

            result_text = Text("Trained Decision Boundary", font_size=22, color=GREEN)
            result_text.move_to(training_panel.get_center())

            self.play(
                FadeIn(boundary),
                Write(result_text),
            )
        except Exception as e:
            result_text = Text("Training Complete!", font_size=22, color=GREEN)
            result_text.move_to(training_panel.get_center())
            self.play(Write(result_text))

        self.wait(1.0)

        # Key insight
        insight = Text(
            "Autograd enables efficient training by\nautomatically computing all gradients!",
            font_size=28
        )
        insight.to_edge(DOWN, buff=0.3)
        insight_bg = BackgroundRectangle(insight, color=BLACK, fill_opacity=0.8, buff=0.15)
        self.play(FadeIn(insight_bg), Write(insight))
        self.wait(2.0)


# ============================================================================
# Scene 7: Summary
# ============================================================================

class SummaryScene(InteractiveScene):
    """Final summary of autograd concepts."""

    def construct(self):
        # Title
        title = Text("Automatic Differentiation: Key Takeaways", font_size=44)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title))
        self.wait(0.5)

        # Summary points
        points = [
            "1. Chain Rule enables computing derivatives of complex functions",
            "2. Forward pass builds the computation graph and saves activations",
            "3. Activations are saved because they're needed for gradient computation",
            "4. Each layer computes gradients for BOTH parameters AND inputs",
            "5. Backward pass processes nodes in reverse topological order",
        ]

        point_mobs = VGroup()
        for point in points:
            point_mob = Text(point, font_size=26)
            point_mobs.add(point_mob)

        point_mobs.arrange(DOWN, aligned_edge=LEFT, buff=0.35)
        point_mobs.move_to(ORIGIN).shift(UP * 0.3)

        for point_mob in point_mobs:
            self.play(Write(point_mob), run_time=1.0)
            self.wait(0.3)

        self.wait(1.0)

        # Final diagram
        final_text = Text("This is how PyTorch's autograd works!", font_size=32, color=YELLOW)
        final_text.to_edge(DOWN, buff=0.5)
        final_bg = BackgroundRectangle(final_text, color=BLACK, fill_opacity=0.8, buff=0.15)

        self.play(FadeIn(final_bg), Write(final_text))
        self.wait(2.0)
