"""
Lottery Ticket Hypothesis - Manim Animation
For Master's Deep Learning Course (Lecture 2)

Based on: Frankle & Carlin, "The Lottery Ticket Hypothesis", ICLR 2019
"""

from manim import *
import numpy as np

# Configuration
config.frame_width = 16
config.frame_height = 9

# Color scheme
STRONG_WEIGHT_COLOR = BLUE
WEAK_WEIGHT_COLOR = RED_C
NEUTRAL_COLOR = GRAY_B
PRUNED_COLOR = RED_E
HIGHLIGHT_COLOR = YELLOW
BG_COLOR = "#1a1a2e"


class NeuralNetwork(VGroup):
    """A customizable neural network visualization."""

    def __init__(
        self,
        layer_sizes=[6, 8, 8, 4],
        neuron_radius=0.15,
        layer_spacing=2.0,
        neuron_spacing=0.5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.layer_sizes = layer_sizes
        self.neuron_radius = neuron_radius
        self.layer_spacing = layer_spacing
        self.neuron_spacing = neuron_spacing

        self.neurons = []  # List of lists of circles
        self.weights = []  # List of dicts mapping (i,j) -> line
        self.weight_values = []  # Store weight magnitudes
        self.initial_weight_values = []  # Store initial weights for reset
        self.mask = []  # Pruning mask

        self._create_neurons()
        self._create_weights()

    def _create_neurons(self):
        """Create neuron circles for each layer."""
        total_width = (len(self.layer_sizes) - 1) * self.layer_spacing
        start_x = -total_width / 2

        for layer_idx, size in enumerate(self.layer_sizes):
            layer_neurons = []
            total_height = (size - 1) * self.neuron_spacing
            start_y = total_height / 2
            x = start_x + layer_idx * self.layer_spacing

            for neuron_idx in range(size):
                y = start_y - neuron_idx * self.neuron_spacing
                neuron = Circle(
                    radius=self.neuron_radius,
                    fill_color=NEUTRAL_COLOR,
                    fill_opacity=0.8,
                    stroke_color=WHITE,
                    stroke_width=1.5
                ).move_to([x, y, 0])
                layer_neurons.append(neuron)
                self.add(neuron)

            self.neurons.append(layer_neurons)

    def _create_weights(self):
        """Create weight connections between layers."""
        np.random.seed(42)  # For reproducibility

        for layer_idx in range(len(self.layer_sizes) - 1):
            layer_weights = {}
            layer_values = {}
            layer_mask = {}
            layer_init_values = {}

            for i, neuron_from in enumerate(self.neurons[layer_idx]):
                for j, neuron_to in enumerate(self.neurons[layer_idx + 1]):
                    # Random initial weight
                    weight_val = np.random.randn() * 0.5

                    # Create line
                    line = Line(
                        neuron_from.get_center(),
                        neuron_to.get_center(),
                        stroke_width=self._weight_to_width(abs(weight_val)),
                        stroke_color=NEUTRAL_COLOR,
                        stroke_opacity=0.6
                    )

                    layer_weights[(i, j)] = line
                    layer_values[(i, j)] = weight_val
                    layer_init_values[(i, j)] = weight_val
                    layer_mask[(i, j)] = 1  # All weights active initially

                    # Add line behind neurons
                    self.add(line)

            self.weights.append(layer_weights)
            self.weight_values.append(layer_values)
            self.initial_weight_values.append(dict(layer_init_values))
            self.mask.append(layer_mask)

        # Move neurons to front
        for layer in self.neurons:
            for neuron in layer:
                self.remove(neuron)
                self.add(neuron)

    def _weight_to_width(self, magnitude):
        """Convert weight magnitude to stroke width."""
        return np.clip(magnitude * 4, 0.5, 4)

    def _weight_to_color(self, magnitude, trained=False):
        """Convert weight magnitude to color."""
        if trained:
            if magnitude > 0.5:
                return STRONG_WEIGHT_COLOR
            elif magnitude < 0.2:
                return WEAK_WEIGHT_COLOR
            else:
                return NEUTRAL_COLOR
        return NEUTRAL_COLOR

    def get_weight_lines(self):
        """Get all weight lines as a flat list."""
        lines = []
        for layer_weights in self.weights:
            lines.extend(layer_weights.values())
        return lines

    def animate_training(self, scene, duration=3):
        """Animate training: weights change magnitude."""
        np.random.seed(123)
        animations = []

        for layer_idx, layer_weights in enumerate(self.weights):
            for (i, j), line in layer_weights.items():
                # Simulate training: some weights grow, some shrink
                old_val = self.weight_values[layer_idx][(i, j)]
                # Training tends to make important weights larger
                new_val = old_val + np.random.randn() * 0.8
                if np.random.rand() > 0.3:  # 70% chance to grow
                    new_val = abs(new_val) + 0.2
                else:
                    new_val = abs(new_val) * 0.3

                self.weight_values[layer_idx][(i, j)] = new_val

                new_width = self._weight_to_width(new_val)
                new_color = self._weight_to_color(new_val, trained=True)

                animations.append(
                    line.animate.set_stroke(
                        width=new_width,
                        color=new_color,
                        opacity=0.8
                    )
                )

        scene.play(*animations, run_time=duration)

    def show_magnitude_colors(self, scene, duration=1.5):
        """Color weights by magnitude (pre-pruning visualization)."""
        animations = []

        for layer_idx, layer_weights in enumerate(self.weights):
            for (i, j), line in layer_weights.items():
                magnitude = abs(self.weight_values[layer_idx][(i, j)])
                color = self._weight_to_color(magnitude, trained=True)
                animations.append(line.animate.set_stroke(color=color))

        scene.play(*animations, run_time=duration)

    def prune_weights(self, scene, keep_ratio=0.2, duration=2):
        """Prune lowest magnitude weights."""
        # Collect all weight magnitudes
        all_magnitudes = []
        for layer_idx, layer_values in enumerate(self.weight_values):
            for (i, j), val in layer_values.items():
                all_magnitudes.append((layer_idx, i, j, abs(val)))

        # Sort by magnitude
        all_magnitudes.sort(key=lambda x: x[3])

        # Determine threshold
        num_to_prune = int(len(all_magnitudes) * (1 - keep_ratio))
        weights_to_prune = all_magnitudes[:num_to_prune]

        # Create pruning animations
        prune_animations = []
        for layer_idx, i, j, _ in weights_to_prune:
            line = self.weights[layer_idx][(i, j)]
            self.mask[layer_idx][(i, j)] = 0
            prune_animations.append(
                line.animate.set_stroke(opacity=0)
            )

        scene.play(*prune_animations, run_time=duration)

        # Remove pruned weights from view
        for layer_idx, i, j, _ in weights_to_prune:
            self.weights[layer_idx][(i, j)].set_stroke(opacity=0)

    def reset_to_init(self, scene, duration=2):
        """Reset remaining weights to initial values (the key LTH insight)."""
        animations = []

        for layer_idx, layer_weights in enumerate(self.weights):
            for (i, j), line in layer_weights.items():
                if self.mask[layer_idx][(i, j)] == 1:  # Only reset non-pruned
                    init_val = self.initial_weight_values[layer_idx][(i, j)]
                    self.weight_values[layer_idx][(i, j)] = init_val

                    new_width = self._weight_to_width(abs(init_val))

                    animations.append(
                        line.animate.set_stroke(
                            width=new_width,
                            color=NEUTRAL_COLOR,
                            opacity=0.8
                        )
                    )

        scene.play(*animations, run_time=duration)

    def retrain_sparse(self, scene, duration=2):
        """Animate retraining the sparse network."""
        np.random.seed(456)
        animations = []

        for layer_idx, layer_weights in enumerate(self.weights):
            for (i, j), line in layer_weights.items():
                if self.mask[layer_idx][(i, j)] == 1:
                    # Retrain: weights become strong again
                    new_val = abs(np.random.randn() * 0.5) + 0.5
                    self.weight_values[layer_idx][(i, j)] = new_val

                    animations.append(
                        line.animate.set_stroke(
                            width=self._weight_to_width(new_val),
                            color=STRONG_WEIGHT_COLOR,
                            opacity=0.9
                        )
                    )

        scene.play(*animations, run_time=duration)


class AccuracyCurve(VGroup):
    """Animated accuracy curve."""

    def __init__(self, width=3, height=2, final_accuracy=0.95, **kwargs):
        super().__init__(**kwargs)
        self.width = width
        self.height = height
        self.final_accuracy = final_accuracy

        # Axes
        self.axes = Axes(
            x_range=[0, 10, 2],
            y_range=[0, 1, 0.2],
            x_length=width,
            y_length=height,
            axis_config={"color": WHITE, "stroke_width": 2},
            tips=False
        )
        self.add(self.axes)

        # Labels
        x_label = Text("Steps", font_size=18).next_to(self.axes.x_axis, DOWN, buff=0.2)
        y_label = Text("Accuracy", font_size=18).next_to(self.axes.y_axis, LEFT, buff=0.2).rotate(90 * DEGREES)
        self.axes.add(x_label, y_label)

        # Curve (sigmoid-like)
        self.curve = self.axes.plot(
            lambda x: final_accuracy * (1 - np.exp(-0.5 * x)),
            x_range=[0, 10],
            color=GREEN
        )
        self.add(self.curve)

    def animate_curve(self, scene, duration=2):
        """Animate the curve drawing."""
        scene.play(Create(self.curve), run_time=duration)

        # Add final accuracy label
        final_label = Text(
            f"{self.final_accuracy * 100:.0f}%",
            font_size=24,
            color=GREEN
        ).next_to(self.axes.c2p(10, self.final_accuracy), RIGHT, buff=0.1)
        scene.play(FadeIn(final_label), run_time=0.5)
        self.add(final_label)
        self.final_label = final_label


class LotteryTicketHypothesis(Scene):
    """Main animation scene."""

    def construct(self):
        self.camera.background_color = BG_COLOR

        self.scene1_title()
        self.scene2_dense_network()
        self.scene3_training()
        self.scene4_pruning()
        self.scene5_reset()
        self.scene6_retrain()
        self.scene7_control()
        self.scene8_summary()

    def scene1_title(self):
        """Title card."""
        title = Text(
            "The Lottery Ticket Hypothesis",
            font_size=56,
            color=WHITE
        )
        subtitle = Text(
            "Frankle & Carlin, ICLR 2019",
            font_size=28,
            color=GRAY_B
        ).next_to(title, DOWN, buff=0.5)

        self.play(FadeIn(title, shift=UP * 0.3), run_time=1)
        self.play(FadeIn(subtitle), run_time=0.8)
        self.wait(1.5)
        self.play(FadeOut(title), FadeOut(subtitle), run_time=0.8)
        self.wait(0.3)

    def scene2_dense_network(self):
        """Show dense network at initialization."""
        # Section title
        section_title = Text("1. Dense Network Initialization", font_size=32, color=GRAY_B)
        section_title.to_edge(UP, buff=0.4)
        self.play(FadeIn(section_title), run_time=0.5)

        # Create network
        self.network = NeuralNetwork(layer_sizes=[6, 8, 8, 4])
        self.network.shift(LEFT * 1)

        # Animate network appearing layer by layer
        for layer_idx, layer in enumerate(self.network.neurons):
            neurons_group = VGroup(*layer)
            self.play(FadeIn(neurons_group, scale=0.8), run_time=0.4)

            # Show weights to this layer
            if layer_idx > 0:
                weights = [
                    self.network.weights[layer_idx - 1][(i, j)]
                    for i in range(self.network.layer_sizes[layer_idx - 1])
                    for j in range(self.network.layer_sizes[layer_idx])
                ]
                self.play(*[FadeIn(w) for w in weights], run_time=0.3)

        # Labels
        label1 = Text("Dense Network", font_size=28).to_edge(RIGHT, buff=1).shift(UP * 1.5)
        label2 = MathTex(r"\text{Random initialization } \theta_0", font_size=32)
        label2.next_to(label1, DOWN, buff=0.4)
        label3 = MathTex(r"|\theta_0| = N \text{ parameters}", font_size=28, color=GRAY_B)
        label3.next_to(label2, DOWN, buff=0.3)

        self.play(FadeIn(label1), run_time=0.5)
        self.play(FadeIn(label2), run_time=0.5)
        self.play(FadeIn(label3), run_time=0.5)

        self.wait(1.5)

        # Store for later
        self.section_title = section_title
        self.labels = VGroup(label1, label2, label3)

        # Transition
        self.play(
            FadeOut(section_title),
            FadeOut(self.labels),
            run_time=0.5
        )

    def scene3_training(self):
        """Train the dense network."""
        # Section title
        section_title = Text("2. Train Dense Network", font_size=32, color=GRAY_B)
        section_title.to_edge(UP, buff=0.4)
        self.play(FadeIn(section_title), run_time=0.5)

        # Move network left to make room for accuracy curve
        self.play(self.network.animate.shift(LEFT * 1), run_time=0.5)

        # Create accuracy curve
        self.acc_curve = AccuracyCurve(width=3.5, height=2, final_accuracy=0.95)
        self.acc_curve.to_edge(RIGHT, buff=0.8).shift(DOWN * 0.5)

        # Show axes
        self.play(FadeIn(self.acc_curve.axes), run_time=0.5)

        # Training label
        training_label = Text("Training...", font_size=28, color=YELLOW)
        training_label.next_to(self.network, UP, buff=0.5)
        self.play(FadeIn(training_label), run_time=0.3)

        # Animate training and accuracy curve together
        self.network.animate_training(self, duration=2.5)
        self.acc_curve.animate_curve(self, duration=2)

        # Update label
        trained_label = MathTex(r"\text{Trained weights } \theta", font_size=32)
        trained_label.next_to(self.network, UP, buff=0.5)
        self.play(
            ReplacementTransform(training_label, trained_label),
            run_time=0.5
        )

        self.wait(1.5)

        # Transition
        self.trained_label = trained_label
        self.play(
            FadeOut(section_title),
            FadeOut(trained_label),
            FadeOut(self.acc_curve),
            run_time=0.5
        )

    def scene4_pruning(self):
        """Magnitude-based pruning."""
        # Section title
        section_title = Text("3. Magnitude-Based Pruning", font_size=32, color=GRAY_B)
        section_title.to_edge(UP, buff=0.4)
        self.play(FadeIn(section_title), run_time=0.5)

        # Show legend
        legend = VGroup(
            VGroup(
                Line(ORIGIN, RIGHT * 0.5, stroke_color=STRONG_WEIGHT_COLOR, stroke_width=3),
                Text("Strong (keep)", font_size=20)
            ).arrange(RIGHT, buff=0.2),
            VGroup(
                Line(ORIGIN, RIGHT * 0.5, stroke_color=WEAK_WEIGHT_COLOR, stroke_width=2),
                Text("Weak (prune)", font_size=20)
            ).arrange(RIGHT, buff=0.2)
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        legend.to_edge(RIGHT, buff=0.8).shift(UP * 2)

        self.play(FadeIn(legend), run_time=0.5)

        # Show magnitude colors
        self.network.show_magnitude_colors(self, duration=1.5)
        self.wait(1)

        # Pruning info
        prune_info = VGroup(
            Text("Prune 80% lowest weights", font_size=24),
            MathTex(r"\text{Create mask } m \in \{0,1\}^N", font_size=28),
            Text("Keep: 20% of parameters", font_size=24, color=GREEN)
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        prune_info.to_edge(RIGHT, buff=0.6).shift(DOWN * 0.5)

        self.play(FadeIn(prune_info), run_time=0.8)
        self.wait(1)

        # Perform pruning
        self.network.prune_weights(self, keep_ratio=0.2, duration=2)

        self.wait(1.5)

        # Transition
        self.play(
            FadeOut(section_title),
            FadeOut(legend),
            FadeOut(prune_info),
            run_time=0.5
        )

    def scene5_reset(self):
        """Key insight: reset to original initialization."""
        # Section title - emphasized
        section_title = Text("4. Key Insight: Reset to θ₀", font_size=36, color=YELLOW)
        section_title.to_edge(UP, buff=0.4)
        self.play(FadeIn(section_title, scale=1.1), run_time=0.6)

        # Explanation
        explanation = VGroup(
            Text("Keep the sparse structure (mask m)", font_size=24),
            Text("Reset weights to original initialization", font_size=24, color=YELLOW),
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        explanation.to_edge(RIGHT, buff=0.5).shift(UP * 1)

        self.play(FadeIn(explanation[0]), run_time=0.6)
        self.wait(0.5)
        self.play(FadeIn(explanation[1]), run_time=0.6)

        # Visual indicator: rewind
        rewind_text = Text("⟲ REWIND", font_size=36, color=YELLOW)
        rewind_text.next_to(self.network, UP, buff=0.5)
        self.play(FadeIn(rewind_text, scale=1.2), run_time=0.4)

        # Reset animation
        self.network.reset_to_init(self, duration=2)

        self.play(FadeOut(rewind_text), run_time=0.3)

        # Formula
        formula = MathTex(
            r"\text{Winning Ticket} = m \odot \theta_0",
            font_size=36,
            color=GREEN
        )
        formula.to_edge(RIGHT, buff=0.6).shift(DOWN * 1)

        box = SurroundingRectangle(formula, color=GREEN, buff=0.2)

        self.play(FadeIn(formula), Create(box), run_time=0.8)

        self.wait(2)

        # Transition
        self.play(
            FadeOut(section_title),
            FadeOut(explanation),
            FadeOut(formula),
            FadeOut(box),
            run_time=0.5
        )

    def scene6_retrain(self):
        """Retrain the winning ticket."""
        # Section title
        section_title = Text("5. Train the Winning Ticket", font_size=32, color=GRAY_B)
        section_title.to_edge(UP, buff=0.4)
        self.play(FadeIn(section_title), run_time=0.5)

        # Create new accuracy curve
        acc_curve2 = AccuracyCurve(width=3, height=1.8, final_accuracy=0.95)
        acc_curve2.to_edge(RIGHT, buff=0.6).shift(DOWN * 0.5)

        self.play(FadeIn(acc_curve2.axes), run_time=0.4)

        # Training label
        training_label = Text("Training sparse network...", font_size=24, color=YELLOW)
        training_label.next_to(self.network, UP, buff=0.5)
        self.play(FadeIn(training_label), run_time=0.3)

        # Retrain animation
        self.network.retrain_sparse(self, duration=2)
        acc_curve2.animate_curve(self, duration=1.5)

        self.play(FadeOut(training_label), run_time=0.3)

        # Comparison table
        comparison = VGroup(
            Text("Comparison:", font_size=24, color=WHITE),
            VGroup(
                Text("Dense Network:", font_size=20),
                Text("95% acc, 100% params", font_size=20, color=GRAY_B)
            ).arrange(RIGHT, buff=0.3),
            VGroup(
                Text("Winning Ticket:", font_size=20),
                Text("95% acc, 20% params", font_size=20, color=GREEN)
            ).arrange(RIGHT, buff=0.3),
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        comparison.to_edge(RIGHT, buff=0.5).shift(UP * 2)

        self.play(FadeIn(comparison), run_time=0.8)

        # Highlight
        highlight = Text("Same accuracy, 5× fewer parameters!", font_size=24, color=YELLOW)
        highlight.next_to(comparison, DOWN, buff=0.5)
        self.play(FadeIn(highlight, scale=1.1), run_time=0.5)

        self.wait(2)

        # Store for comparison in scene 7
        self.acc_curve2 = acc_curve2

        # Transition
        self.play(
            FadeOut(section_title),
            FadeOut(acc_curve2),
            FadeOut(comparison),
            FadeOut(highlight),
            run_time=0.5
        )

    def scene7_control(self):
        """Control experiment: random mask fails."""
        # Section title
        section_title = Text("6. Control: Random Sparse Mask", font_size=32, color=GRAY_B)
        section_title.to_edge(UP, buff=0.4)
        self.play(FadeIn(section_title), run_time=0.5)

        # Move current network aside
        self.play(self.network.animate.shift(LEFT * 2).scale(0.7), run_time=0.6)

        label_winner = Text("Winning Ticket", font_size=20, color=GREEN)
        label_winner.next_to(self.network, DOWN, buff=0.3)
        self.play(FadeIn(label_winner), run_time=0.3)

        # Create new network with random mask
        random_net = NeuralNetwork(layer_sizes=[6, 8, 8, 4])
        random_net.shift(RIGHT * 2).scale(0.7)

        # Show it
        self.play(FadeIn(random_net), run_time=0.5)

        label_random = Text("Random Mask", font_size=20, color=RED_C)
        label_random.next_to(random_net, DOWN, buff=0.3)
        self.play(FadeIn(label_random), run_time=0.3)

        # Apply random pruning (different seed)
        np.random.seed(999)  # Different seed = different mask
        for layer_idx, layer_weights in enumerate(random_net.weights):
            for (i, j), line in layer_weights.items():
                if np.random.rand() > 0.2:  # Prune 80% randomly
                    line.set_stroke(opacity=0)
                    random_net.mask[layer_idx][(i, j)] = 0

        # Explanation
        explanation = Text("Same sparsity (20%), but random structure", font_size=22)
        explanation.to_edge(UP, buff=1.2)
        self.play(FadeIn(explanation), run_time=0.5)

        # Train random network (poor results)
        training_label = Text("Training...", font_size=22, color=YELLOW)
        training_label.next_to(random_net, UP, buff=0.3)
        self.play(FadeIn(training_label), run_time=0.3)

        # Animate some training on random net
        animations = []
        for layer_idx, layer_weights in enumerate(random_net.weights):
            for (i, j), line in layer_weights.items():
                if random_net.mask[layer_idx][(i, j)] == 1:
                    animations.append(
                        line.animate.set_stroke(color=ORANGE, opacity=0.7)
                    )
        self.play(*animations, run_time=1.5)
        self.play(FadeOut(training_label), run_time=0.2)

        # Show accuracy comparison
        comparison = VGroup(
            VGroup(
                Text("Winning Ticket:", font_size=22),
                Text("95%", font_size=28, color=GREEN)
            ).arrange(RIGHT, buff=0.3),
            VGroup(
                Text("Random Mask:", font_size=22),
                Text("65%", font_size=28, color=RED_C)
            ).arrange(RIGHT, buff=0.3),
        ).arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        comparison.to_edge(RIGHT, buff=0.3).shift(UP * 0.5)

        self.play(FadeIn(comparison), run_time=0.8)

        # Conclusion
        conclusion = Text(
            "Structure matters, not just sparsity!",
            font_size=26,
            color=YELLOW
        )
        conclusion.to_edge(DOWN, buff=1)
        box = SurroundingRectangle(conclusion, color=YELLOW, buff=0.15)

        self.play(FadeIn(conclusion), Create(box), run_time=0.6)

        self.wait(2.5)

        # Transition
        self.play(
            FadeOut(section_title),
            FadeOut(self.network),
            FadeOut(label_winner),
            FadeOut(random_net),
            FadeOut(label_random),
            FadeOut(explanation),
            FadeOut(comparison),
            FadeOut(conclusion),
            FadeOut(box),
            run_time=0.8
        )

    def scene8_summary(self):
        """Summary and conclusion."""
        # Title
        title = Text("Summary", font_size=44, color=WHITE)
        title.to_edge(UP, buff=0.8)
        self.play(FadeIn(title), run_time=0.5)

        # Key points
        points = VGroup(
            Text("1. Dense neural networks contain sparse subnetworks", font_size=26),
            Text("2. These 'winning tickets' can match full network accuracy", font_size=26),
            Text("3. Critical: must use original initialization θ₀", font_size=26, color=YELLOW),
            Text("4. Random sparse masks do NOT work", font_size=26),
        ).arrange(DOWN, buff=0.5, aligned_edge=LEFT)
        points.next_to(title, DOWN, buff=0.8)

        for point in points:
            self.play(FadeIn(point, shift=RIGHT * 0.2), run_time=0.6)
            self.wait(0.3)

        self.wait(1)

        # Visual summary: mini before/after
        mini_dense = Text("Dense Network", font_size=20)
        arrow = Arrow(LEFT, RIGHT, color=WHITE, buff=0.1)
        mini_sparse = Text("Winning Ticket (20%)", font_size=20, color=GREEN)

        summary_visual = VGroup(mini_dense, arrow, mini_sparse).arrange(RIGHT, buff=0.3)
        summary_visual.next_to(points, DOWN, buff=0.8)

        self.play(FadeIn(summary_visual), run_time=0.6)

        # Citation
        citation = Text(
            'Frankle & Carlin, "The Lottery Ticket Hypothesis", ICLR 2019',
            font_size=22,
            color=GRAY_B
        )
        citation.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(citation), run_time=0.5)

        self.wait(3)

        # Fade out everything
        self.play(
            *[FadeOut(mob) for mob in self.mobjects],
            run_time=1
        )


# For rendering individual scenes during development
class TitleScene(Scene):
    def construct(self):
        self.camera.background_color = BG_COLOR
        LotteryTicketHypothesis.scene1_title(self)


if __name__ == "__main__":
    # This allows running with: manim -pql lottery_ticket.py LotteryTicketHypothesis
    pass