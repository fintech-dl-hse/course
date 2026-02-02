"""Computation graph visualization utilities."""

from manim_imports_ext import *
import torch
import numpy as np


class ComputationGraphNode(VGroup):
    """A node in the computation graph representing an operation."""

    def __init__(
        self,
        operation_name,
        input_value=None,
        output_value=None,
        node_type="operation",  # "operation", "parameter", "activation"
        width=1.2,
        height=0.8,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.operation_name = operation_name
        self.input_value = input_value
        self.output_value = output_value
        self.node_type = node_type

        # Create rectangle background
        if node_type == "parameter":
            color = PURPLE_C
            fill_opacity = 0.3
        elif node_type == "activation":
            color = BLUE_C
            fill_opacity = 0.3
        else:
            color = WHITE
            fill_opacity = 0.1

        rect = Rectangle(width=width, height=height, color=color, fill_opacity=fill_opacity)
        rect.set_stroke(color, width=2)

        # Create text label
        if output_value is not None:
            if isinstance(output_value, torch.Tensor):
                if output_value.numel() == 1:
                    val_str = f"{output_value.item():.3f}"
                else:
                    val_str = f"shape: {tuple(output_value.shape)}"
            elif isinstance(output_value, (int, float)):
                val_str = f"{output_value:.3f}"
            else:
                val_str = str(output_value)
            label = Text(f"{operation_name}\n{val_str}", font_size=20)
        else:
            label = Text(operation_name, font_size=24)

        label.move_to(rect.get_center())

        self.add(rect, label)
        self.rect = rect
        self.label = label


class ComputationGraphEdge(Arrow):
    """An edge in the computation graph showing data or gradient flow."""

    def __init__(
        self,
        start_node,
        end_node,
        edge_type="forward",  # "forward" or "backward"
        gradient_value=None,
        **kwargs
    ):
        self.edge_type = edge_type
        self.gradient_value = gradient_value

        if edge_type == "backward":
            color = RED
            start_point = end_node.get_bottom()
            end_point = start_node.get_top()
        else:
            color = BLUE
            start_point = start_node.get_right()
            end_point = end_node.get_left()

        super().__init__(
            start_point,
            end_point,
            color=color,
            buff=0.1,
            stroke_width=3 if edge_type == "backward" else 2,
            **kwargs
        )

        # Add gradient value label if provided
        if gradient_value is not None and edge_type == "backward":
            if isinstance(gradient_value, torch.Tensor):
                if gradient_value.numel() == 1:
                    grad_str = f"{gradient_value.item():.3f}"
                else:
                    grad_str = f"grad"
            else:
                grad_str = f"{gradient_value:.3f}"
            label = Text(grad_str, font_size=16, color=RED)
            label.move_to(self.get_center())
            self.add(label)


class ComputationGraph(VGroup):
    """A complete computation graph visualization."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nodes = VGroup()
        self.edges = VGroup()
        self.node_dict = {}

    def add_node(self, node_id, operation_name, input_value=None, output_value=None, node_type="operation", position=None):
        """Add a node to the graph."""
        node = ComputationGraphNode(operation_name, input_value, output_value, node_type)
        if position is not None:
            node.move_to(position)
        self.node_dict[node_id] = node
        self.nodes.add(node)
        self.add(node)
        return node

    def add_edge(self, start_id, end_id, edge_type="forward", gradient_value=None):
        """Add an edge between two nodes."""
        if start_id not in self.node_dict or end_id not in self.node_dict:
            return None
        start_node = self.node_dict[start_id]
        end_node = self.node_dict[end_id]
        edge = ComputationGraphEdge(start_node, end_node, edge_type, gradient_value)
        self.edges.add(edge)
        self.add(edge)
        return edge

    def layout_nodes(self, layout="vertical", spacing=1.5):
        """Layout nodes in the graph."""
        if layout == "vertical":
            positions = []
            y_start = 2.0
            for i, node in enumerate(self.nodes):
                x = (i % 3 - 1) * 2.5
                y = y_start - (i // 3) * spacing
                positions.append([x, y, 0])
            for node, pos in zip(self.nodes, positions):
                node.move_to(pos)
        elif layout == "horizontal":
            for i, node in enumerate(self.nodes):
                node.move_to([i * spacing - len(self.nodes) * spacing / 2, 0, 0])

