"""Shared ManimCE helpers for seminar scenes.

Re-exports the small set of primitives used across seminars.
"""
from .neural import Neuron, LabeledBox, TensorColumn, arrow_between

__all__ = ["Neuron", "LabeledBox", "TensorColumn", "arrow_between"]
