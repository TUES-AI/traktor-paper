"""Backward-compatible alias for the DINOv2 PCVM backend."""

from VMM.pcvm_d import DINOv2VisualEncoder, PCVMDINO, PCVMjNet

PCVMJEPA = PCVMDINO

__all__ = ['DINOv2VisualEncoder', 'PCVMDINO', 'PCVMJEPA', 'PCVMjNet']
