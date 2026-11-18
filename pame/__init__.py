"""PAME: protocol-aware mask embedding for multi-dataset tooth instance segmentation (MMDetection 3.x plugin).
Load in a config with ``custom_imports = dict(imports=['pame'], allow_failed_imports=False)``."""
from .protocol_mask2former_head import ProtocolMask2FormerHead
from .protocol_roi_head import ProtocolRoIHead
from .transforms import AddSourceId, RandomFlipFDI

__all__ = ['ProtocolMask2FormerHead', 'ProtocolRoIHead', 'AddSourceId', 'RandomFlipFDI']
__version__ = '1.0.0'
