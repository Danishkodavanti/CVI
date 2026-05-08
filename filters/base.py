from abc import ABC, abstractmethod
import numpy as np
from core.landmark_detector import LandmarkDetector
from core.geometry import GeometricTransformer
from core.compositor import AlphaCompositor
from core.asset_store import AssetStore


class BaseFilter(ABC):
    def __init__(self, asset_store: AssetStore):
        self.assets = asset_store
        self.geo = GeometricTransformer()
        self.compositor = AlphaCompositor()

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def apply(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Apply filter to frame given 478-point landmarks. Returns modified BGR frame."""
        ...