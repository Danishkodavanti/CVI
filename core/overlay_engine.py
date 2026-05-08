import numpy as np
from typing import Optional
from core.asset_store import AssetStore
from filters.dog_filter import DogFilter
from filters.sunglasses_filter import SunglassesFilter
from filters.shader_filter import ShaderFilter
from filters.crown_filter import CrownFilter
from filters.mask_filter import MaskFilter


class FilterOverlayEngine:
    """
    Central router: picks the active filter and delegates apply().
    Manages filter registry and cycling.
    """

    def __init__(self, assets_dir: str = "assets"):
        self._store = AssetStore(assets_dir)
        self._filters = [
            DogFilter(self._store),
            SunglassesFilter(self._store),
            CrownFilter(self._store),
            ShaderFilter(self._store, mode="beauty"),
            ShaderFilter(self._store, mode="vintage"),
            ShaderFilter(self._store, mode="neon"),
            ShaderFilter(self._store, mode="heatmap"),
            MaskFilter(self._store, mode="phantom"),
            MaskFilter(self._store, mode="clown"),
            MaskFilter(self._store, mode="warrior"),
        ]
        self._idx = 0

    @property
    def active_filter_name(self) -> str:
        return self._filters[self._idx].name

    @property
    def filter_names(self) -> list[str]:
        return [f.name for f in self._filters]

    def next_filter(self):
        self._idx = (self._idx + 1) % len(self._filters)

    def prev_filter(self):
        self._idx = (self._idx - 1) % len(self._filters)

    def set_filter(self, idx: int):
        self._idx = idx % len(self._filters)

    def apply(self, frame: np.ndarray,
              landmarks: Optional[np.ndarray]) -> np.ndarray:
        if landmarks is None:
            return frame
        try:
            return self._filters[self._idx].apply(frame, landmarks)
        except Exception:
            return frame