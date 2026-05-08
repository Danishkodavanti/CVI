import cv2
import numpy as np
from pathlib import Path
from typing import Optional


class AssetStore:
    """Caches loaded BGRA sprites and raw assets."""

    def __init__(self, assets_dir: str = "assets"):
        self._dir = Path(assets_dir)
        self._cache: dict[str, np.ndarray] = {}

    def get_sprite(self, name: str) -> Optional[np.ndarray]:
        if name in self._cache:
            return self._cache[name]

        path = self._dir / "sprites" / name
        if not path.exists():
            return None

        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None

        if img.shape[2] == 3:
            alpha = np.full((*img.shape[:2], 1), 255, dtype=np.uint8)
            img = np.concatenate([img, alpha], axis=2)

        self._cache[name] = img
        return img

    def get_or_create_solid_sprite(self, name: str, color_bgra: tuple,
                                   size: tuple[int, int]) -> np.ndarray:
        if name in self._cache:
            return self._cache[name]
        img = np.full((size[1], size[0], 4), color_bgra, dtype=np.uint8)
        self._cache[name] = img
        return img

    def clear(self):
        self._cache.clear()