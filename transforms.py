from typing import Any

import numpy as np
import torch


class ToInt16:
    def __call__(self, x: np.ndarray, *args: Any, **kwds: Any) -> np.ndarray:
        print(x)
        return (x / x.max() * 255).astype("uint8")


class ToFloat:
    def __call__(
        self, x: torch.Tensor, *args: Any, **kwds: Any
    ) -> torch.Tensor:
        return x.type(torch.float)
