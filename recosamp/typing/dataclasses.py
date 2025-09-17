from dataclasses import dataclass
from torch import Tensor


@dataclass
class ImportanceData:
    user_importance: Tensor
    item_importance: Tensor
    user_importance_threshold: float
    item_importance_threshold: float
