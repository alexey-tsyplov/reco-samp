from typing import NamedTuple

from torch import Tensor


class UserPositiveNegativeTriplet(NamedTuple):
    user_id: Tensor
    positive_item_id: Tensor
    negative_item_id: Tensor


__all__ = ["UserPositiveNegativeTriplet"]
