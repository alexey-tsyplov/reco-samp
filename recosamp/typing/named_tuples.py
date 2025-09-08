from typing import NamedTuple

from torch import Tensor

class UserPositiveNegativeTriplet(NamedTuple):
    """
    Represent container for triplet
    (user_id, positive_item_id, negative_item_id).

    In torch.utils.data.Dataset inheritors usually
    - user_id is zero-dimensional torch.Tensor with
    one user_id
    - positive_item_id is zero-dimensional torch.Tensor
    with one interacted item or 1-D tensor containing
    the whole sequence of items user interacted with.
    - negative_item_id dimensionality depends on
    positive_item_id: its positive_item_id.ndims + 1
    and the added dimensionality is equal to the number
    of sampled negatives.

    In torch.utils.data.DataLoader collate_fn prepends
    batch_size dimensionality in each case.
    """
    user_id: Tensor
    positive_item_id: Tensor
    negative_item_id: Tensor
