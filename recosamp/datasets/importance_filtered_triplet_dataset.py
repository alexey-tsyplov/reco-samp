from typing import Literal, Optional

import torch
from polars import DataFrame
from torch import Generator, Tensor

from recosamp.typing import ImportanceData, UserPositiveNegativeTriplet

from .triplet_dataset import TripletDataset


class ImportanceFilteredTripletDataset(TripletDataset):
    """
    A version of the TripletDataset where the output is filtered
    based on user and item importance. If a user's importance is
    lower than the threshold, their ID is replaced with 0, as is
    the case for items. The importance thresholds can be changed
    during the training process.
    """

    def __init__(
        self,
        interactions: DataFrame,
        probabilities: Tensor,
        importance_data: ImportanceData,
        user_id_column: str = "user_id",
        item_id_column: str = "item_id",
        num_negatives: int = 1,
        generator: Optional[Generator] = None,
    ) -> None:
        super().__init__(
            interactions,
            probabilities,
            user_id_column=user_id_column,
            item_id_column=item_id_column,
            num_negatives=num_negatives,
            generator=generator,
        )

        self.importance_data = importance_data

    def __getitem__(self, row: int) -> UserPositiveNegativeTriplet:
        triplet = super().__getitem__(row)
        return UserPositiveNegativeTriplet(
            self._replace_with_padding_idx(triplet.user_id, mode="user"),
            self._replace_with_padding_idx(triplet.positive_item_id, mode="item"),
            self._replace_with_padding_idx(triplet.negative_item_id, mode="item"),
        )

    def _replace_with_padding_idx(self, ids: Tensor, mode: Literal["user", "item"]) -> Tensor:
        if mode == "user":
            return torch.where(
                self.importance_data.user_importance[ids - 1].lt(self.importance_data.user_importance_threshold),
                self.PADDING_IDX,
                ids,
            )

        if mode == "item":
            return torch.where(
                self.importance_data.item_importance[ids - 1].lt(self.importance_data.item_importance_threshold),
                self.PADDING_IDX,
                ids,
            )

        raise ValueError(f"mode must be either user or item, got {mode}")
