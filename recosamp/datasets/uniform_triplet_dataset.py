from typing import Optional

import torch
from polars import DataFrame
from torch import Generator

from .triplet_dataset import TripletDataset


class UniformTripletDataset(TripletDataset):
    """
    Represents TripletDataset with equal probabilities
    to sample items to each other as negative example.
    """
    def __init__(
        self,
        interactions: DataFrame,
        user_id_column: str = "user_id",
        item_id_column: str = "item_id",
        num_negatives: int = 1,
        generator: Optional[Generator] = None,
    ) -> None:
        num_items = interactions.n_unique(item_id_column)
        super().__init__(
            interactions,
            user_id_column=user_id_column,
            item_id_column=item_id_column,
            num_negatives=num_negatives,
            probabilities=torch.ones((num_items, num_items), dtype=torch.float) / num_items,
            generator=generator,
        )
