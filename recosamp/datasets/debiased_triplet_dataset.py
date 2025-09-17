from typing import Optional

import torch
from polars import DataFrame
from torch import Generator, Tensor

from .triplet_dataset import TripletDataset


class DebiasedTripletDataset(TripletDataset):
    """
    A version of the TripletDataset where negative examples are
    sampled only from more popular items. This means that in each
    triplet (user_id, positive_item_id, negative_item_id) there
    is always

    popularity(negative_item_id) > popularity(positive_item_id).
    """

    def __init__(
        self,
        interactions: DataFrame,
        probabilities: Tensor,
        item_popularity: Tensor,
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

        self.item_popularity = item_popularity

    def _sample_negative_items(self, user_id: Tensor, positive_item_id: Tensor) -> Tensor:
        positives = self.all_user_items[user_id.item()]

        probabilities = self.probabilities[positive_item_id - 1].clone()
        probabilities[..., positives] = 0.0
        probabilities[self.item_popularity[positive_item_id].unsqueeze(-1).gt(self.item_popularity)] = 0.0

        return torch.multinomial(probabilities, self.num_negatives, replacement=False, generator=self.generator) + 1
