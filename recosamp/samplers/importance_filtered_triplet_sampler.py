import math
from typing import Iterator, Optional

import torch
from torch.utils.data import Sampler, DistributedSampler

from recosamp.datasets import TripletDataset
from recosamp.typing import ImportanceData


class ImportanceFilteredTripletSampler(Sampler[int]):
    def __init__(
        self,
        triplet_dataset: TripletDataset,
        importance_data: ImportanceData,
    ) -> None:
        super().__init__()
        self.triplet_dataset = triplet_dataset
        self.importance_data = importance_data

    def __len__(self):
        mask = self.importance_data.user_importance[self.triplet_dataset.user_id].ge(
            self.importance_data.user_importance_threshold
        )
        if not self.triplet_dataset.is_sequence_like:
            mask &= self.importance_data.item_importance[self.triplet_dataset.item_id].ge(
                self.importance_data.item_importance_threshold
            )

        return mask.int().sum().item()

    def __iter__(self) -> Iterator[int]:
        mask = self.importance_data.user_importance[self.triplet_dataset.user_id].ge(
            self.importance_data.user_importance_threshold
        )
        if not self.triplet_dataset.is_sequence_like:
            mask &= self.importance_data.item_importance[self.triplet_dataset.item_id].ge(
                self.importance_data.item_importance_threshold
            )

        return iter(torch.argwhere(mask).flatten().tolist())


class ImportanceFilteredTripletDistributedSampler(DistributedSampler[int]):
    def __init__(
        self,
        triplet_dataset: TripletDataset,
        importance_data: ImportanceData,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
    ) -> None:
        super().__init__(dataset=triplet_dataset, num_replicas=num_replicas, rank=rank)
        self.triplet_dataset = triplet_dataset
        self.importance_data = importance_data

        self.num_samples = int(math.ceil(len(self.triplet_dataset) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __len__(self):
        mask = self.importance_data.user_importance[self.triplet_dataset.user_id].ge(
            self.importance_data.user_importance_threshold
        )
        if not self.triplet_dataset.is_sequence_like:
            mask &= self.importance_data.item_importance[self.triplet_dataset.item_id].ge(
                self.importance_data.item_importance_threshold
            )

        return mask[self.rank : self.total_size : self.num_replicas].int().sum().item()

    def __iter__(self) -> Iterator[int]:
        mask = self.importance_data.user_importance[self.triplet_dataset.user_id].ge(
            self.importance_data.user_importance_threshold
        )
        if not self.triplet_dataset.is_sequence_like:
            mask &= self.importance_data.item_importance[self.triplet_dataset.item_id].ge(
                self.importance_data.item_importance_threshold
            )

        indices = torch.argwhere(mask).flatten().tolist()
        return iter(indices[self.rank : self.total_size : self.num_replicas])
