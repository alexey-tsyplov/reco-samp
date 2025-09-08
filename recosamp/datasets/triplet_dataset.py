from typing import Optional

import polars as pl
import torch
from polars import DataFrame
from torch import Generator, Tensor
from torch.utils.data import Dataset

from recosamp.typing import IntTensorDict, UserPositiveNegativeTriplet
from recosamp.utils import check_dataset, is_sequence_like


class TripletDataset(Dataset[UserPositiveNegativeTriplet]):
    """
    Represents dataset which contains user-item interaction pairs
    and samples negative examples for each such pair.

    Attributes:
        length (int): the length of the dataset
        user_id (Tensor): 1-D tensor containing user IDs
        item_id (Tensor): 1-D tensor containing item IDs or 2-D
        tensor containing users' interaction sequences. If 2-D,
        each sequence must be padded to equal length.

        num_negatives (int): the number of negative examples
        sampled for each pair.

        probabilities (Tensor): square matrix containing the
        probabilities to sample items to each other as a
        negative example.

        generator (Generator): A torch random numbers generator
        all_user_items (IntTensorDict): A hash table containing
        all positive interactions by user ID. Each user may have
        a different number of such interactions.
    """

    def __init__(
        self,
        interactions: DataFrame,
        probabilities: Tensor,
        user_id_column: str = "user_id",
        item_id_column: str = "item_id",
        num_negatives: int = 1,
        generator: Optional[Generator] = None,
    ) -> None:
        """
        TripletDataset is constructed using polars.DataFrame,
        with two columns: user IDs and item IDs, which represent
        positive interactions. More requirements for the dataset
        can be found in the `check_dataset` documentation.

        Args:
            interactions (polars.Dataframe): A DataFrame with schema
            |-- user_id_column: pl.Int64
            |-- item_id_column: pl.Int64 | pl.Array(pl.Int64, n)

            probabilities (Tensor): the probabilities to sample items
            to each other as a negative example.

            user_id_column (str): user IDs column in DataFrame
            item_id_column (str): item IDs column in DataFrame
            num_negatives (int): the number of negative examples to sample
            generator (torch.Generator): a random number generator
        """
        check_dataset(interactions, user_id_column, item_id_column)
        self.length = interactions.height
        self.user_id = torch.from_numpy(interactions[user_id_column].to_numpy()).long()
        self.item_id = torch.from_numpy(interactions[item_id_column].to_numpy()).long()
        self.num_negatives = num_negatives
        self.probabilities = probabilities
        self.generator = generator

        self.all_user_items = self._collect_all_user_items(interactions, user_id_column, item_id_column)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, row: int) -> UserPositiveNegativeTriplet:
        return UserPositiveNegativeTriplet(
            user_id=self.user_id[row],
            positive_item_id=self.item_id[row],
            negative_item_id=self._sample_negative_items(self.user_id[row], self.item_id[row]),
        )

    def _sample_negative_items(self, user_id: Tensor, positive_item_id: Tensor) -> Tensor:
        """
        Samples negative examples for a given user-item pair or
        for each item in a sequence of user interactions.

        1. Selects row (or rows) from self.probabilities for positive
        item (each positive item in sequence).
        2. Finds all user interactions.
        3. For all found interactions, set the probability of sampling it as a negative example to 0.
        4. Sample self.num_negatives examples for the item (each item in sequence)

        Args:
            user_id (Tensor): 0-D tensor with the only user ID
            positive_item_id (Tensor): 0-D tensor with the only item ID
            or 1-D tensor containing the whole interactions sequence.

        Returns:
            (Tensor): 1-D or 2-D tensor of negative examples for input pair.

        Note:
            Uses -1 and +1 to come from item numeration to torch.Tensor numeration
            and back. Basically item ID 0 is a padding ID.
        """
        positives = self.all_user_items[user_id.item()]

        probabilities = self.probabilities[positive_item_id - 1].clone()
        probabilities[..., positives] = 0.0
        probabilities /= probabilities.sum(dim=-1, keepdim=True)

        return torch.multinomial(probabilities, self.num_negatives, replacement=False, generator=self.generator) + 1

    @staticmethod
    def _collect_all_user_items(
        interactions: DataFrame,
        user_id_column: str,
        item_id_column: str,
    ) -> IntTensorDict:
        """
        Collects a hash table containing all items interacted with
        for each user based on their ID. Gathers non-sequential
        datasets, removes duplicates if any.
        """
        gathered_interactions = (
            interactions
            if is_sequence_like(interactions)
            else interactions.group_by(user_id_column).agg(pl.col(item_id_column).explode())
        )

        return {
            user_id: torch.tensor(item_id_sequence).unique()
            for user_id, item_id_sequence in gathered_interactions.iter_rows()
        }
