import polars as pl
from polars import DataFrame, Series


def is_sequence_like(item_id: Series) -> bool:
    """
    Checks if column contains item sequences or not.

    Args:
        item_id (polars.Series): A column with item IDs
    """
    return isinstance(item_id.dtype, pl.Array)


def check_dataset(
    interactions: DataFrame,
    user_id_column: str = "user_id",
    item_id_column: str = "item_id",
) -> None:
    """
    Check some assertions about input dataset.

    If the item ID column is an array-like structure,
    it is expected to contain sequences of equal lengths.
    The user numeration must be in the range (1, the number of users + 1)
    and the item numeration must be in the
    range (1, the number of items + 1). If the dataset is a sequence-like
    data structure, each user should only appear once in the dataset.

    Args:
        interactions (polars.Dataframe): A DataFrame with schema
        |-- user_id_column: pl.Int64
        |-- item_id_column: pl.Int64 | pl.Array(pl.Int64, n)

        user_id_column (str): user IDs column in DataFrame
        item_id_column (str): item IDs column in DataFrame

    Raises:
        Assertion error if one of the conditions is violated.
    """
    assert user_id_column in interactions.columns, f"There is no {user_id_column} column in dataset"
    assert item_id_column in interactions.columns, f"There is no {item_id_column} column in dataset"

    num_users = interactions.n_unique(user_id_column)
    num_items = interactions.n_unique(item_id_column)
    if is_sequence_like(interactions[item_id_column]):
        assert interactions.height == num_users, "In sequence datasets each user shall appear only once"

    assert (
        interactions[user_id_column].min() == 1 and interactions[user_id_column].max() >= num_users
    ), "User numeration shall be from 1 to the number of users"

    assert (
        interactions[item_id_column].min() == 1 and interactions[item_id_column].max() >= num_items
    ), "Item numeration shall be from 1 to the number of items"
