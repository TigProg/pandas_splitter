import itertools
from collections.abc import Iterator

import pandas as pd


def get_batch_boundaries(it: Iterator, chunk_size: int) -> Iterator[tuple[int, int]]:
    i = j = 0
    prev_boundaries = None
    for elem, sub_it in itertools.groupby(it):
        length = sum(1 for _ in sub_it)
        j += length
        if j - i >= chunk_size:
            if prev_boundaries is not None:
                yield prev_boundaries
            prev_boundaries = i, j
            i = j

    if prev_boundaries:
        yield prev_boundaries[0], j
    elif j != 0:
        yield 0, j


def batched_dataframe(df: pd.DataFrame,
                      chunk_size: int,
                      datetime_column_name: str = 'dt') -> Iterator[pd.DataFrame]:
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive')
    if datetime_column_name not in df.columns:
        raise KeyError(f'column [{datetime_column_name}] not exist in df')

    n_rows = df.shape[0]
    if chunk_size >= n_rows:
        yield df
        return

    df_sorted = df.sort_values(by=[datetime_column_name])
    dt_series = df_sorted[datetime_column_name]
    for start, end in get_batch_boundaries(dt_series, chunk_size=chunk_size):
        yield df_sorted.iloc[start:end]
