import pandas as pd

from pandas_splitter import get_batch_boundaries, batched_dataframe
import pytest


@pytest.mark.parametrize('it,chunk_size,expected', [
    ([], 3, []),
    ([1], 3, [(0, 1)]),
    ([1, 2], 3, [(0, 2)]),
    ([0, 1, 2, 3], 1, [(0, 1), (1, 2), (2, 3), (3, 4)]),
    ([0, 1, 2, 3, 4, 5], 3, [(0, 3), (3, 6)]),
    ([0, 1, 2, 3, 4, 5, 6], 3, [(0, 3), (3, 7)]),
    ([1, 1, 2, 2, 3, 3], 2, [(0, 2), (2, 4), (4, 6)]),
    ([1, 1, 2, 2, 3, 3], 3, [(0, 6)]),
    ([1, 2, 3, 4, 5, 5, 5, 5], 3, [(0, 3), (3, 8)]),
])
def test__batched_sorted_iter(it, chunk_size, expected):
    assert list(get_batch_boundaries(it=it, chunk_size=chunk_size)) == expected


@pytest.mark.parametrize('chunk_size', [-100, -5, 0])
def test__batched_sorted_iter__negative(chunk_size):
    with pytest.raises(ValueError):
        list(batched_dataframe(
                df=pd.DataFrame.from_records([[1], [2]], columns=['dt']),
                chunk_size=chunk_size
        ))


def test__batched_sorted_iter__empty():
    ...


def test__batched_sorted_iter__positive():
    ...
