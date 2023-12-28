from datetime import datetime

import pandas as pd
import pytest

from pandas_splitter import get_batch_boundaries, batched_dataframe


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
    ([
         datetime(2023, 1, 1, 10),
         datetime(2023, 1, 1, 20),
         datetime(2023, 1, 1, 20),
         datetime(2023, 1, 2, 10),
         datetime(2023, 1, 2, 20),
         datetime(2023, 1, 3, 10),
         datetime(2023, 1, 4, 10),
     ], 2, [(0, 3), (3, 5), (5, 7)])
])
def test__batched_sorted_iter(it, chunk_size, expected):
    assert list(get_batch_boundaries(it=it, chunk_size=chunk_size)) == expected


def invariant_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(by=list(df.columns)).reset_index(drop=True)


class TestBatchedDataframe:
    @pytest.mark.parametrize('chunk_size', [-100, -5, 0])
    def test_incorrect_chunk_size(self, chunk_size):
        with pytest.raises(ValueError):
            list(batched_dataframe(
                df=pd.DataFrame.from_records([[1], [2]], columns=['dt']),
                chunk_size=chunk_size
            ))

    def test_empty(self):
        # arrange
        df = pd.DataFrame(columns=['dt'])
        expected = pd.DataFrame(columns=['dt'])  # for correct columns

        # act
        actual = batched_dataframe(df=df, chunk_size=2)
        actual_list = list(actual)

        # assert
        assert len(actual_list) == 1
        assert actual_list[0].equals(expected)

    def test_big_chunk_size(self):
        # arrange
        df = pd.DataFrame(
            [
                [datetime(2023, 1, 1, 1), 1],
                [datetime(2023, 1, 1, 3), 2],
                [datetime(2023, 1, 1, 2), 3],
                [datetime(2023, 1, 1, 5), 4],
                [datetime(2023, 1, 1, 0), 5],
            ],
            columns=['dt', 'other'],
        )
        chunk_size = 100

        # act
        actual = batched_dataframe(df=df, chunk_size=chunk_size)
        actual_list = list(actual)

        # assert
        assert len(actual_list) == 1
        assert invariant_dataframe(df=actual_list[0]).equals(
            invariant_dataframe(df=df),
        )

    @pytest.mark.parametrize('chunk_size', [1, 10, 100])
    def test_base(self, chunk_size):
        # arrange
        df_unique = pd.date_range(
            "2023-01-01 00:00:00", "2023-01-01 00:01:00", freq="s",
        )
        df_repeated = pd.DataFrame({"dt": df_unique.repeat(10)})

        # act
        actual = batched_dataframe(df=df_repeated, chunk_size=chunk_size)
        actual_list = list(actual)
        actual_union = pd.concat(actual_list)

        # assert
        unique_dts = set()
        for batch in actual_list:
            assert batch.shape[0] >= chunk_size

            current_dts = set(batch['dt'])
            assert len(unique_dts & current_dts) == 0
            current_dts.update(current_dts)

        assert invariant_dataframe(actual_union).equals(
            invariant_dataframe(df_repeated),
        )
