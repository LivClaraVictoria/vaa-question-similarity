import os
import sys

import pandas as pd
import pytest
from plotly import graph_objects as go

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rsfp import data
import pytest


def test_build_all():
    df_voters, df_candidates, df_questions = data.build_all(verbose=True)
    df_voters19, df_candidates19, df_questions19 = data.build_all19(verbose=True)

    # check types
    assert isinstance(df_voters, pd.DataFrame)
    assert isinstance(df_candidates, pd.DataFrame)
    assert isinstance(df_questions, pd.DataFrame)
    assert isinstance(df_voters19, pd.DataFrame)
    assert isinstance(df_candidates19, pd.DataFrame)
    assert isinstance(df_questions19, pd.DataFrame)

    assert isinstance(df_voters, data.SVDataFrame)
    assert isinstance(df_candidates, data.SVDataFrame)
    assert isinstance(df_voters19, data.SVDataFrame)
    assert isinstance(df_candidates19, data.SVDataFrame)

    # check number of answer, weight and cleavage cols
    assert df_voters.a().shape[1] == 75
    assert df_candidates.a().shape[1] == 75
    assert df_voters19.a().shape[1] == 75
    assert df_candidates19.a().shape[1] == 75

    assert df_voters.w().shape[1] == 75
    assert df_candidates.w().shape[1] == 0
    assert df_voters19.w().shape[1] == 75
    assert df_candidates19.w().shape[1] == 0

    assert df_voters.c().shape[1] == 8
    assert df_candidates.c().shape[1] == 8
    assert df_voters19.c().shape[1] == 8
    assert df_candidates19.c().shape[1] == 8
