import os
import sys

import pandas as pd
import pytest
from plotly import graph_objects as go

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rsfp import utils


def test_inject_default_kwargs():
    kwargs = {'a': 1, 'b': 2}
    default_kwargs = {'b': 3, 'c': 4}
    result = utils.inject_default_kwargs(kwargs, **default_kwargs)
    assert result == {'a': 1, 'b': 2, 'c': 4}


def test_plot_histogram_kde():
    df = pd.DataFrame({'value_column': [1, 2, 3, 4, 5]})
    result = utils.plot_histogram_kde(df, 'value_column')
    assert isinstance(result, go.Figure)


def test_plot_histogram_kde_with_group_column():
    df = pd.DataFrame({
        'value_column': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'group_column': ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B']
    })
    result = utils.plot_histogram_kde(df, 'value_column', 'group_column')
    assert isinstance(result, go.Figure)
