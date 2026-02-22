from typing import Callable

import numpy as np
from scipy.stats import spearmanr


def rank_biased_overlap(a, b, p=0.5, depth=None):
    """
    Calculate the Rank Biased Overlap (RBO) between two lists of ranks.

    RBO measures the similarity between two ranked lists, taking into account
    the depth of overlap with a weighted geometric series.

    Parameters
    ----------
    a : array-like
        First array of ranks.
    b : array-like
        Second array of ranks.
    p : float, optional
        Weighting factor for geometric series. Defaults to 0.5.
    depth : int, optional
        Depth of overlap to consider. If None, defaults to the length of the
        input lists. Defaults to None.

    Returns
    -------
    float
        Rank Biased Overlap (RBO) between the two lists of ranks.
    """
    assert len(a) == len(b), "both distance vectors have to be of equal length"

    if depth is None:
        depth = len(a)

    order_a, order_b = np.argsort(a), np.argsort(b)
    weighted_overlaps = [p ** (i - 1) * (len(set(order_a[:i]) & set(order_b[:i])) / i) for i in
                         range(1, depth + 1)]

    # normalize overlap
    if p == 1:
        total_sum = 0
        for i, el in enumerate(weighted_overlaps):
            total_sum = (total_sum * i + el) / (i + 1)
        return total_sum
    else:
        return (1 - p) * np.sum(weighted_overlaps)


def average_rank_correlation(voter_candidate_distances1: np.ndarray, voter_candidate_distances2: np.ndarray, rank_correlation_function: Callable = None):
    """
    Calculate the average rank correlation between corresponding rows in two matrices.

    Parameters
    ----------
    voter_candidate_distances1 : numpy.ndarray
        First matrix.
    voter_candidate_distances2 : numpy.ndarray
        Second matrix.
    rank_correlation_function : str, optional
        The method used to compute the correlation. Default is "spearman".

    Returns
    -------
    float
        Average Spearman rank correlation.
    """

    # check if the matrices have the same shape
    assert voter_candidate_distances1.shape == voter_candidate_distances2.shape, "Matrices must have the same shape"

    if rank_correlation_function is None:
        rank_correlation_function = lambda a, b: spearmanr(a, b)[0]

    # Initialize variables
    avg_correlation = np.mean([
        rank_correlation_function(row1, row2)
        for row1, row2 in zip(voter_candidate_distances1, voter_candidate_distances2)
    ])

    return avg_correlation
