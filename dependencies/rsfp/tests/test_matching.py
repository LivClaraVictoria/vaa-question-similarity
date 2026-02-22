import pytest
import numpy as np
from rsfp.matching import get_distances, DISTANCE_METHODS, calculate_distances
from rsfp.data import build_voters, build_candidates

# Create dummy data
df_voters = build_voters().sample(100, random_state=0)

df_candidates = build_candidates().sample(10, random_state=0)


@pytest.mark.parametrize("distance_method", DISTANCE_METHODS)
def test_get_distances(distance_method):
    # Call the function with the dummy data and the specific distance method
    result = get_distances(df_voters, df_candidates, distance_method)

    # Check if the output is a numpy array
    assert isinstance(result, np.ndarray)

    # Check if the shape of the output is as expected (number of voters x number of candidates)
    assert result.shape == (len(df_voters), len(df_candidates))

    # Check if the output does not contain any NaN values
    assert not np.isnan(result).any()


@pytest.mark.parametrize("distance_method", DISTANCE_METHODS)
def test_calculate_distances(distance_method):
    # Call the function with the dummy data and the specific distance method
    voter_answers = np.array(
        [[1, 2, 3, 4, 5],
         [5, 4, 3, 2, 1],
         [1, 1, 1, 1, 1]]
    )

    result = calculate_distances(voter_answers, voter_answers, distance_method=distance_method)

    # Check if the output is a numpy array
    assert isinstance(result, np.ndarray)

    # Check if the shape of the output is as expected
    assert result.shape == (len(voter_answers), len(voter_answers))

    # Check if the output does not contain any NaN values
    assert not np.isnan(result).any()

    # Check if the output is symmetric
    assert np.allclose(result, result.T)

    # Check if the diagonal values are zero
    if distance_method in ['L1', 'L2', 'angular_unweighted', 'angular']:
        assert np.allclose(np.diag(result), 0, atol=1e-3)

    # Check if the output is correct for the specific distance method
    if distance_method == 'L2':
        expected_result = np.array(
            [[0., 6.32455532, 5.47722558],
             [6.32455532, 0., 5.47722558],
             [5.47722558, 5.47722558, 0.]]
        )
        assert np.allclose(result, expected_result)
    elif distance_method == 'L1':
        expected_result = np.array(
            [[0.0, 12.0, 10.0],
             [12.0, 0.0, 10.0],
             [10.0, 10.0, 0.0]]
        )
        assert np.allclose(result, expected_result)
    elif distance_method == 'AC':
        expected_result = np.array(
            [[-5, -1, -1],
             [-1, -5, -1],
             [-1, -1, -5]]
        )
        assert np.allclose(result, expected_result)
    elif distance_method in ['angular_unweighted', 'angular']:
        # check if all distances are between 0 and 90 degrees
        assert np.all(result >= 0)
        assert np.all(result <= 90)

