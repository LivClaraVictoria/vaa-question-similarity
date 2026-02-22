import os
import time

import numpy as np
from scipy.optimize import dual_annealing

from rsfp.constants import SEATS_PER_CANTON
from rsfp.data import build_all
from rsfp.matching import calculate_distances


# HELPER FUNCTIONS

def map_to_answer_possibilities_candidate(answer_vector):
    # Ensure the input vector does not contain NaN values
    if np.isnan(answer_vector).any():
        raise ValueError("Answer vector contains NaN values. Please ensure the input is valid.")

    # Define the possible discrete values for each segment
    possible_values_1 = np.array([0, 25, 75, 100])
    possible_values_2 = np.array([0, 17, 33, 50, 67, 83, 100])
    possible_values_3 = np.array([0, 25, 50, 75, 100])

    # Vectorized operation to map the first 60 elements
    diffs_1 = np.abs(answer_vector[:60, np.newaxis] - possible_values_1)
    answer_vector[:60] = possible_values_1[np.argmin(diffs_1, axis=1)]

    # Vectorized operation to map elements 60 to 66
    diffs_2 = np.abs(answer_vector[60:67, np.newaxis] - possible_values_2)
    answer_vector[60:67] = possible_values_2[np.argmin(diffs_2, axis=1)]

    # Vectorized operation to map elements 67 to 74
    diffs_3 = np.abs(answer_vector[67:75, np.newaxis] - possible_values_3)
    answer_vector[67:75] = possible_values_3[np.argmin(diffs_3, axis=1)]

    return answer_vector


def load_x0_if_available(district, df_c_district, use_first_percent: bool = False):
    """
    This function checks if an .npy file with the optimized answer vector for the specified district exists.
    If the file exists, it loads and returns the vector. Otherwise, it returns None.

    Args:
    - district (str): The district name or identifier.

    Returns:
    - np.ndarray: The loaded answer vector if the file exists, otherwise None.
    """
    filename = f'answer_optimization/answer_optimization_vector_{district}{"_firstperc" if use_first_percent else ""}.npy'

    # Check if the file exists
    if os.path.isfile(filename):
        # Load and return the vector
        return np.load(filename)
    elif not use_first_percent:
        # Return None if the file does not exist
        return df_c_district[df_c_district['_n_recommendations_spc_L2_sv'] == df_c_district[
            '_n_recommendations_spc_L2_sv'].max()].a().to_numpy().ravel()
    else:
        return None


# MAIN SCRIPT

# Load the data

df_voters, df_candidates, df_questions = build_all(verbose=True)

df_voters = df_voters.load_candidate_voting_recommendations('df_voters_recommendations_spc_10_methods',
                                                            selected_distance_methods=['L2_sv'])
df_candidates = df_candidates.load_candidate_recommendation_counts('df_candidates_recommendations_spc_10_methods',
                                                                   selected_distance_methods=['L2_sv'])

# Define the parameters for the optimization

# Optimization Hyperparameters
initial_temp = 10
restart_temp_ratio = 1e-4
seed = 0
max_time_limit = 10 * 60  # 10 minutes for largest canton, scaled down based on number of voters by canton

# Use only the first percent of voters who responded earliest
use_first_percent = True

for district, time_limit in (
        (df_voters['_district'].value_counts() / df_voters['_district'].value_counts().max()).sort_values(
            ascending=True) * max_time_limit).to_dict().items():
    print(district, flush=True)
    # Load Data
    df_v_district = df_voters.district(district).sort_values('_time').head(
        int(0.01 * len(df_voters.district(district)))) if use_first_percent else df_voters.district(district)
    df_c_district = df_candidates.district(district)

    print(f"#Voters: {len(df_v_district)}", flush=True)

    voter_answers = df_v_district.a().to_numpy()
    voter_weights = df_v_district.w().to_numpy()

    # get distance of voters to last visible candidate
    threshold_dists = df_v_district[f'_matchDist_{SEATS_PER_CANTON[district]}_L2_sv'].to_numpy()

    # Initialize variable to keep track of the best visibility
    best_visibility = float('-inf')


    # Define the objective function that takes the candidate answer vector as input and returns the visibility
    def objective_function(candidate_answer_vector, voter_answers, voter_weights, threshold_dists):
        global best_visibility

        candidate_answer_vector_discretized = map_to_answer_possibilities_candidate(candidate_answer_vector)

        crafted_candidate_dists = calculate_distances(voter_answers, candidate_answer_vector_discretized.reshape(1, -1),
                                                      voter_weights)

        visibility = (crafted_candidate_dists.ravel() <= threshold_dists).mean()

        # Check if this is a new high score
        if visibility > best_visibility:
            best_visibility = visibility
            print(f"New high score: Visibility = {visibility}", flush=True)

            # Save the candidate answer vector to a .npy file
            np.save(
                f'answer_optimization/answer_optimization_vector_{district}{"_firstperc" if use_first_percent else ""}.npy',
                candidate_answer_vector_discretized)

        return -visibility  # Return the negative visibility since we are minimizing by default


    # Define bounds for each element of the candidate answer vector (values between 0 and 100)
    bounds = [(0, 100) for _ in range(75)]

    # Record the start time
    start_time = time.time()


    def time_limited_objective(candidate_answer_vector):
        # Check if the time limit has been exceeded
        if time.time() - start_time > time_limit:
            raise TimeoutError("Time limit exceeded")

        # Call the original objective function
        return objective_function(candidate_answer_vector, voter_answers, voter_weights, threshold_dists)


    try:
        # Simulated Annealing optimization
        result = dual_annealing(
            time_limited_objective,
            bounds,
            maxiter=100000,  # Set a large maxiter, the time limit will be the actual stopper
            initial_temp=initial_temp,
            restart_temp_ratio=restart_temp_ratio,
            x0=load_x0_if_available(district, df_c_district, use_first_percent),  # Provide the initial guess,
            # no_local_search=True,
            seed=seed
        )

        # # Define the initial guess (x0)
        # x0 = load_x0_if_available(district, df_c_district, use_first_percent)
        # if x0 is None:
        #     x0 = np.random.uniform(low=0, high=100, size=75)  # If no initial guess is available, generate a random one
        #
        # # Powell's optimization method
        # result = minimize(
        #     time_limited_objective,
        #     x0,
        #     method='Powell',
        #     bounds=bounds,
        #     options={'maxiter': 100000, 'disp': True}
        # )
    except TimeoutError:
        # Handle the timeout and retrieve the best solution found so far
        print(f"Time limit of {time_limit} seconds exceeded!")
    else:
        # Normal completion
        optimized_vector = result.x
        optimized_visibility = -result.fun

        # Print the results
        print("Optimized Candidate Answer Vector:", optimized_vector)
        print("Optimized Visibility:", optimized_visibility)
