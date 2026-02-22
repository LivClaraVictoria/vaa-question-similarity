import numpy as np
from tqdm.notebook import tqdm

from .constants import (
    ANSWER_POSSIBILITIES,
    ROW_COL_INDICES,
    L1_DIST_MAT,
    DIRECTIONAL_DIST_MAT,
    HYBRID_DIST_MAT,
    L2_DIST_MAT,
    QUESTION_TYPE2IDX,
    QUESTION_TYPE2IDX19,
    SEATS_PER_CANTON,
    SEATS_PER_CANTON19,
    L1_DIST_MAT_UNSCALED,
    FILTERING_METHODS,
)
from .data import SVDataFrame
from .utils import update_question_type2idx, get_cols


def add_candidate_voting_recommendations(
    df_voters: SVDataFrame,
    df_candidates: SVDataFrame,
    distance_method: str = "L2_sv",
    filtering_method: str = None,
    inv_covariance_matrix: np.ndarray = None,
    question_type2idx: dict[int, list[int]] = None,
    n_recommendations: int | dict[str, int] = None,
    district_col: str = "_district",
    progress_bar: bool = True,
    fair_candidate_recommendation_distribution: bool = False,
):
    """
    Add candidate recommendations to voter dataframe based on distances computed using specified distance method.

    This function calculates the distances between voters and candidates using a specified distance method and
    generates candidate recommendations for each voter. It can also apply filtering methods to prioritize candidates
    with fewer disagreements and distribute recommendations fairly among candidates.

    Parameters
    ----------
    df_voters : SVDataFrame
        DataFrame containing voter data.
    df_candidates : SVDataFrame
        DataFrame containing candidate data.
    distance_method : str, optional
        Distance method to use for computing distances, by default 'L2_sv'. Supported methods include 'L2', 'L1',
        'mahalanobis_unweighted', 'DM_L1_BONUS', and others.
    filtering_method : str, optional
        Name of filtering method to use, by default None. Filtering methods can prioritize candidates with fewer
        disagreements. Supported methods are defined in FILTERING_METHODS.
    inv_covariance_matrix : np.ndarray, optional
        Inverse of covariance matrix for Mahalanobis distance calculation. If needed, by default computed based on df_candidates.
    question_type2idx : dict[int, list[int]], optional
        Dictionary mapping question types to the indices of the corresponding questions in the answer array, by default None.
        This is required for distance methods that use distance matrices. If needed, by default QUESTION_TYPE2IDX is used.
    n_recommendations : int | dict[str, int], optional
        Number of recommendations to generate for each voter. Can be an integer or a dictionary specifying
        the number of recommendations per district. By default, SEATS_PER_CANTON is used.
    district_col : str, optional
        Name of the column in the dataframes specifying the district, by default '_district'.
    progress_bar : bool, optional
        Whether to show a progress bar, by default True.
    fair_candidate_recommendation_distribution : bool, optional
        Whether to distribute candidate recommendations fairly among candidates, by default False.

    Returns
    -------
    SVDataFrame
        DataFrame with candidate recommendations added. If fair_candidate_recommendation_distribution is True,
        returns a tuple of DataFrames (df_voters, df_candidates) with additional recommendation counts for candidates.

    Notes
    -----
    - The function ensures that voter and candidate dataframes are from the same term.
    - For Mahalanobis distance, the inverse covariance matrix is calculated if not provided.
    - For 'DM_L1_BONUS' distance method, the appropriate question_type2idx is selected based on the term.
    - The number of recommendations per district is determined based on the term if not provided.
    - The function can optionally show a progress bar for processing districts.
    - If filtering_method is specified, pairwise disagreement counts are calculated and used to modify distances.
    - Recommendations are assigned to voters based on the smallest distances, with tie-breaking using last names (if available).
    - If fair_candidate_recommendation_distribution is enabled, recommendations are distributed fairly among candidates.

    Examples
    --------
    >>> # using tie-breaking
    >>> df_voters_with_recommendations = add_candidate_voting_recommendations(df_voters, df_candidates)
    >>> df_candidates_with_counts = df_candidates.add_recommendation_counts(df_voters_with_recommendations)
    >>> # using fair distribution
    >>> df_voters_with_recommendations, df_candidates_with_fair_counts = add_candidate_voting_recommendations(
    ...     df_voters, df_candidates, fair_candidate_recommendation_distribution=True)
    """
    assert (
        df_voters.term == df_candidates.term
    ), "Voter and candidate dataframes must be from the same term."

    if distance_method == "mahalanobis_unweighted":
        inv_covariance_matrix = (
            np.linalg.inv(np.cov(df_candidates.a().values, rowvar=False))
            if inv_covariance_matrix is None
            else inv_covariance_matrix
        )
    if distance_method == "DM_L1_BONUS":
        question_type2idx = (
            (QUESTION_TYPE2IDX if df_voters.term == 2023 else QUESTION_TYPE2IDX19)
            if question_type2idx is None
            else question_type2idx
        )
    n_recommendations = (
        (SEATS_PER_CANTON if df_voters.term == 2023 else SEATS_PER_CANTON19)
        if n_recommendations is None
        else n_recommendations
    )

    df_voters = df_voters.copy()
    df_candidates = df_candidates.copy()

    iterator = (
        df_voters[district_col].unique()
        if not progress_bar
        else tqdm(df_voters[district_col].unique())
    )
    for district in iterator:
        voter_district_mask = df_voters["_district"] == district
        candidate_district_mask = df_candidates["_district"] == district

        # get the number of recommendations for the current district
        n_recs = min(
            (
                n_recommendations
                if isinstance(n_recommendations, int)
                else n_recommendations[district]
            ),
            candidate_district_mask.sum(),
        )

        # calculate pairwise distances between all voters and candidates
        distances = get_distances(
            df_voters[voter_district_mask],
            df_candidates[candidate_district_mask],
            distance_method=distance_method,
            inv_covariance_matrix=inv_covariance_matrix,
            question_type2idx=question_type2idx,
            progress_bar=progress_bar,
        )

        if filtering_method is not None:
            # calculate pairwise disagreement counts between all voters and candidates
            disagreement_counts = get_disagreement_counts(
                df_voters[voter_district_mask],
                df_candidates[candidate_district_mask],
                **FILTERING_METHODS[filtering_method],
            )
            # get indices of top recommendations modify distances so that candidates with fewer disagreements are
            # always preferred, even if they are more distant
            modified_distances = (
                distances + (distances.max() - distances.min()) * disagreement_counts
            )
            rec_candidates_idx = np.argsort(modified_distances, axis=1)[:, :n_recs]

            if fair_candidate_recommendation_distribution:
                candidate_recommendations_spc, candidate_recommendations_top = (
                    distribute_candidate_recommendations(
                        voter_candidate_distances=modified_distances,
                        n_votes_in_district=n_recs,
                    )
                )
        else:
            # get indices of recommendations using smartvotes' method of breaking ties based on last names
            if "lastname" in df_candidates.columns:
                # if lastname is available, use it to break ties (Smartvote)
                distances_last_names = np.zeros_like(
                    distances, dtype=[("distance", float), ("last_name_rank", int)]
                )
                distances_last_names["distance"] = distances
                distances_last_names["last_name_rank"] = (
                    df_candidates.district(district)["lastname"]
                    .rank(method="min")
                    .astype(int)
                    .to_numpy()
                )
                rec_candidates_idx = np.argsort(
                    distances_last_names, axis=1, order=("distance", "last_name_rank")
                )[:, :n_recs]
            else:
                # if lastname is missing, use default method (np.argsort)
                rec_candidates_idx = np.argsort(distances, axis=1)[:, :n_recs]

            if fair_candidate_recommendation_distribution:
                candidate_recommendations_spc, candidate_recommendations_top = (
                    distribute_candidate_recommendations(
                        voter_candidate_distances=distances, n_votes_in_district=n_recs
                    )
                )

        # get map from candidate index to candidate ID
        candidate_idx_to_id = df_candidates.loc[
            candidate_district_mask, "ID_candidate"
        ].values

        # assign recommendation IDs and distances to voters
        filtering_suffix = (
            f"_{filtering_method}" if filtering_method is not None else ""
        )
        df_voters.loc[
            voter_district_mask,
            [
                f"_matchID_{i}_{distance_method}{filtering_suffix}"
                for i in range(1, n_recs + 1)
            ],
        ] = candidate_idx_to_id[rec_candidates_idx]
        df_voters.loc[
            voter_district_mask,
            [
                f"_matchDist_{i}_{distance_method}{filtering_suffix}"
                for i in range(1, n_recs + 1)
            ],
        ] = distances[np.arange(distances.shape[0])[:, None], rec_candidates_idx]

        if fair_candidate_recommendation_distribution:
            # assign candidate recommendation counts to candidates
            df_candidates.loc[
                candidate_district_mask,
                f"_n_recommendations_fair_spc_{distance_method}{filtering_suffix}",
            ] = candidate_recommendations_spc
            df_candidates.loc[
                candidate_district_mask,
                f"_n_recommendations_fair_top_{distance_method}{filtering_suffix}",
            ] = candidate_recommendations_top

    if fair_candidate_recommendation_distribution:
        return df_voters, df_candidates
    else:
        return df_voters


def add_list_voting_recommendations(
    df_voters: SVDataFrame,
    df_candidates: SVDataFrame,
    distance_method: str = "L2_sv",
    filtering_method: str = None,
    average_list_positions: bool = False,
    inv_covariance_matrix: np.ndarray = None,
    question_type2idx: dict[int, list[int]] = None,
    district_col: str = "_district",
    list_col: str = "ID_list",
    progress_bar: bool = False,
):
    """
    Add list recommendations to voter dataframe based on average distances to all candidates in a list using a
    specific distance method.

    This function calculates the distances between voters and candidates using a specified distance method and
    generates list recommendations for each voter. It can also apply filtering methods to prioritize candidates
    with fewer disagreements and use average list positions as representative positions of the list.

    Parameters
    ----------
    df_voters : SVDataFrame
        DataFrame containing voter data.
    df_candidates : SVDataFrame
        DataFrame containing candidate data.
    distance_method : str, optional
        Distance method to use for computing distances, by default 'L2'.
    filtering_method : str, optional
        Name of filtering method to use, by default None.
    average_list_positions : bool, optional
        Whether to use the average positions of candidates in a list as representative position of the list, by default False.
    inv_covariance_matrix : np.ndarray, optional
        Inverse covariance matrix for Mahalanobis distance calculation, by default None.
    question_type2idx : dict[int, list[int]], optional
        Dictionary mapping question types to the indices of the corresponding questions in the answer array, by default None.
    district_col : str, optional
        Name of the column in the dataframes specifying the district, by default '_district'.
    list_col : str, optional
        Name of the column in the candidate dataframe specifying the list, by default 'ID_list'.
    progress_bar : bool, optional
        Whether to show a progress bar, by default False.

    Returns
    -------
    SVDataFrame
        DataFrame with candidate recommendations added.

    Notes
    -----
    - The function ensures that voter and candidate dataframes are from the same term.
    - For Mahalanobis distance, the inverse covariance matrix is calculated if not provided.
    - For 'DM_L1_BONUS' distance method, the appropriate question_type2idx is selected based on the term.
    - The function can optionally show a progress bar for processing districts.
    - If filtering_method is specified, pairwise disagreement counts are calculated and used to modify distances.
    - Recommendations are assigned to voters based on the smallest distances.
    - If average_list_positions is True, the average positions of candidates in a list are used as the representative position of the list.
    - Filtering method is not supported when using average list positions.
    """
    assert (
        df_voters.term == df_candidates.term
    ), "Voter and candidate dataframes must be from the same term."

    if distance_method == "mahalanobis_unweighted":
        inv_covariance_matrix = (
            np.linalg.inv(np.cov(df_candidates.a().values, rowvar=False))
            if inv_covariance_matrix is None
            else inv_covariance_matrix
        )
    if distance_method == "DM_L1_BONUS":
        question_type2idx = (
            (QUESTION_TYPE2IDX if df_voters.term == 2023 else QUESTION_TYPE2IDX19)
            if question_type2idx is None
            else question_type2idx
        )

    df_voters = df_voters.copy()

    iterator = (
        df_voters[district_col].unique()
        if not progress_bar
        else tqdm(df_voters[district_col].unique())
    )
    for district in iterator:
        voter_district_mask = df_voters["_district"] == district
        candidate_district_mask = df_candidates["_district"] == district

        if average_list_positions:
            assert (
                filtering_method is None
            ), "Filtering method is not supported when using average list positions."

            # calculate distances to average list positions
            averaged_list_answers = (
                df_candidates[candidate_district_mask]
                .groupby("ID_list")[get_cols(df_candidates)]
                .mean()
            )

            list_distances = calculate_distances(
                df_voters[voter_district_mask].a().values,
                averaged_list_answers.values,
                df_voters[voter_district_mask].w().values,
                distance_method=distance_method,
                inv_covariance_matrix=inv_covariance_matrix,
                question_type2idx=question_type2idx,
            )

            unique_district_lists = np.array(averaged_list_answers.index)

        else:
            # calculate average of distances to all candidates in a list
            distances = get_distances(
                df_voters[voter_district_mask],
                df_candidates[candidate_district_mask],
                distance_method=distance_method,
                inv_covariance_matrix=inv_covariance_matrix,
                question_type2idx=question_type2idx,
                progress_bar=progress_bar,
            )

            if filtering_method is not None:
                # calculate pairwise disagreement counts between all voters and candidates
                disagreement_counts = get_disagreement_counts(
                    df_voters[voter_district_mask],
                    df_candidates[candidate_district_mask],
                    **FILTERING_METHODS[filtering_method],
                )
                # modify distances so that candidates with fewer disagreements are always preferred, even if they are
                # more distant
                distances += (
                    distances.max(axis=1).reshape(-1, 1)
                    - distances.min(axis=1).reshape(-1, 1)
                ) * disagreement_counts

            # group candidates by list and calculate average distances
            list_distances = []
            unique_district_lists = np.sort(
                df_candidates[candidate_district_mask][list_col].unique()
            )
            for district_list in unique_district_lists:
                # calculate average dists to all candidates of that list for all voters
                candidate_idx = np.where(
                    df_candidates[candidate_district_mask][list_col] == district_list
                )[0]
                list_distances.append(distances[:, candidate_idx].mean(axis=1))
            list_distances = np.vstack(list_distances).T

        # get closest lists IDs and distances
        closest_lists = np.argsort(list_distances, axis=1)
        closest_list_dists = np.sort(list_distances, axis=1)
        closest_list_names = unique_district_lists[closest_lists]

        # assign recommendations to voters
        filtering_suffix = (
            f"_{filtering_method}" if filtering_method is not None else ""
        )
        df_voters.loc[
            voter_district_mask,
            [
                f"_ListID_{i}_{distance_method}{filtering_suffix}"
                for i in range(1, closest_lists.shape[1] + 1)
            ],
        ] = closest_list_names
        df_voters.loc[
            voter_district_mask,
            [
                f"_ListDist_{i}_{distance_method}{filtering_suffix}"
                for i in range(1, closest_lists.shape[1] + 1)
            ],
        ] = closest_list_dists

    return df_voters


def get_distances(
    df_voters: SVDataFrame,
    df_candidates: SVDataFrame,
    distance_method: str = "L2_sv",
    inv_covariance_matrix: np.ndarray = None,
    question_type2idx: dict[int, list[int]] = None,
    progress_bar: bool = False,
):
    """
    This function is a wrapper around the `calculate_distances` function. It takes `SVDataFrame` objects for voters
    and candidates as input and calculates the pairwise distances between each voter and each candidate. The calculation
    is done sequentially over the voters.

    Parameters
    ----------
    df_voters : SVDataFrame
        DataFrame containing voter data.
    df_candidates : SVDataFrame
        DataFrame containing candidate data.
    distance_method : str, optional
        Distance method to use for computing distances, by default 'L2_sv'. Supported methods include 'L2', 'L1',
        'mahalanobis_unweighted', 'DM_L1_BONUS', and others.
    inv_covariance_matrix : np.ndarray, optional
        Inverse of covariance matrix for Mahalanobis distance calculation, by default None. Required if
        distance_method is 'mahalanobis_unweighted'.
    question_type2idx : dict[int, list[int]], optional
        Dictionary mapping question types to the indices of the corresponding questions in the answer array, by default None.
        This is required for distance methods that use distance matrices. If needed, by default QUESTION_TYPE2IDX is used.
    progress_bar : bool, optional
        Whether to show a progress bar, by default False.

    Returns
    -------
    np.ndarray
        Array of distances between each voter and each candidate.
    """
    voter_weights = df_voters.weights().values
    voter_answers = df_voters.answers().values
    candidate_answers = df_candidates.answers().values

    distances = []
    iterator = (
        range(len(voter_answers))
        if not progress_bar
        else tqdm(range(len(voter_answers)))
    )
    for voter_index in iterator:
        non_nan_mask = np.where(~np.isnan(voter_answers[voter_index].astype(float)))[0]
        distance = calculate_distances(
            voter_answers[voter_index, non_nan_mask].reshape(1, -1),
            candidate_answers[:, non_nan_mask],
            voter_weights[voter_index, non_nan_mask].reshape(1, -1),
            distance_method=distance_method,
            inv_covariance_matrix=(
                inv_covariance_matrix[non_nan_mask][:, non_nan_mask]
                if inv_covariance_matrix is not None
                else None
            ),
            question_type2idx=update_question_type2idx(
                question_type2idx, np.isnan(voter_answers[voter_index].astype(float))
            ),
        )
        distances.append(distance)
    distances = np.vstack(distances)

    return distances


def calculate_distances(
    voter_answers: np.ndarray,
    candidate_answers: np.ndarray,
    voter_weights: np.ndarray = None,
    distance_method: str = "L2_sv",
    inv_covariance_matrix: np.ndarray = None,
    question_type2idx: dict[int, list[int]] = None,
) -> np.ndarray:
    """
    Calculate the distances between voter answers and candidate answers using a specified distance method.

    Parameters
    ----------
    voter_answers : np.ndarray
        Array of voter answers, shape (N, D) where N is the number of voters and D is the number of dimensions.
    candidate_answers : np.ndarray
        Array of candidate answers, shape (M, D) where M is the number of candidates and D is the number of dimensions.
    voter_weights : np.ndarray, optional
        Array of voter weights, shape (N, D). If not provided, all weights are assumed to be 1.
    distance_method : str, optional
        The method to calculate the distance.
        Options are:
        - 'L2_sv'
        - 'L2'
        - 'L1'
        - 'AC'
        - 'angular_unweighted'
        - 'angular'
        - 'DM_L1'
        - 'DM_L1_BONUS'
        - 'DM_L2'
        - 'DM_HYBRID'
        - 'DM_DIRECTIONAL'
        - 'mahalanobis_unweighted'
        Default is 'L2'
    inv_covariance_matrix : np.ndarray, optional
        Inverse of covariance matrix, required if distance_method is 'mahalanobis_unweighted'.
    question_type2idx : dict[int, list[int]], optional
        Dictionary mapping question types to the indices of the corresponding questions in the answer array, by default None.

    Returns
    -------
    np.ndarray
        Array of distances, shape (N, M) where N is the number of voters and M is the number of candidates.
    """

    if voter_weights is None:
        voter_weights = np.ones_like(voter_answers)

    voter_answers = (
        voter_answers.reshape(1, -1) if voter_answers.ndim == 1 else voter_answers
    )
    voter_weights = (
        voter_weights.reshape(1, -1) if voter_weights.ndim == 1 else voter_weights
    )
    candidate_answers = (
        candidate_answers.reshape(1, -1)
        if candidate_answers.ndim == 1
        else candidate_answers
    )

    voter_answers = voter_answers.astype(float)
    voter_weights = voter_weights.astype(float)

    # Expand dimensions for broadcasting
    voter_answers_expanded = np.expand_dims(voter_answers, axis=1)
    voter_weights_expanded = np.expand_dims(voter_weights, axis=1)
    candidate_answers_expanded = np.expand_dims(candidate_answers, axis=0)

    # Calculate distances
    if distance_method == "L2":
        distances = np.sqrt(
            np.nansum(
                (voter_answers_expanded - candidate_answers_expanded) ** 2
                * voter_weights_expanded,
                axis=-1,
            )
        )
    elif distance_method == "L2_sv":
        distances = np.sqrt(
            np.nansum(
                (
                    voter_weights_expanded
                    * (voter_answers_expanded - candidate_answers_expanded)
                )
                ** 2,
                axis=-1,
            )
        )
    elif distance_method == "L1":
        distances = np.nansum(
            np.abs((voter_answers_expanded - candidate_answers_expanded))
            * voter_weights_expanded,
            axis=-1,
        )
    elif distance_method == "AC":
        distances = -np.nansum(
            (voter_answers_expanded == candidate_answers_expanded)
            * voter_weights_expanded,
            axis=-1,
        )
    elif distance_method == "angular_unweighted":
        distances = angular_distances(
            np.nan_to_num(voter_answers - 50), candidate_answers - 50
        )
    elif distance_method == "angular":
        distances = angular_distances(
            np.nan_to_num(voter_answers - 50), candidate_answers - 50, voter_weights
        )
    elif distance_method == "DM_L1":
        distances = calculate_dist_mat_distances(
            voter_answers, candidate_answers, voter_weights, L1_DIST_MAT
        )
    elif distance_method == "DM_L1_BONUS":
        distances = []
        for question_type in [4, 5, 7]:
            # build L1 distance matrix with bonus for equal answers
            dist_mat = L1_DIST_MAT_UNSCALED[:, ROW_COL_INDICES[question_type]][
                ROW_COL_INDICES[question_type]
            ].astype(float)
            # add bonus for equal answers (negativ distance)
            dist_mat -= np.diag(dist_mat.sum(axis=0) - np.min(dist_mat.sum(axis=0)))
            # pad distance matrix to 9x9 as calculate_dist_mat_distances expects 9x9 matrix
            padded_dist_mat = np.zeros((9, 9)) - 1e6
            padded_dist_mat[
                np.ix_(ROW_COL_INDICES[question_type], ROW_COL_INDICES[question_type])
            ] = dist_mat
            # find indices of questions with question_type answer options
            question_type_idx = question_type2idx[question_type]
            distances.append(
                calculate_dist_mat_distances(
                    voter_answers[:, question_type_idx],
                    candidate_answers[:, question_type_idx],
                    voter_weights[:, question_type_idx],
                    dist_mat=padded_dist_mat,
                )
            )
        distances = np.stack(distances).sum(axis=0)
    elif distance_method == "DM_L2":
        distances = calculate_dist_mat_distances(
            voter_answers, candidate_answers, voter_weights, L2_DIST_MAT
        )
    elif distance_method == "DM_HYBRID":
        distances = calculate_dist_mat_distances(
            voter_answers, candidate_answers, voter_weights, HYBRID_DIST_MAT
        )
    elif distance_method == "DM_DIRECTIONAL":
        distances = calculate_dist_mat_distances(
            voter_answers, candidate_answers, voter_weights, DIRECTIONAL_DIST_MAT
        )
    elif distance_method == "mahalanobis_unweighted":
        # check if covariance matrix was provided
        assert inv_covariance_matrix is not None, (
            "Inverse of covariance matrix must be provided for Mahalanobis "
            "distance calculation."
        )
        pairwise_diff = voter_answers_expanded - candidate_answers_expanded
        distances = np.sqrt(
            np.sum(
                np.matmul(pairwise_diff, inv_covariance_matrix) * pairwise_diff, axis=-1
            )
        )
    else:
        raise ValueError(f"Invalid distance method: {distance_method}")

    return distances


def calculate_dist_mat_distances(
    voter_answers: np.ndarray,
    candidate_answers: np.ndarray,
    voter_weights: np.ndarray = None,
    dist_mat: np.ndarray = None,
):
    """
    Calculate the distances between voter answers and candidate answers using a specified distance matrix.

    Parameters
    ----------
    voter_answers : np.ndarray
        Array of voter answers, shape (N, D) where N is the number of voters and D is the number of dimensions.
    candidate_answers : np.ndarray
        Array of candidate answers, shape (M, D) where M is the number of candidates and D is the number of dimensions.
    voter_weights : np.ndarray, optional
        Array of voter weights, shape (N, D). If not provided, all weights are assumed to be 1.
    dist_mat : np.ndarray, optional
        Distance matrix to use for calculating distances. If not provided, L2 distance matrix is used.

    Returns
    -------
    np.ndarray
        Array of distances, shape (N, M) where N is the number of voters and M is the number of candidates.

    Notes
    -----
    - The function maps answer values to one of 9 answer categories (ids) using `ANSWER_POSSIBILITIES`.
    - Missing voter answers are handled by mapping NaN values to a filler value.
    - The distance matrix is padded with a zero row and column to handle NaN values.
    - The function calculates the weighted sum of individual distance values to get the total distances.
    """

    if voter_weights is None:
        voter_weights = np.ones_like(voter_answers)

    dist_mat = L2_DIST_MAT if dist_mat is None else dist_mat

    voter_answers = (
        voter_answers.reshape(1, -1) if voter_answers.ndim == 1 else voter_answers
    )
    voter_weights = (
        voter_weights.reshape(1, -1) if voter_weights.ndim == 1 else voter_weights
    )
    candidate_answers = (
        candidate_answers.reshape(1, -1)
        if candidate_answers.ndim == 1
        else candidate_answers
    )

    # mapping from answer values (0, 17, 25, 33, 50, 67, 75, 83, 100) to one of 9 answer categories (ids)
    val2id = np.zeros(101) - 1
    for i, v in enumerate(ANSWER_POSSIBILITIES):
        val2id[v] = i

    # deal with missing voter answers

    # pad distance_matrix with zero row and column that all nan values will be mapped to
    distance_matrix = np.pad(
        dist_mat, ((0, 1), (0, 1)), mode="constant", constant_values=0
    )
    nan_col_row = distance_matrix.shape[0] - 1
    # map nan values to a filler value (99)
    nan_fill_value = 99
    voter_answers = np.nan_to_num(voter_answers, nan=nan_fill_value)
    # map filler value to nan column / row in distance matrix
    val2id[nan_fill_value] = nan_col_row

    # map answer values to ids using val2id
    voter_answer_ids = val2id[voter_answers.astype(int)].astype(int)
    candidate_answer_ids = val2id[candidate_answers.astype(int)].astype(int)

    # get individual distance values for all pairs of voters and candidates
    distances = dist_mat[
        np.expand_dims(voter_answer_ids, axis=1),
        np.expand_dims(candidate_answer_ids, axis=0),
    ]

    # weighted sum of individual distance values -> total_distances has shape (n_voters, n_candidates)
    distances = np.sum(
        distances * np.expand_dims(np.nan_to_num(voter_weights), axis=1), axis=2
    )

    return distances


def angular_distances(
    voter_answers: np.ndarray,
    candidate_answers: np.ndarray,
    voter_weights: np.ndarray = None,
):
    """
    Computes the angular distance (angle in degrees) between every row in the first 2D matrix and every row in the second 2D matrix.

    Parameters
    ----------
    voter_answers : numpy.ndarray
        First 2D matrix of shape (N, D).
    candidate_answers : numpy.ndarray
        Second 2D matrix of shape (M, D).
    voter_weights : numpy.ndarray, optional
        Weights matrix with shape (N, D). Default is None.

    Returns
    -------
    numpy.ndarray
        Resulting angle matrix of shape (N, M).
    """

    if voter_weights is not None:
        # scale the answers by the weights before computing the angles

        voter_answers = voter_answers * voter_weights
        voter_answers /= np.linalg.norm(voter_answers, axis=1, keepdims=True)

        angles = []
        for i, voter_answer in enumerate(voter_answers):
            voter_weight = voter_weights[i]

            cand_answers = candidate_answers * voter_weight
            cand_answers /= np.linalg.norm(cand_answers, axis=1, keepdims=True)

            angle = np.degrees(
                np.arccos(np.clip(voter_answer.reshape(1, -1) @ cand_answers.T, -1, 1))
            )
            angles.append(angle.reshape(1, -1))

        angles = np.vstack(angles)

    else:
        # Compute the norms of each row in matrix_a and matrix_b
        voter_answers = voter_answers / np.linalg.norm(
            voter_answers, axis=1, keepdims=True
        )
        candidate_answers = candidate_answers / np.linalg.norm(
            candidate_answers, axis=1, keepdims=True
        )

        # Compute the angle
        angles = np.degrees(
            np.arccos(np.clip(voter_answers @ candidate_answers.T, -1, 1))
        )

    return angles


def get_disagreement_counts(
    df_voters,
    df_candidates,
    skip_neutral_voter_answers: bool = True,
    voter_map=None,
    candidate_map=None,
):
    """
    Calculate the disagreement counts between voters and candidates.

    Parameters
    ----------
    df_voters : SVDataFrame
        DataFrame containing voter data.
    df_candidates : SVDataFrame
        DataFrame containing candidate data.
    skip_neutral_voter_answers : bool, optional
        If True, neutral voter answers (50) are skipped. Default is True.
    voter_map : function or dict, optional
        Mapping function or dictionary to map voter answers. Default maps answers to extremes (0 or 100).
    candidate_map : function or dict, optional
        Mapping function or dictionary to map candidate answers. Default maps answers to extremes (0 or 100).

    Returns
    -------
    numpy.ndarray
        Matrix of disagreement counts. Each element (i, j) represents the number of disagreements between voter i and candidate j.

    Notes
    -----
    Disagreements are calculated based on extreme answers (0 or 100). Neutral answers (50) can be skipped.
    """

    candidate_map = (
        candidate_map
        if candidate_map is not None
        else lambda x: 100 if x > 50 else 0 if x < 50 else 50
    )
    voter_map = (
        voter_map
        if voter_map is not None
        else lambda x: 100 if x > 50 else 0 if x < 50 else 50
    )

    candidate_extreme_answers = (
        df_candidates.a()
        .map(candidate_map if callable(candidate_map) else candidate_map.get)
        .to_numpy()
    )
    voter_extreme_answers = (
        df_voters.a()
        .map(voter_map if callable(voter_map) else voter_map.get)
        .to_numpy()
    )
    voter_weights = df_voters.w().to_numpy()

    disagreement_matrix = []
    for voter_extreme_answer, voter_weight in zip(voter_extreme_answers, voter_weights):
        strongly_weighted_idx = (
            np.where((voter_extreme_answer != 50) & (voter_weight == 2))[0]
            if skip_neutral_voter_answers
            else np.where(voter_weight == 2)[0]
        )
        disagreement_matrix.append(
            (
                candidate_extreme_answers[:, strongly_weighted_idx]
                != voter_extreme_answer[strongly_weighted_idx]
            )
            .sum(axis=1)
            .reshape(1, -1)
        )

    disagreement_matrix = np.vstack(disagreement_matrix)

    return disagreement_matrix


def distribute_candidate_recommendations(
    voter_candidate_distances: np.ndarray, n_votes_in_district: int
):
    """
    Distribute candidate recommendations fairly based on voter-candidate distances.

    Each voter is recommended a fixed number of candidates with the smallest distances.
    In case of ties at the cutoff, votes are distributed proportionally.

    Parameters
    ----------
    voter_candidate_distances : np.ndarray
        A 2D array of shape (N_voters, N_candidates) containing the distances between voters and candidates.
    n_votes_in_district : int
        The number of votes each voter can cast in their district.

    Returns
    -------
    np.ndarray recommendations
        A 1D array containing the number of recommendations each candidate received.
    np.ndarray top_recommendations
        A 1D array containing the number of top recommendations each candidate received.
    """
    n_voters, n_candidates = voter_candidate_distances.shape
    recommendations, top_recommendations = np.zeros(n_candidates), np.zeros(
        n_candidates
    )

    sorted_indices = np.argsort(voter_candidate_distances, axis=1)
    sorted_distances = np.take_along_axis(
        voter_candidate_distances, sorted_indices, axis=1
    )

    # distances of voters to candidates at the threshold
    threshold_distances = sorted_distances[:, n_votes_in_district - 1]
    # distances of voters to their top recommendations
    top_distances = sorted_distances[:, 0]
    # boolean array indicating whether candidates are below the threshold distance for each voter
    below_threshold = sorted_distances < threshold_distances.reshape(-1, 1)
    # boolean array indicating whether candidates are at the threshold distance for each voter
    at_threshold = sorted_distances == threshold_distances.reshape(-1, 1)
    # boolean array indicating whether candidates are top recommendations
    at_top = sorted_distances == top_distances.reshape(-1, 1)
    # candidates below the threshold distance get a full recommendation
    below_threshold_indices = sorted_indices[below_threshold]
    indices, counts = np.unique(below_threshold_indices, return_counts=True)
    recommendations[indices] = counts
    # number of recommendations left to distribute by each voter
    remaining_recommendations = n_votes_in_district - below_threshold.sum(axis=1)

    n_candidates_at_threshold = at_threshold.sum(axis=1)

    for (
        candidate_indices,
        threshold_candidates,
        top_candidates,
        split_recommendations,
    ) in zip(
        sorted_indices,
        at_threshold,
        at_top,
        (remaining_recommendations / n_candidates_at_threshold),
    ):
        recommendations[
            candidate_indices[threshold_candidates]
        ] += split_recommendations
        top_recommendations[candidate_indices[top_candidates]] += (
            1 / top_candidates.sum()
        )

    return recommendations, top_recommendations
