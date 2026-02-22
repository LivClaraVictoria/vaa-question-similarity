import hashlib
import inspect
import json
import os
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.stats
from tqdm import tqdm

from .constants import (
    CACHE_FOLDER,
    DATA_FOLDER,
    SV23_FOLDER,
    VOTERS_FILE,
    ID2PREF_PARTY,
    ID2EDUCATION,
    ID2DISTRICT,
    ID2LANGUAGE,
    CANDIDATES_FILE,
    ID2PARTY_REC,
    QUESTIONS_FILE,
    QUESTION_ID2CATEGORY,
    SV19_FOLDER,
    VOTERS19_FILE,
    ID2PARTY19,
    ID2DISTRICT19,
    ID2GENDER19,
    ID2LANGUAGE19,
    CANDIDATES19_FILE,
    QUESTIONS19_FILE,
    PARTY_SHORT2PARTY19,
    PARTY_SHORT2PARTY,
    PARTY_SHORT2PARTY_REC19,
    ANSWER_POSSIBILITIES,
    DISTANCE_METHODS,
    SEATS_PER_CANTON,
    SEATS_PER_CANTON19,
    PARTIES_LEFT_TO_RIGHT,
    FILTERING_METHODS,
    TIMESTAMP_FILE,
    KANTONBEZEICHNUNG2DISTRICT,
)
from .utils import (
    count_consecutive_values,
    ensure_path,
    get_cols,
    remove_old_cache_files,
    gini_coefficient,
    find_first_occurrence,
)


class SVDataFrame(pd.DataFrame):
    """
    A custom DataFrame class for handling Smartvote voter and candidate datasets.

    SVDataFrame extends the functionality of pandas.DataFrame by adding specialized methods and attributes
    to work with Smartvote voter and candidate data. It includes tools for processing and analyzing voting
    recommendations, candidate rankings, and other related metrics based on voter preferences.

    Attributes
    ----------
    answer_cols : list of str
        Columns in the DataFrame representing answers to survey questions.

    weight_cols : list of str
        Columns in the DataFrame representing the weights (importance) assigned to the answers.

    cleavage_cols : list of str
        Columns in the DataFrame representing cleavage (ideological) positions.

    term : int
        The election term associated with the data (e.g., 2019 or 2023).

    Methods
    -------
    from_dataframe(cls, df, term: int = None, verbose=False)
        Creates an SVDataFrame from an existing pandas DataFrame, inferring or setting the appropriate metadata.

    answers()
        Returns a DataFrame containing only the answer columns.

    weights()
        Returns a DataFrame containing only the weight columns.

    cleavages()
        Returns a DataFrame containing only the cleavage columns.

    district(district: str)
        Filters the DataFrame for a specific district.

    get_distance_methods()
        Retrieves all distance methods used in the DataFrame.

    save_candidate_voting_recommendations(savefile: str)
        Saves candidate voting recommendation data to a file in the cache folder.

    load_candidate_voting_recommendations(savefile: str, selected_distance_methods: list[str] = None)
        Loads candidate voting recommendations from a cached file and merges them into the DataFrame.

    save_list_voting_recommendations(savefile: str)
        Saves list voting recommendation data to a file in the cache folder.

    load_list_voting_recommendations(savefile: str, selected_distance_methods: list[str] = None)
        Loads list voting recommendations from a cached file and merges them into the DataFrame.

    save_candidate_recommendation_counts(savefile: str)
        Saves candidate recommendation counts to a file in the cache folder.

    load_candidate_recommendation_counts(savefile: str, selected_distance_methods: list[str] = None)
        Loads candidate recommendation counts from a cached file and merges them into the DataFrame.

    add_recommendation_counts(df_voters, normalized: bool = True, verbose: bool = False)
        Adds recommendation counts to the DataFrame based on voter preferences from another DataFrame.

    normalize_recommendation_counts()
        Normalizes existing recommendation counts across districts.

    add_disagreement_counts(df_candidates, progress_bar: bool = False)
        Calculates and adds disagreement counts between voters and recommended candidates.

    get_party_visibilities(distance_methods: list[str] = None, party_col: str = '_party',
                        add_vnormalized_cols: bool = False, fair_share: bool = False)
        Calculates party visibilities based on candidate voting recommendations.

    get_recommendation_metrics(include_zh: bool = True, verbose: bool = False)
        Computes metrics related to candidate recommendation counts.

    map_lists_to_parties(df_candidates, progress_bar: bool = False)
        Maps lists to corresponding parties based on parties of candidates in the list.

    add_party_ranks(df_candidates, frac: bool = True, progress_bar: bool = False)
        Adds ranks of first list corresponding to preferred party based on list voting recommendations.

    get_party_rank_metrics(include_zh: bool = False, verbose: bool = False, progress_bar: bool = False)
        Computes metrics related to party ranks.

    get_voter_metrics(df_candidates: pd.DataFrame = None, include_zh: bool = False, verbose: bool = False,
                      progress_bar: bool = False)
        Calculates metrics related to df_voters, including party ranks and disagreement counts.

    get_party_popularities() -> pd.DataFrame
        Calculates the popularity of parties based on list voting recommendations.

    get_party_popularity_diffs(party_popularities: pd.DataFrame = None, relative_changes: bool = False,
                               include_pref: bool = False) -> pd.DataFrame
        Computes differences in party popularity across different distance methods.

    add_n_votes(self, json_path: str = os.path.join(DATA_FOLDER, 'NRW2023-kandidierende.json'))
        Adds the number of votes each candidate got in the 2023 National Council elections.

    get_compacted(copy: bool = True)
        Returns a compacted version of the DataFrame, simplifying answer possibilities.
    """

    # additional attributes
    _metadata = ["answer_cols", "weight_cols", "cleavage_cols", "term"]

    @property
    def _constructor(self):
        return SVDataFrame

    def __init__(
        self,
        *args,
        answer_cols: list[str] = None,
        weight_cols: list[str] = None,
        cleavage_cols: list[str] = None,
        term: int = None,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.answer_cols = (
            answer_cols
            if answer_cols is not None
            else get_cols(self, col_type="answer", verbose=verbose)
        )
        self.weight_cols = (
            weight_cols
            if weight_cols is not None
            else get_cols(self, col_type="weight", verbose=verbose)
        )
        self.cleavage_cols = (
            cleavage_cols
            if cleavage_cols is not None
            else get_cols(self, col_type="cleavage", verbose=verbose)
        )
        if term is None:
            # if there is a column ID_election and the first row contains the value 222 the term is 2019 else 2023
            self.term = (
                2019
                if (
                    "ID_election" in self.columns and self["ID_election"].iloc[0] == 222
                )
                else 2023
            )
            if verbose:
                warnings.warn(f"Term not specified. Inferred to {self.term}.")
        else:
            self.term = term

    @classmethod
    def from_dataframe(cls, df, term: int = None, verbose=False):
        answer_cols = get_cols(df, col_type="answer", verbose=verbose)
        weight_cols = get_cols(df, col_type="weight", verbose=verbose)
        cleavage_cols = get_cols(df, col_type="cleavage", verbose=verbose)
        return cls(
            df,
            answer_cols=answer_cols,
            weight_cols=weight_cols,
            cleavage_cols=cleavage_cols,
            term=term,
            verbose=verbose,
        )

    def answers(self):
        return self[self.answer_cols]

    def weights(self):
        return self[self.weight_cols]

    def cleavages(self):
        return self[self.cleavage_cols]

    def a(self):
        return self.answers()

    def w(self):
        return self.weights()

    def c(self):
        return self.cleavages()

    def district(self, district: str):
        return self[self["_district"] == district]

    def get_distance_methods(self):
        """
        Retrieves all distance methods used in the SVDataFrame.

        Returns
        -------
        list of str
            A list of distance methods found in the DataFrame's columns.
        """

        general_cols = [
            method
            for method in DISTANCE_METHODS
            if any(
                [
                    c.endswith(method) or c.endswith(f"{method}_normalized")
                    for c in self.columns
                    if not c.startswith("_maxDist")
                ]
            )
        ]
        filtering_cols = [
            f"{method}_{filtering_method}"
            for method in DISTANCE_METHODS
            for filtering_method in FILTERING_METHODS
            if any([c.endswith(f"{method}_{filtering_method}") for c in self.columns])
        ]

        return general_cols + filtering_cols

    def save_candidate_voting_recommendations(self, savefile: str) -> None:
        """
        Saves the candidate voting recommendation data to a file in the cache folder.

        Parameters
        ----------
        savefile : str
            The filename for saving the voting recommendation data. The method ensures the file is saved
            with a `.parquet` extension in the cache folder.

        Returns
        -------
        None
            This method does not return anything. It saves the specified data to a file.
        """

        cols_to_save = ["recID" if self.term == 2023 else "ID_recommendation"] + [
            c for c in self.columns if c.startswith("_match")
        ]
        self[cols_to_save].to_parquet(
            os.path.join(CACHE_FOLDER, savefile.removesuffix(".parquet") + ".parquet")
        )

    def load_candidate_voting_recommendations(
        self, savefile: str, selected_distance_methods: list[str] = None
    ) -> pd.DataFrame:
        """
        Loads candidate voting recommendations from a cached file and merges them into the current DataFrame.

        Parameters
        ----------
        savefile : str
            The filename of the saved voting recommendations to load from the cache folder.

        selected_distance_methods : list of str, optional
            A list of distance methods to filter and load. If not provided, all available methods are loaded.

        Returns
        -------
        pd.DataFrame
            The updated DataFrame with loaded voting recommendations merged by the appropriate ID column
            ('recID' for 2023 or 'ID_recommendation' for 2019).
        """

        df_recommendations = pd.read_parquet(
            os.path.join(CACHE_FOLDER, savefile.removesuffix(".parquet") + ".parquet")
        )

        # only load selected distance methods
        if selected_distance_methods:
            df_recommendations = df_recommendations[
                ["recID" if self.term == 2023 else "ID_recommendation"]
                + [
                    c
                    for c in df_recommendations.columns
                    if any([c.endswith(m) for m in selected_distance_methods])
                ]
            ]

        return self.drop(
            columns=[c for c in self.columns if c.startswith("_match")]
        ).merge(
            df_recommendations, on="recID" if self.term == 2023 else "ID_recommendation"
        )

    def save_list_voting_recommendations(self, savefile: str) -> None:
        """
        Saves the list voting recommendation data to a file in the cache folder.

        Parameters
        ----------
        savefile : str
            The filename for saving the list voting recommendation data. The method ensures the file is saved
            with a `.parquet` extension in the cache folder.

        Returns
        -------
        None
            This method does not return anything. It saves the specified data to a file.
        """

        cols_to_save = ["recID" if self.term == 2023 else "ID_recommendation"] + [
            c for c in self.columns if c.startswith("_List")
        ]
        self[cols_to_save].to_parquet(
            os.path.join(CACHE_FOLDER, savefile.removesuffix(".parquet") + ".parquet")
        )

    def load_list_voting_recommendations(
        self, savefile: str, selected_distance_methods: list[str] = None
    ) -> pd.DataFrame:
        """
        Loads list voting recommendations from a cached file and merges them into the current DataFrame.

        Parameters
        ----------
        savefile : str
            The filename of the saved list voting recommendations to load from the cache folder.

        selected_distance_methods : list of str, optional
            A list of distance methods to filter and load. If not provided, all available methods are loaded.

        Returns
        -------
        pd.DataFrame
            The updated DataFrame with loaded list voting recommendations merged by the appropriate ID column
            ('recID' for 2023 or 'ID_recommendation' for 2019).
        """

        df_recommendations = pd.read_parquet(
            os.path.join(CACHE_FOLDER, savefile.removesuffix(".parquet") + ".parquet")
        )

        # only load selected distance methods
        if selected_distance_methods:
            df_recommendations = df_recommendations[
                ["recID" if self.term == 2023 else "ID_recommendation"]
                + [
                    c
                    for c in df_recommendations.columns
                    if any([c.endswith(m) for m in selected_distance_methods])
                ]
            ]

        return self.drop(
            columns=[c for c in self.columns if c.startswith("_List")]
        ).merge(
            df_recommendations, on="recID" if self.term == 2023 else "ID_recommendation"
        )

    def save_candidate_recommendation_counts(self, savefile: str) -> None:
        """
        This method extracts the candidate recommendation count columns from the DataFrame and saves them to a
        Parquet file in the specified cache folder.

        Parameters
        ----------
        savefile : str
            The filename for saving the recommendation counts. The method ensures the file is saved with a
            `.parquet` extension in the cache folder.

        Returns
        -------
        None
            This method does not return anything. It saves the specified data to a file.
        """

        cols_to_save = ["ID_candidate"] + [
            c for c in self.columns if c.startswith("_n_recommendations")
        ]
        self[cols_to_save].to_parquet(
            os.path.join(CACHE_FOLDER, savefile.removesuffix(".parquet") + ".parquet")
        )

    def load_candidate_recommendation_counts(
        self, savefile: str, selected_distance_methods: list[str] = None
    ) -> pd.DataFrame:
        """
        Loads candidate recommendation counts from a cached file and merges them into the current DataFrame.

        Parameters
        ----------
        savefile : str
            The filename of the saved recommendation counts to load from the cache folder.

        selected_distance_methods : list of str, optional
            A list of distance methods to filter and load. If not provided, all available methods are loaded.

        Returns
        -------
        pd.DataFrame
            The updated DataFrame with loaded recommendation counts merged by candidate ID.
        """

        df_recommendations = pd.read_parquet(
            os.path.join(CACHE_FOLDER, savefile.removesuffix(".parquet") + ".parquet")
        )

        # only load selected distance methods
        if selected_distance_methods:
            df_recommendations = df_recommendations[
                ["ID_candidate"]
                + [
                    c
                    for c in df_recommendations.columns
                    if any(
                        [
                            c.endswith(m) or c.endswith(f"{m}_normalized")
                            for m in selected_distance_methods
                        ]
                    )
                ]
            ]

        return self.drop(
            columns=[c for c in self.columns if c.startswith("_n_recommendations")]
        ).merge(df_recommendations, on="ID_candidate")

    def add_recommendation_counts(
        self, df_voters, normalized: bool = True, verbose: bool = False
    ) -> pd.DataFrame:
        """
        Adds recommendation counts to the candidate DataFrame based on candidate voting recommendations in the
        voter DataFrame.

        This method calculates the following types of recommendation counts:

        1. _n_recommendations_spc_<method> : How often a candidate is in the top-seats recommendations of any voter
        based on the number of seats in the canton.
        2. _n_recommendations_top_<method> : How often a candidate is in the top-1 recommendation of any voter.
        3. _n_recommendations_vis_<method> : In what fraction of recommendations a candidate is in the top-seats
        recommendations (visible).

        The method can also normalize these counts across districts if specified.

        Parameters
        ----------
        df_voters : pandas.DataFrame
            DataFrame containing voter preferences and their corresponding candidate matchings.
            The DataFrame is expected to have specific columns that represent candidate matches
            (e.g., `_matchID_<n>_<method>`), where `<n>` is the rank of the match and `<method>` is the distance method used.

        normalized : bool, optional
            If True, the recommendation counts are normalized by the number of voters and candidates in each district,
            adjusting for the number of seats available. Default is True.

        verbose : bool, optional
            If True, prints additional information about the distance methods used during the calculation. Default is False.

        Returns
        -------
        pd.DataFrame
            This method modifies the DataFrame in place and returns it.

        Raises
        ------
        AssertionError
            If `self.term` is not set to either 2019 or 2023.

        Notes
        -----
        - The method drops any existing recommendation count columns before recalculating them.
        - The method uses predefined district seat information for the years 2019 and 2023 to perform normalization.
        - The following new columns are added to the DataFrame, where `<method>` represents each distance method:
          - `_n_recommendations_spc_<method>`: Number of top-seats (seats-per-canton) recommendations of a candidate.
          - `_n_recommendations_top_<method>`: Number of top recommendations of a candidate.
          - `_n_recommendations_vis_<method>`: Visibility of a candidate.
          - `_n_recommendations_spc_<method>_normalized`: Expectation-normalized seats-per-canton recommendations.
          - `_n_recommendations_top_<method>_normalized`: Expectation-normalized top recommendations.
        """

        assert (
            self.term == 2023 or self.term == 2019
        ), "Term not specified. Need to specify term in DataFrame."
        # drop all recommendation count columns
        self.drop(
            columns=[c for c in self.columns if c.startswith("_n_recommendations")],
            inplace=True,
        )
        distance_methods = df_voters.get_distance_methods()
        n_match_cols = max(
            [
                int(c[len("_matchID_") : -len(f"_{distance_methods[0]}")])
                for c in df_voters.columns
                if c.startswith("_matchID_") and c.endswith(f"_{distance_methods[0]}")
            ]
        )
        distance_methods = df_voters.get_distance_methods()
        if verbose:
            print("Distance methods used for recommendations:")
            print(distance_methods)

        for method in distance_methods:
            map_candidate_id_to_spc_recs = {
                value: count
                for value, count in zip(
                    *np.unique(
                        df_voters[
                            [
                                f"_matchID_{i}_{method}"
                                for i in range(1, n_match_cols + 1)
                            ]
                        ].values,
                        return_counts=True,
                    )
                )
            }
            # add recommendation counts (seats per canton)
            self[f"_n_recommendations_spc_{method}"] = (
                self["ID_candidate"].map(map_candidate_id_to_spc_recs).fillna(0)
            )

            map_candidate_id_to_top_recs = {
                value: count
                for value, count in zip(
                    *np.unique(
                        df_voters[f"_matchID_1_{method}"].values, return_counts=True
                    )
                )
            }
            # add top recommendation counts
            self[f"_n_recommendations_top_{method}"] = (
                self["ID_candidate"].map(map_candidate_id_to_top_recs).fillna(0)
            )

            # add voter-normalized recommendation counts (visibility in %)
            n_voters_per_district = df_voters["_district"].value_counts()
            self[f"_n_recommendations_vis_{method}"] = (
                100
                * self[f"_n_recommendations_spc_{method}"]
                / self["_district"].map(n_voters_per_district)
            )

        if normalized:
            # expectation-normalization (value of 1 corresponds to each candidate being recommended equally often)
            for district in self["_district"].unique():
                candidate_district_mask = self["_district"] == district
                n_district_voters = (df_voters["_district"] == district).sum()
                n_district_candidates = candidate_district_mask.sum()
                n_district_seats = (
                    SEATS_PER_CANTON[district]
                    if self.term == 2023
                    else SEATS_PER_CANTON19[district]
                )
                n_district_votes = n_district_voters * n_district_seats

                for method in distance_methods:
                    self.loc[
                        candidate_district_mask,
                        f"_n_recommendations_spc_{method}_normalized",
                    ] = self.loc[
                        candidate_district_mask, f"_n_recommendations_spc_{method}"
                    ] / (
                        n_district_votes / n_district_candidates
                    )
                    self.loc[
                        candidate_district_mask,
                        f"_n_recommendations_top_{method}_normalized",
                    ] = self.loc[
                        candidate_district_mask, f"_n_recommendations_top_{method}"
                    ] / (
                        n_district_voters / n_district_candidates
                    )

        return self

    def normalize_recommendation_counts(
        self,
    ):

        assert (
            self.term == 2023 or self.term == 2019
        ), "Term not specified. Need to specify term in DataFrame."
        # Drop all recommendation count columns that are already normalized
        self.drop(
            columns=[
                c
                for c in self.columns
                if c.startswith("_n_recommendations") and c.endswith("_normalized")
            ],
            inplace=True,
        )
        distance_methods = self.get_distance_methods()

        spc_col = [
            c
            for c in self.columns
            if c.startswith("_n_recommendations")
            and c.endswith(f"spc_{distance_methods[0]}")
        ][0]

        for district in self["_district"].unique():
            candidate_district_mask = self["_district"] == district
            n_district_seats = (
                SEATS_PER_CANTON[district]
                if self.term == 2023
                else SEATS_PER_CANTON19[district]
            )
            n_district_votes = self.loc[candidate_district_mask, spc_col].sum()
            n_district_voters = n_district_votes / n_district_seats
            n_district_candidates = candidate_district_mask.sum()

            for rec_col in [
                c
                for c in self.columns
                if c.startswith("_n_recommendations") and not c.endswith("_normalized")
            ]:
                if "_spc_" in rec_col:
                    self.loc[candidate_district_mask, rec_col + "_normalized"] = (
                        self.loc[candidate_district_mask, rec_col]
                        / (n_district_votes / n_district_candidates)
                    )
                elif "_top_" in rec_col:
                    self.loc[candidate_district_mask, rec_col + "_normalized"] = (
                        self.loc[candidate_district_mask, rec_col]
                        / (n_district_voters / n_district_candidates)
                    )

        return self

    def add_disagreement_counts(
        self, df_candidates, progress_bar: bool = False
    ) -> None:
        """
        This method computes several metrics that quantify the disagreements between voters and recommended candidates.
        Disagreement counts are calculated both for all answers and only for strongly weighted answers,
        and they are normalized by the number of answers provided by the voter.
        The results are added as new columns to the DataFrame.

        Parameters
        ----------
        df_candidates : pd.DataFrame
            DataFrame containing candidate data, including their responses to survey questions.
            This is used to compare with voter responses to identify disagreements.

        progress_bar : bool, optional
            If True, displays a progress bar during the calculation. Default is False.

        Returns
        -------
        None
            This method modifies the DataFrame in place by adding new columns for the disagreement counts.

        Notes
        -----
        - Disagreements are calculated based on the difference in answers between voters and candidates.
          A disagreement occurs if the voter and candidate have opposing views (one above 50 and the other
          below 50 on a given question).
        - Strong disagreements are identified where the voter assigned a strong weight to their answer (weight = 2).
        - Disagreement counts are calculated for both the top candidate recommendation and general seats-per-canton
          recommendations, the fractional metrics are normalized by the total number of answers or strong weights.

        - New columns added to the DataFrame include:
          - `_disagreements_top_<method>`
          - `_disagreements_strong_top_<method>`
          - `_disagreements_spc_<method>`
          - `_disagreements_strong_spc_<method>`
          - `_disagreements_top_frac_<method>`
          - `_disagreements_strong_top_frac_<method>`
          - `_disagreements_spc_frac_<method>`
          - `_disagreements_strong_spc_frac_<method>`
        """

        method_iter = (
            tqdm(self.get_distance_methods())
            if progress_bar
            else self.get_distance_methods()
        )
        for method in method_iter:
            district_iter = (
                tqdm(self["_district"].unique())
                if progress_bar
                else self["_district"].unique()
            )
            for district in district_iter:
                voter_district_mask = self["_district"] == district
                df_voters_district = self[voter_district_mask]

                n_strong_weights = (df_voters_district.w() == 2).sum(axis=1)
                n_answers = (~df_voters_district.a().isna()).sum(axis=1)
                n_seats = (
                    SEATS_PER_CANTON[district]
                    if self.term == 2023
                    else SEATS_PER_CANTON19[district]
                )

                disagreements, strong_disagreements = [], []
                for i in range(1, n_seats + 1):
                    v_c_answers = df_voters_district[
                        [f"_matchID_{i}_{method}"]
                        + get_cols(self)
                        + get_cols(self, col_type="weight")
                    ].merge(
                        df_candidates[["ID_candidate"] + get_cols(df_candidates)],
                        how="left",
                        left_on=f"_matchID_{i}_{method}",
                        right_on="ID_candidate",
                        suffixes=["", "_candidate"],
                    )
                    disagreements.append(
                        pd.concat(
                            [
                                (v_c_answers[col] - 50)
                                * (v_c_answers[f"{col}_candidate"] - 50)
                                < 0
                                for col in get_cols(v_c_answers)
                            ],
                            axis=1,
                        )
                        .sum(axis=1)
                        .to_numpy()
                    )
                    strong_disagreements.append(
                        pd.concat(
                            [
                                (
                                    (v_c_answers[col] - 50)
                                    * (v_c_answers[f"{col}_candidate"] - 50)
                                    < 0
                                )
                                & (v_c_answers[col.replace("answer", "weight")] == 2)
                                for col in get_cols(v_c_answers)
                            ],
                            axis=1,
                        )
                        .sum(axis=1)
                        .to_numpy()
                    )

                strong_disagreements = np.stack(strong_disagreements).astype(float)
                # set strong disagreements to nan if voter didn't use strong weights
                strong_disagreements[:, n_strong_weights == 0] = np.nan

                self.loc[voter_district_mask, f"_disagreements_top_{method}"] = (
                    disagreements[0]
                )
                self.loc[voter_district_mask, f"_disagreements_strong_top_{method}"] = (
                    strong_disagreements[0]
                )
                self.loc[voter_district_mask, f"_disagreements_spc_{method}"] = (
                    np.sum(disagreements, axis=0) / n_seats
                )
                self.loc[voter_district_mask, f"_disagreements_strong_spc_{method}"] = (
                    np.sum(strong_disagreements, axis=0) / n_seats
                )

                self.loc[voter_district_mask, f"_disagreements_top_frac_{method}"] = (
                    self.loc[voter_district_mask, f"_disagreements_top_{method}"]
                    / n_answers
                )
                self.loc[
                    voter_district_mask, f"_disagreements_strong_top_frac_{method}"
                ] = (
                    self.loc[voter_district_mask, f"_disagreements_strong_top_{method}"]
                    / n_strong_weights
                )
                self.loc[voter_district_mask, f"_disagreements_spc_frac_{method}"] = (
                    self.loc[voter_district_mask, f"_disagreements_spc_{method}"]
                    / n_answers
                )
                self.loc[
                    voter_district_mask, f"_disagreements_strong_spc_frac_{method}"
                ] = (
                    self.loc[voter_district_mask, f"_disagreements_strong_spc_{method}"]
                    / n_strong_weights
                )

    def get_party_visibilities(
        self,
        distance_methods: list[str] = None,
        party_col: str = "_party",
        add_vnormalized_cols: bool = False,
        fair_share: bool = False,
    ) -> pd.DataFrame:
        """
        This function computes the party visibilities in the same way as the Swiss government does after elections.
        Each voter has one vote that is split proportionally over the parties corresponding to the candidates
        they were recommended if the district they reside in has multiple seats.

        Parameters
        ----------
        self : object
            The instance of the class containing the data and methods.
        distance_methods : list of str, optional
            A list of distance methods to be used for calculating party visibilities.
            If not provided, it defaults to using all distance methods present.
        party_col : str, optional
            The column name representing the party in the dataframe. Default is '_party'.
        add_vnormalized_cols: bool, optional
            If True, the function retains the voter-normalized recommendation count columns in the dataframe.
            Default is False, which drops these columns after calculation.
        fair_share: bool, optional
            If True, the function calculates the party visibilities based on the fair distribution of votes.

        Returns
        -------
        pd.DataFrame
            A dataframe containing the party visibilities for each distance method and based on actual election votes if available.
        """

        distance_methods = (
            self.get_distance_methods()
            if distance_methods is None
            else distance_methods
        )
        n_voters = self[
            f'_n_recommendations_{"fair_" if fair_share else ""}top_{distance_methods[0]}'
        ].sum()
        n_recs = self["_district"].map(
            SEATS_PER_CANTON if self.term == 2023 else SEATS_PER_CANTON19
        )

        # Add voter normalized recommendation counts (so that each voter has 1 vote/recommendation)
        for method in distance_methods:
            self[f"_n_recommendations_spc_{method}_vnormalized"] = (
                self[f'_n_recommendations_{"fair_" if fair_share else ""}spc_{method}']
                / n_recs
            )

        # Calculate party visibilities based on recommendations
        df_party_visibilities = (
            self.groupby(party_col).agg(
                {
                    f"_n_recommendations_spc_{method}_vnormalized": "sum"
                    for method in distance_methods
                }
            )
            / n_voters
        )

        # Rename columns
        df_party_visibilities = df_party_visibilities.rename(
            columns={
                f"_n_recommendations_spc_{method}_vnormalized": f"Party Visibility ({method})"
                for method in distance_methods
            }
        )

        # If actual election votes are available, calculate party visibility based on them
        if "_n_votes" in self.columns:
            # Calculate vnormalized votes for each district
            self["_n_votes_vnormalized"] = self["_n_votes"] / self["_district"].map(
                SEATS_PER_CANTON if self.term == 2023 else SEATS_PER_CANTON19
            )
            # Calculate party visibility based on actual votes
            df_actual_votes_visibilities = (
                self.groupby(party_col)["_n_votes_vnormalized"].sum()
                / self["_n_votes_vnormalized"].sum()
            )
            df_actual_votes_visibilities = df_actual_votes_visibilities.rename(
                "Party Visibility (Actual Votes)"
            )
            # Combine the recommendation-based and actual vote-based party visibilities
            df_party_visibilities = df_party_visibilities.join(
                df_actual_votes_visibilities, how="outer"
            )

        # Drop voter normalized recommendation count columns if not needed
        if not add_vnormalized_cols:
            self.drop(
                columns=[
                    f"_n_recommendations_spc_{method}_vnormalized"
                    for method in distance_methods
                ],
                inplace=True,
            )

        return df_party_visibilities

    def get_recommendation_metrics(
        self, include_zh: bool = True, verbose: bool = False
    ):
        """
        This method computes several metrics related to candidate recommendation counts, including weighted mean differences,
        Gini coefficients, and Pearson correlations. Metrics are calculated for both general and top recommendations, and
        optionally for a specific district ('ZH').

        Parameters
        ----------
        include_zh : bool, optional
            If True, metrics are also calculated for the 'ZH' district. Default is True.
        verbose : bool, optional
            If True, prints the distance methods used for metrics. Default is False.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the calculated metrics for each distance method.

        Raises
        ------
        AssertionError
            If no recommendation counts are found in the DataFrame.

        Notes
        -----
        - The method requires that recommendation counts have been added to the DataFrame using `add_recommendation_counts`.
        - Metrics calculated include:
          - `weighted_mean_diff_spc`: Weighted mean difference for seats-per-canton recommendations.
          - `gini_spc`: Gini coefficient for seats-per-canton recommendations.
          - `as_rec_spc_corr`: Pearson correlation between answer strengths and seats-per-canton recommendations.
          - `weighted_mean_diff_top`: Weighted mean difference for top recommendations.
          - `gini_top`: Gini coefficient for top recommendations.
          - `as_rec_top_corr`: Pearson correlation between answer strengths and top recommendations.
          - If `include_zh` is True, the same metrics are calculated for the 'ZH' district.
        """

        assert any(
            [c for c in self.columns if c.startswith("_n_recommendations_spc_")]
        ), "No recommendation counts found in DataFrame. Need to run add_recommendation_counts first."

        metric_dict = defaultdict(list)

        answer_strengths = self["_answer_strength"].values
        answer_strengths_zh = self[self["_district"] == "ZH"]["_answer_strength"].values
        normal_std = np.std(answer_strengths)
        normal_mean = np.mean(answer_strengths)

        normal_std_zh = np.std(answer_strengths_zh)
        normal_mean_zh = np.mean(answer_strengths_zh)

        distance_methods = self.get_distance_methods()
        if verbose:
            print("Distance methods used for metrics:")
            print(distance_methods)

        for method in distance_methods:
            metric_dict["method"].append(method)

            n_recs_spc = self[f"_n_recommendations_spc_{method}_normalized"].values
            metric_dict["weighted_mean_diff_spc"].append(
                np.average(answer_strengths, weights=n_recs_spc) - normal_mean
            )
            # metric_dict['weighted_std_factor_spc'].append(np.sqrt(
            #     np.average((answer_strengths - np.average(answer_strengths)) ** 2, weights=n_recs_spc)) / normal_std)
            metric_dict["gini_spc"].append(gini_coefficient(n_recs_spc))
            metric_dict["as_rec_spc_corr"].append(
                scipy.stats.pearsonr(answer_strengths, n_recs_spc)[0]
            )

            n_recs_top = self[f"_n_recommendations_top_{method}_normalized"].values
            metric_dict["weighted_mean_diff_top"].append(
                np.average(answer_strengths, weights=n_recs_top) - normal_mean
            )
            # metric_dict['weighted_std_factor_top'].append(np.sqrt(
            #     np.average((answer_strengths - np.average(answer_strengths)) ** 2, weights=n_recs_top)) / normal_std)
            metric_dict["gini_top"].append(gini_coefficient(n_recs_top))
            metric_dict["as_rec_top_corr"].append(
                scipy.stats.pearsonr(answer_strengths, n_recs_top)[0]
            )

            if include_zh:
                n_recs_spc_zh = self[self["_district"] == "ZH"][
                    f"_n_recommendations_spc_{method}_normalized"
                ].values
                metric_dict["weighted_mean_diff_spc_zh"].append(
                    np.average(answer_strengths_zh, weights=n_recs_spc_zh)
                    - normal_mean_zh
                )
                # metric_dict['weighted_std_factor_spc_zh'].append(np.sqrt(
                #     np.average((answer_strengths_zh - np.average(answer_strengths_zh)) ** 2,
                #                weights=n_recs_spc_zh)) / normal_std_zh)
                metric_dict["gini_spc_zh"].append(gini_coefficient(n_recs_spc_zh))
                metric_dict["as_rec_spc_corr_zh"].append(
                    scipy.stats.pearsonr(answer_strengths_zh, n_recs_spc_zh)[0]
                )

                n_recs_top_zh = self[self["_district"] == "ZH"][
                    f"_n_recommendations_top_{method}_normalized"
                ].values
                metric_dict["weighted_mean_diff_top_zh"].append(
                    np.average(answer_strengths_zh, weights=n_recs_top_zh)
                    - normal_mean_zh
                )
                # metric_dict['weighted_std_factor_top_zh'].append(np.sqrt(
                #     np.average((answer_strengths_zh - np.average(answer_strengths_zh)) ** 2,
                #                weights=n_recs_top_zh)) / normal_std_zh)
                metric_dict["gini_top_zh"].append(gini_coefficient(n_recs_top_zh))
                metric_dict["as_rec_top_corr_zh"].append(
                    scipy.stats.pearsonr(answer_strengths_zh, n_recs_top_zh)[0]
                )

        return pd.DataFrame(metric_dict)

    def map_lists_to_parties(self, df_candidates, progress_bar: bool = False):
        """
        This method maps the ListID resulting from the list recommendations to ListParty by determining the corresponding
        party for each list. The party for each list is determined based on the parties of the candidates on the list.

        The party is determined as follows:
        - If there is a single most frequent party among the candidates on the list (excluding 'Parteilos'), that party is chosen.
        - If there are multiple parties with the same highest frequency, the party that appears first in the descending order
          of overall candidate counts is chosen.
        - If all candidates on the list are 'Parteilos', the list is assigned 'Parteilos'.

        Parameters
        ----------
        df_candidates : pd.DataFrame
            DataFrame containing candidate data, including their party affiliations and list IDs.
        progress_bar : bool, optional
            If True, displays a progress bar during the mapping process. Default is False.

        Returns
        -------
        None
            This method modifies the DataFrame in place by adding new columns for the mapped ListParty.

        Raises
        ------
        ValueError
            If no party lists are found in the DataFrame.

        Notes
        -----
        - The method requires that list voting recommendations have been added to the DataFrame using `add_list_voting_recommendations`.
        - New columns added to the DataFrame include:
          - `_ListParty_<method>`: The party corresponding to each ListID for each distance method.
        """

        if not any([c.startswith("_ListID_") for c in self.columns]):
            raise ValueError(
                "No party lists found in DataFrame. Need to run add_list_voting_recommendations first."
            )

        parties_descending_popularity = list(
            df_candidates.groupby("_party")["ID_candidate"]
            .count()
            .sort_values(ascending=False)
            .index
        )

        list2party = (
            df_candidates.groupby("ID_list")["_party"]
            .apply(
                lambda x: (
                    x[~x.isin(["Parteilos"])].mode()[0]
                    if len(x[~x.isin(["Parteilos"])].mode()) == 1
                    else (
                        sorted(
                            x[~x.isin(["Parteilos"])].mode(),
                            key=parties_descending_popularity.index,
                        )[0]
                        if len(x[~x.isin(["Parteilos"])].mode()) > 1
                        else "Parteilos"
                    )
                )
            )
            .to_dict()
        )

        list_id_columns = [c for c in self.columns if "ListID" in c]
        iterator = tqdm(list_id_columns) if progress_bar else list_id_columns

        with warnings.catch_warnings():
            warnings.simplefilter(
                action="ignore", category=pd.errors.PerformanceWarning
            )
            for col in iterator:
                new_col_name = col.replace("ListID", "ListParty")
                self[new_col_name] = self[col].map(list2party.get)

    def add_party_ranks(
        self, df_candidates, frac: bool = True, progress_bar: bool = False
    ):
        """
        This method calculates the ranks of the first list corresponding to a voter's preferred party based on list voting
        recommendations. If the mapping from list IDs to parties is not found, it will first run `map_lists_to_parties`.

        Parameters
        ----------
        df_candidates : pd.DataFrame
            DataFrame containing candidate data, including their party affiliations and list IDs.
        frac : bool, optional
            If True, also calculates the fractional rank of the list. Default is True.
        progress_bar : bool, optional
            If True, displays a progress bar during the calculation. Default is False.

        Returns
        -------
        None
            This method modifies the DataFrame in place by adding new columns for the list party ranks.

        Notes
        -----
        - The method requires that list voting recommendations have been added to the DataFrame using `add_list_voting_recommendations`.
        - If the mapping from list IDs to parties is not found, the method will run `map_lists_to_parties` first.
        - New columns added to the DataFrame include:
          - `_ListPartyRank_<method>`: The rank of the first list corresponding to the voter's preferred party for each distance method.
          - `_ListPartyRankFrac_<method>`: The fractional rank of the first list corresponding to the voter's preferred party for each distance method (if `frac` is True).
        """

        if not any([c for c in self.columns if c.startswith("_ListParty_")]):
            warnings.warn(
                "No mapping from list IDs to parties found in DataFrame. Will run map_lists_to_parties first."
            )
            self.map_lists_to_parties(df_candidates, progress_bar=progress_bar)

        distance_methods = self.get_distance_methods()
        iterator = distance_methods if not progress_bar else tqdm(distance_methods)
        for method in iterator:
            voters_party_mask = self["_party"].isin(df_candidates["_party"].unique())

            n_list_rec_columns = max(
                [
                    int(c[len("_ListParty_") : -len(f"_{distance_methods[0]}")])
                    for c in self.columns
                    if c.startswith("_ListParty_")
                    and c.endswith(f"_{distance_methods[0]}")
                ]
            )
            self.loc[voters_party_mask, f"_ListPartyRank_{method}"] = (
                find_first_occurrence(
                    self[voters_party_mask][
                        [
                            f"_ListParty_{i}_{method}"
                            for i in range(1, n_list_rec_columns + 1)
                        ]
                    ].values,
                    self[voters_party_mask]["_party"].values,
                )
            )

            if frac:
                n_list_rec_columns = max(
                    [
                        int(c[len("_ListID_") : -len(f"_{method}")])
                        for c in self.columns
                        if c.startswith("_ListID_") and c.endswith(f"_{method}")
                    ]
                )
                n_list_recs = (
                    ~self[
                        [
                            f"_ListID_{i}_{method}"
                            for i in range(1, n_list_rec_columns + 1)
                        ]
                    ].isna()
                ).sum(axis=1)

                self[f"_ListPartyRankFrac_{method}"] = self[
                    f"_ListPartyRank_{method}"
                ] / (n_list_recs - 1)

    def get_party_rank_metrics(
        self,
        include_zh: bool = False,
        verbose: bool = False,
        progress_bar: bool = False,
    ):
        """
        This method computes various accuracy metrics related to list recommendations, including party match accuracy,
        mean party rank, and the percentage of lists in the top 10% and 50%. Metrics are optionally calculated for a
        specific district ('ZH').

        Parameters
        ----------
        include_zh : bool, optional
            If True, metrics are also calculated for the 'ZH' district. Default is False.
        verbose : bool, optional
            If True, prints the distance methods used for metrics. Default is False.
        progress_bar : bool, optional
            If True, displays a progress bar during the calculation. Default is False.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the calculated accuracy metrics for each distance method.

        Raises
        ------
        ValueError
            If no party ranks are found in the DataFrame.

        Notes
        -----
        - The method requires that party ranks have been added to the DataFrame using `add_party_ranks`.
        - Metrics calculated include:
          - `party_match_accuracy`: The accuracy of matching the voter's preferred party.
          - `party_rank_mean`: The mean rank of the voter's preferred party.
          - `party_rank_frac_mean`: The mean fractional rank of the voter's preferred party.
          - `party_in_top_10_perc`: The percentage of lists in the top 10%.
          - `party_in_top_50_perc`: The percentage of lists in the top 50%.
          - If `include_zh` is True, the same metrics are calculated for the 'ZH' district.
        """

        if not any([c for c in self.columns if c.startswith("_ListPartyRank_")]):
            raise ValueError(
                "No party ranks found in DataFrame. Need to run add_party_ranks first."
            )

        metric_dict = defaultdict(list)
        distance_methods = self.get_distance_methods()
        if verbose:
            print("Distance methods used for metrics:")
            print(distance_methods)

        iterator = distance_methods if not progress_bar else tqdm(distance_methods)
        for method in iterator:
            metric_dict["method"].append(method)

            # add party rank metrics
            party_ranks = self[f"_ListPartyRank_{method}"].dropna()
            party_rank_fracs = self[f"_ListPartyRankFrac_{method}"].dropna()

            metric_dict["party_match_accuracy"].append((party_ranks == 0).mean())
            metric_dict["party_rank_mean"].append(np.mean(party_ranks))
            metric_dict["party_rank_frac_mean"].append(np.mean(party_rank_fracs))
            metric_dict["party_in_top_10_perc"].append((party_rank_fracs <= 0.1).mean())
            metric_dict["party_in_top_50_perc"].append((party_rank_fracs <= 0.5).mean())

            if include_zh:
                party_ranks_zh = self[self["_district"] == "ZH"][
                    f"_ListPartyRank_{method}"
                ].dropna()
                party_rank_fracs_zh = self[self["_district"] == "ZH"][
                    f"_ListPartyRankFrac_{method}"
                ].dropna()

                metric_dict["party_match_accuracy_zh"].append(
                    (party_ranks_zh == 0).mean()
                )
                metric_dict["party_rank_mean_zh"].append(np.mean(party_ranks_zh))
                metric_dict["party_rank_frac_mean_zh"].append(
                    np.mean(party_rank_fracs_zh)
                )
                metric_dict["party_in_top_10_perc_zh"].append(
                    (party_rank_fracs_zh <= 0.1).mean()
                )
                metric_dict["party_in_top_90_perc"].append(
                    (party_rank_fracs <= 0.9).mean()
                )

        return pd.DataFrame(metric_dict)

    def get_voter_metrics(
        self,
        df_candidates: pd.DataFrame = None,
        include_zh: bool = False,
        verbose: bool = False,
        progress_bar: bool = False,
    ):
        """
        This method computes several metrics related to voters, such as party match accuracy, mean party rank, and
        disagreement counts between voters and recommended candidates. Metrics are optionally calculated for a specific
        district ('ZH').

        Parameters
        ----------
        df_candidates : pd.DataFrame, optional
            DataFrame containing candidate data, including their party affiliations and list IDs. Default is None.
        include_zh : bool, optional
            If True, metrics are also calculated for the 'ZH' district. Default is False.
        verbose : bool, optional
            If True, prints the distance methods used for metrics. Default is False.
        progress_bar : bool, optional
            If True, displays a progress bar during the calculation. Default is False.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the calculated metrics for each distance method.

        Notes
        -----
        - The method requires that party ranks have been added to the DataFrame using `add_party_ranks`.
        - The method requires that disagreement counts have been added to the DataFrame using `add_disagreement_counts`.
        - Metrics calculated include:
          - `party_match_accuracy`: The accuracy of matching the voter's preferred party.
          - `party_rank_mean`: The mean rank of the voter's preferred party.
          - `party_rank_frac_mean`: The mean fractional rank of the voter's preferred party.
          - `party_in_top_10_perc`: The percentage of lists in the top 10%.
          - `party_in_top_50_perc`: The percentage of lists in the top 50%.
          - `disagreement_top_frac_mean`: The mean fractional disagreement for top recommendations.
          - `disagreement_strong_top_frac_mean`: The mean fractional strong disagreement for top recommendations.
          - `disagreement_spc_frac_mean`: The mean fractional disagreement for seats-per-canton recommendations.
          - `disagreement_strong_spc_frac_mean`: The mean fractional strong disagreement for seats-per-canton recommendations.
          - If `include_zh` is True, the same metrics are calculated for the 'ZH' district.
        """

        if not any([c for c in self.columns if c.startswith("_ListPartyRank_")]):
            # warning that we need to run add_party_ranks first
            warnings.warn(
                "No party ranks found in DataFrame. Will run add_party_ranks first."
            )
            self.add_party_ranks(df_candidates, progress_bar=progress_bar)

        if not any([c for c in self.columns if c.startswith("_disagreements")]):
            # warning that we need to run add_disagreement_counts first
            warnings.warn(
                "No disagreement counts found in DataFrame. Will run add_disagreement_counts first."
            )
            self.add_disagreement_counts(df_candidates, progress_bar=progress_bar)

        metric_dict = defaultdict(list)
        distance_methods = self.get_distance_methods()
        if verbose:
            print("Distance methods used for metrics:")
            print(distance_methods)

        iterator = distance_methods if not progress_bar else tqdm(distance_methods)
        for method in iterator:
            metric_dict["method"].append(method)

            # add party rank metrics
            party_ranks = self[f"_ListPartyRank_{method}"].dropna()
            party_rank_fracs = self[f"_ListPartyRankFrac_{method}"].dropna()

            metric_dict["party_match_accuracy"].append((party_ranks == 0).mean())
            metric_dict["party_rank_mean"].append(np.mean(party_ranks))
            metric_dict["party_rank_frac_mean"].append(np.mean(party_rank_fracs))
            metric_dict["party_in_top_10_perc"].append((party_rank_fracs <= 0.1).mean())
            metric_dict["party_in_top_50_perc"].append((party_rank_fracs <= 0.5).mean())

            # add disagreement metrics
            metric_dict["disagreement_top_frac_mean"].append(
                self[f"_disagreements_top_frac_{method}"].mean()
            )
            metric_dict["disagreement_strong_top_frac_mean"].append(
                self[f"_disagreements_strong_top_frac_{method}"].mean()
            )
            metric_dict["disagreement_spc_frac_mean"].append(
                self[f"_disagreements_spc_frac_{method}"].mean()
            )
            metric_dict["disagreement_strong_spc_frac_mean"].append(
                self[f"_disagreements_strong_spc_frac_{method}"].mean()
            )

            if include_zh:
                party_ranks_zh = self[self["_district"] == "ZH"][
                    f"_ListPartyRank_{method}"
                ].dropna()
                party_rank_fracs_zh = self[self["_district"] == "ZH"][
                    f"_ListPartyRankFrac_{method}"
                ].dropna()

                metric_dict["party_match_accuracy_zh"].append(
                    (party_ranks_zh == 0).mean()
                )
                metric_dict["party_rank_mean_zh"].append(np.mean(party_ranks_zh))
                metric_dict["party_rank_frac_mean_zh"].append(
                    np.mean(party_rank_fracs_zh)
                )
                metric_dict["party_in_top_10_perc_zh"].append(
                    (party_rank_fracs_zh <= 0.1).mean()
                )
                metric_dict["party_in_top_90_perc"].append(
                    (party_rank_fracs <= 0.9).mean()
                )

        return pd.DataFrame(metric_dict)

    def get_party_list_popularities(self) -> pd.DataFrame:
        """
        IMPORTANT: This method doesn't calculate the party visibilities.

        Calculate the proportions of top list recommendation parties for each distance method.

        This method computes the proportions of top parties for each distance method and for the 'ZH' district.
        It also calculates the preference for each party and reorders the results based on a specified party order.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the proportions of top parties for each distance method and the 'ZH' district,
            as well as the preference for each party, reordered based on a specified party order.

        Notes
        -----
        - The method uses the `PARTIES_LEFT_TO_RIGHT` constant to reorder the party preferences.
        - The proportions are calculated using the `value_counts` method with normalization.
        """

        # Calculate proportions of top parties for each method
        df_party_popularities = pd.concat(
            [
                self[f"_ListParty_1_{method}"]
                .value_counts(normalize=True)
                .to_frame()
                .rename(columns={"proportion": f"top_{method}"})
                for method in self.get_distance_methods()
            ],
            axis=1,
        ).join(
            pd.concat(
                [
                    self[self["_district"] == "ZH"][f"_ListParty_1_{method}"]
                    .value_counts(normalize=True)
                    .to_frame()
                    .rename(columns={"proportion": f"top_zh_{method}"})
                    for method in self.get_distance_methods()
                ],
                axis=1,
            )
        )

        # Calculate preference for each party
        party_preferences = (
            self["_party"]
            .value_counts(normalize=True)
            .to_frame()
            .rename(columns={"proportion": "pref_party"})
            .join(df_party_popularities, how="right")
        )

        # Reorder based on specified party order
        party_preferences_ordered = party_preferences.loc[
            [p for p in PARTIES_LEFT_TO_RIGHT if p in party_preferences.index]
        ].T

        return party_preferences_ordered

    def get_party_list_popularity_diffs(
        self,
        party_popularities: pd.DataFrame = None,
        relative_changes: bool = False,
        include_pref: bool = False,
    ) -> pd.DataFrame:
        """
        Calculate differences in party popularity across different distance methods.

        This method computes the differences in party popularity for each distance method compared to the median popularity
        of other methods. Optionally, it can also include the preference for each party and calculate relative changes.

        Parameters
        ----------
        party_popularities : pd.DataFrame, optional
            DataFrame containing the proportions of top parties for each distance method. If None, the method will calculate
            the party popularities using `get_party_list_popularities`. Default is None.
        relative_changes : bool, optional
            If True, the differences are calculated as relative changes (percentage differences). Default is False.
        include_pref : bool, optional
            If True, includes the preference for each party in the output DataFrame. Default is False.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the differences in party popularity for each distance method, and optionally the preference
            for each party.

        Notes
        -----
        - The method uses the `get_party_list_popularities` method to calculate party popularities if not provided.
        - The differences are calculated by subtracting the median popularity of other methods from the popularity of each method.
        - If `relative_changes` is True, the differences are divided by the median popularity to get relative changes.
        - If `include_pref` is True, the preference for each party is included in the output DataFrame.
        """

        if party_popularities is None:
            warnings.warn(
                "No party popularities DataFrame provided. Calculating party popularities."
            )
            party_popularities = self.get_party_list_popularities()

        cols = []

        if include_pref:
            median = party_popularities.loc[
                [i for i in party_popularities.index if "top" in i and "zh" not in i]
            ].median()
            diff = party_popularities.loc["pref_party"] - median
            diff = diff / median if relative_changes else diff
            cols.append(diff)

        distance_methods = self.get_distance_methods()
        for method in distance_methods:
            median_excl = party_popularities.loc[
                [
                    i
                    for i in party_popularities.index
                    if "top" in i and "zh" not in i and i != f"top_{method}"
                ]
            ].median()
            diff = party_popularities.loc[f"top_{method}"] - median_excl
            diff = diff / median_excl if relative_changes else diff
            cols.append(diff)

        df_party_popularity_diff = pd.concat(cols, axis=1)
        df_party_popularity_diff.columns = (
            distance_methods if not include_pref else ["pref_party"] + distance_methods
        )
        df_party_popularity_diff = df_party_popularity_diff.T
        return df_party_popularity_diff

    def add_n_votes(
        self, json_path: str = os.path.join(DATA_FOLDER, "NRW2023-kandidierende.json")
    ) -> "SVDataFrame":
        """
        Add vote counts and election status to the 2023 candidate DataFrame.

        This method merges vote counts and election status from a JSON file into the candidate DataFrame.
        It also calculates expectation-normalized vote counts for each district.

        Parameters
        ----------
        json_path : str, optional
            The path to the JSON file containing vote counts and election status. Default is 'NRW2023-kandidierende.json'.

        Returns
        -------
        SVDataFrame
            The updated candidate DataFrame with added vote counts, election status, and normalized vote counts.

        Notes
        -----
        - If the '_n_votes' column already exists in the DataFrame, a warning is issued and the method returns the original DataFrame.
        - The JSON file is expected to contain vote counts and election status for each candidate.
        - The method maps canton names to their abbreviations and merges the vote counts with the candidate DataFrame.
        - Expectation-normalized vote counts are calculated for each district to account for differences in the number of votes cast.
        """

        # Check if the term is 2023
        if self.term != 2023:
            warnings.warn(
                "This method can only add election vote results for the 2023 candidate dataset. Operation aborted."
            )
            return self

        # Check if votes have already been added
        if "_n_votes" in self.columns:
            warnings.warn("Votes have already been added to the DataFrame. Skipping.")
            return self

        # Load the JSON data
        with open(json_path) as json_data:
            data = json.load(json_data)
            df_results = pd.DataFrame(data["level_kantone"])

        # Map canton names to their abbreviations
        df_results["_district"] = df_results["kanton_bezeichnung"].map(
            KANTONBEZEICHNUNG2DISTRICT.get
        )

        # Create a full_name and year_of_birth column to join with self
        df_results["full_name"] = df_results["vorname"] + " " + df_results["name"]
        df_results["year_of_birth"] = df_results["geburtsjahr"]

        # Merge the vote counts with self
        self = self.merge(
            df_results[
                ["full_name", "year_of_birth", "stimmen_kandidat", "flag_gewaehlt"]
            ],
            on=["full_name", "year_of_birth"],
            how="left",
        ).rename(columns={"stimmen_kandidat": "_n_votes", "flag_gewaehlt": "_elected"})

        # Drop any existing _n_votes_normalized column to avoid conflicts
        self.drop(columns="_n_votes_normalized", inplace=True, errors="ignore")

        # Calculate normalized votes for each district
        for district in self["_district"].unique():
            candidate_mask = (self["_district"] == district) & (
                ~self["_n_votes"].isna()
            )
            n_district_votes = self.loc[candidate_mask, "_n_votes"].sum()
            n_district_candidates = candidate_mask.sum()

            if n_district_candidates > 0:  # Ensure division by zero doesn't occur
                self.loc[candidate_mask, "_n_votes_normalized"] = self.loc[
                    candidate_mask, "_n_votes"
                ] / (n_district_votes / n_district_candidates)

        return self

    def get_compacted(self, copy: bool = True):
        """
        Returns a compacted version of the DataFrame, simplifying answer possibilities.

        This method creates a compacted version of the DataFrame by mapping the answer possibilities to a simplified
        scale (0, 50, 100). The mapping is applied to the answer columns of the DataFrame.

        Parameters
        ----------
        copy : bool, optional
            If True, returns a copy of the DataFrame with the compacted answers. If False, modifies the DataFrame in place.
            Default is True.

        Returns
        -------
        SVDataFrame
            A compacted version of the DataFrame with simplified answer possibilities.

        Notes
        -----
        - The method uses the `ANSWER_POSSIBILITIES` constant to map the answers to the simplified scale.
        - The mapping is applied only to the answer columns of the DataFrame.
        """

        if copy:
            df = self.copy()
        else:
            df = self

        compact_map = {
            ap: 0 if ap < 50 else 50 if ap == 50 else 100 for ap in ANSWER_POSSIBILITIES
        }

        df[df.answer_cols] = df[df.answer_cols].map(compact_map.get)

        return df

        if copy:
            df = self.copy()
        else:
            df = self

        compact_map = {
            ap: 0 if ap < 50 else 50 if ap == 50 else 100 for ap in ANSWER_POSSIBILITIES
        }

        df[df.answer_cols] = df[df.answer_cols].map(compact_map.get)

        return df


# FACTORY FUNCTIONS -----------------------------------------------------------------------------------------


def build_all(
    clean: bool = True, verbose: bool = False
) -> tuple[SVDataFrame, SVDataFrame, pd.DataFrame]:
    """
    Build all DataFrames for the 2023 Smartvote dataset.

    Parameters
    ----------
    clean : bool, optional
        Whether to clean the DataFrames. Default is True.
    verbose : bool, optional
        Whether to print additional information. Default is False.

    Returns
    -------
    df_voters : SVDataFrame
        The voter DataFrame.
    df_candidates : SVDataFrame
        The candidate DataFrame.
    df_questions : pd.DataFrame
        The question DataFrame.
    """
    return (
        build_voters(clean=clean, verbose=verbose),
        build_candidates(clean=clean, verbose=verbose),
        build_questions(),
    )


def build_all19(
    clean: bool = True, verbose: bool = False
) -> tuple[SVDataFrame, SVDataFrame, pd.DataFrame]:
    """
    Build all DataFrames for the 2023 Smartvote dataset.

    Parameters
    ----------
    clean : bool, optional
        Whether to clean the DataFrames. Default is True.
    verbose : bool, optional
        Whether to print additional information. Default is False.

    Returns
    -------
    df_voters19 : SVDataFrame
        The voter DataFrame.
    df_candidates19 : SVDataFrame
        The candidate DataFrame.
    df_questions19 : pd.DataFrame
        The question DataFrame.
    """
    return (
        build_voters19(clean=clean, verbose=verbose),
        build_candidates19(clean=clean, verbose=verbose),
        build_questions19(),
    )


def build_voters(clean: bool = True, verbose: bool = False) -> SVDataFrame:
    """
    Build the voter DataFrame for the 2023 Smartvote dataset.

    Parameters
    ----------
    clean : bool, optional
        Whether to clean the DataFrame. Default is True.
    verbose : bool, optional
        Whether to print additional information. Default is False.

    Returns
    -------
    df_voters : SVDataFrame
        The voter DataFrame.
    """
    # preprocessing
    if not os.path.exists(os.path.join(CACHE_FOLDER, "df_voters_topmatch.parquet")):
        warnings.warn(
            "No preprocessed voter DataFrame found. Preprocessing (removing Smartvote's match columns + "
            "type casting) starts now. This takes approximately 15 minutes (~160 iterations)."
        )
        preprocess_voters()
    else:
        if verbose:
            print("Preprocessed voter DataFrame found.")

    if clean:
        hash_str = hashlib.sha256(inspect.getsource(clean_voters).encode()).hexdigest()
        cache_path = os.path.join(CACHE_FOLDER, f"df_voters-{hash_str}.parquet")
        if os.path.exists(cache_path):
            if verbose:
                print("Cleaned voter DataFrame found.")
            # load cached dataframe
            df_voters = pd.read_parquet(cache_path)
        else:
            print("No cleaned voter DataFrame found. Cleaning starts now.")
            # load preprocessed dataframe
            df_voters = pd.read_parquet(
                os.path.join(CACHE_FOLDER, "df_voters_topmatch.parquet")
            )
            # cleaning
            df_voters = clean_voters(df_voters)
            # cache cleaned dataframe
            df_voters.to_parquet(ensure_path(cache_path))
            remove_old_cache_files(cache_path)
    else:
        # load preprocessed dataframe
        df_voters = pd.read_parquet(
            os.path.join(CACHE_FOLDER, "df_voters_topmatch.parquet")
        )

    return SVDataFrame(df_voters, term=2023, verbose=verbose)


def preprocess_voters(chunk_size: int = 10000) -> None:
    """
    Preprocess the voter DataFrame for the 2023 Smartvote dataset.

    Parameters
    ----------
    chunk_size : int, optional
        The chunk size for reading the CSV file. Default is 10000.
    """
    df_voters_row = pd.read_csv(
        os.path.join(DATA_FOLDER, SV23_FOLDER, VOTERS_FILE), nrows=1
    )
    df_voters_row = df_voters_row.drop(
        columns=[
            c
            for c in df_voters_row.columns
            if c.startswith("match") and c not in ["matchID_1", "matchValue_1"]
        ]
    )

    # define column types
    type_mappings = dict(
        {
            c: "int64"
            for c in df_voters_row.columns
            if c.startswith("N_")
            or c.endswith("TYPE")
            or c
            in [
                "user_language",
                "electionID",
                "districtID",
                "language",
                "birthYEAR",
                "age",
                "zip",
                "education",
                "interest",
                "position",
            ]
            or c.startswith("matchID_")
        },
        **{
            c: "float64"
            for c in df_voters_row.columns
            if c.startswith("weight_")
            or c.startswith("smartmap_")
            or c.startswith("cleavage_")
            or c.startswith("answer_")
            or c.endswith("_completion")
            or c.startswith("matchValue_")
        },
        **{c: "datetime64[ns]" for c in df_voters_row.columns if c == "recTIME"},
    )

    numeric_cols = [c for c, t in type_mappings.items() if t in ["int64", "float64"]]
    datetime_cols = [c for c, t in type_mappings.items() if t in ["datetime64[ns]"]]

    original_voter_file = os.path.join(DATA_FOLDER, SV23_FOLDER, VOTERS_FILE)
    preprocess_file_csv = ensure_path(
        os.path.join(CACHE_FOLDER, "df_voters_topmatch.csv")
    )
    preprocess_file = os.path.join(CACHE_FOLDER, "df_voters_topmatch.parquet")

    # read csv file in chunks
    for i, chunk in tqdm(
        enumerate(pd.read_csv(original_voter_file, chunksize=chunk_size))
    ):

        # drop all match columns except for top match
        chunk = chunk.drop(
            columns=[
                c
                for c in chunk.columns
                if c.startswith("match") and c not in ["matchID_1", "matchValue_1"]
            ]
        )

        # cast chunk
        for c in numeric_cols:
            chunk[c] = pd.to_numeric(chunk[c], errors="coerce")
        for c in datetime_cols:
            chunk[c] = pd.to_datetime(chunk[c], errors="coerce")

        if i == 0:
            # write first chunk to new csv file
            chunk.to_csv(preprocess_file_csv, index=False)
        else:
            # append subsequent chunks to the existing csv file
            chunk.to_csv(preprocess_file_csv, mode="a", index=False)

    # open csv file and save as parquet
    df_voters = pd.read_csv(preprocess_file_csv)

    # cast chunk
    for c in numeric_cols:
        df_voters[c] = pd.to_numeric(df_voters[c], errors="coerce")
    for c in datetime_cols:
        df_voters[c] = pd.to_datetime(df_voters[c], errors="coerce")

    df_voters.replace(" ", np.nan, inplace=True)

    df_voters.to_parquet(preprocess_file)
    warnings.warn(
        f"Preprocessing finished. Preprocessed voter DataFrame saved to {preprocess_file}"
    )


def clean_voters(df_voters: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the voter DataFrame for the 2023 Smartvote dataset.

    This function performs several cleaning steps on the input voter DataFrame to ensure data quality and consistency.
    It handles missing values, removes corrupted or unrealistic data, and adds additional columns for further analysis.

    Parameters
    ----------
    df_voters : pd.DataFrame
        The input voter DataFrame containing raw voter data.

    Returns
    -------
    pd.DataFrame
        The cleaned voter DataFrame with additional columns for analysis.

    Notes
    -----
    - Empty strings in the DataFrame are replaced with NaN values.
    - The redundant 'age' column is dropped.
    - Rows with missing 'electionID' values are removed.
    - Unrealistic 'birthYEAR' values (less than 1900 or greater than 2005) are set to NaN.
    - Irregular 'zip' values (less than 1000 or greater than 9658) are set to NaN.
    - Only recommendations from the deluxe questionnaire with at least 15 answers and at most 75 answers are kept.
    - A '_time' column is added by merging with a time DataFrame, and the 'recTIME' column is dropped.
    - Recommendations before the publishing date (2023-08-23) and after the election date (2023-10-23) are removed.
    - Recommendations with more than 14 consecutive equal answers are removed.
    - Duplicate voter IDs are removed, keeping the record with the most answers, latest time, and highest recID.
    - Decoded columns for party, education, district, and language are added.
    - Additional statistics columns '_answer_strength', '_answer_strength_std', '_maxDist_L2_sv', and '_maxDistCorrect_L2_sv' are added.
    - The index of the DataFrame is reset.
    """
    # empty strings -> nan
    df_voters.replace(" ", np.nan, inplace=True)

    # drop redundant age column
    df_voters.drop(columns=["age"], inplace=True)

    # remove corrupted recommendations with no electionID
    df_voters.dropna(subset=["electionID"], inplace=True)

    # unrealistic birthYEAR -> nan
    df_voters.loc[
        ((df_voters["birthYEAR"] < 1900) | (df_voters["birthYEAR"] > 2005)), "birthYEAR"
    ] = np.nan

    # irregular zip -> nan
    df_voters.loc[((df_voters["zip"] > 9658) | (df_voters["zip"] < 1000)), "zip"] = (
        np.nan
    )

    # only keep recommendations from deluxe questionnaire with at least 60 answers and at most 75
    df_voters = df_voters[
        (df_voters["N_answers"] >= 15) & (df_voters["N_answers"] <= 75)
    ]

    # add time column
    df_time = pd.read_csv(os.path.join(DATA_FOLDER, SV23_FOLDER, TIMESTAMP_FILE))
    df_time["_time"] = pd.to_datetime(df_time["recTIME_REC"])
    df_voters = df_voters.merge(df_time[["recID", "_time"]], on="recID")
    df_voters.drop(columns="recTIME", inplace=True)

    # remove recommendations before publishing and after election date
    df_voters = df_voters[
        ("2023-08-23" <= df_voters["_time"]) & (df_voters["_time"] < "2023-10-23")
    ]

    # remove recommendations with more than 14 consecutive equal answers
    answer_cols = get_cols(df_voters, col_type="answer")
    df_voters = df_voters[count_consecutive_values(df_voters[answer_cols].values) <= 14]

    # remove duplicate voter_ids
    df_voters = df_voters.sort_values(
        by=["N_answers", "_time", "recID"], ascending=False
    ).drop_duplicates(subset=["voterID"], keep="first")

    # add decoded columns
    decoded_columns = {
        "_party": df_voters["pref_party"].astype("float64").map(ID2PREF_PARTY),
        "_education": df_voters["education"].map(ID2EDUCATION),
        "_district": df_voters["districtID"].map(ID2DISTRICT),
        "_language": df_voters["language"].map(ID2LANGUAGE),
    }
    df_decoded = pd.concat(decoded_columns, axis=1)

    # merge decoded columns with original DataFrame
    df_voters = pd.concat([df_voters, df_decoded], axis=1)

    # add stats columns
    df_voters["_answer_strength"] = (df_voters[answer_cols] - 50).abs().mean(axis=1)
    df_voters["_answer_strength_std"] = (df_voters[answer_cols] - 50).abs().std(axis=1)

    weight_cols = get_cols(df_voters, col_type="weight")
    df_voters["_maxDist_L2_sv"] = np.sqrt(
        ((df_voters[weight_cols] * 100) ** 2).sum(axis=1)
    )

    max_diff = df_voters[answer_cols].map(lambda x: max(x, 100 - x)).to_numpy()
    weights = df_voters[weight_cols].to_numpy()
    df_voters["_maxDistCorrect_L2_sv"] = np.sqrt(
        np.nansum((max_diff * weights) ** 2, axis=1)
    )

    # reset index
    df_voters = df_voters.reset_index(drop=True)

    return df_voters


def build_candidates(clean: bool = True, verbose: bool = False) -> SVDataFrame:
    """
    Build the candidate DataFrame for the 2023 Smartvote dataset.

    Parameters
    ----------
    clean : bool, optional
        Whether to clean the DataFrame. Default is True.
    verbose : bool, optional
        Whether to print additional information. Default is False.

    Returns
    -------
    df_candidates : SVDataFrame
        The candidate DataFrame.
    """
    if clean:
        hash_str = hashlib.sha256(
            inspect.getsource(clean_candidates).encode()
        ).hexdigest()
        cache_path = os.path.join(CACHE_FOLDER, f"df_candidates-{hash_str}.parquet")
        if os.path.exists(cache_path):
            if verbose:
                print("Cleaned candidate DataFrame found.")
            # load cached dataframe
            df_candidates = pd.read_parquet(cache_path)
        else:
            print("No cleaned candidate DataFrame found. Cleaning starts now.")
            # load original dataframe
            df_candidates = pd.read_csv(
                os.path.join(DATA_FOLDER, SV23_FOLDER, CANDIDATES_FILE), index_col=0
            )
            # cleaning
            df_candidates = clean_candidates(df_candidates)
            # cache cleaned dataframe
            df_candidates.to_parquet(ensure_path(cache_path))
            remove_old_cache_files(cache_path)
    else:
        # load preprocessed dataframe
        df_candidates = pd.read_csv(
            os.path.join(DATA_FOLDER, SV23_FOLDER, CANDIDATES_FILE), index_col=0
        )

    return SVDataFrame(df_candidates, term=2023, verbose=verbose)


def clean_candidates(df_candidates) -> pd.DataFrame:
    """
    Clean the candidate DataFrame for the 2023 Smartvote dataset.

    This function performs several cleaning steps on the input candidate DataFrame to ensure data quality and consistency.
    It handles missing values, removes unnecessary columns, and adds additional columns for further analysis.

    Parameters
    ----------
    df_candidates : pd.DataFrame
        The input candidate DataFrame containing raw candidate data.

    Returns
    -------
    pd.DataFrame
        The cleaned candidate DataFrame with additional columns for analysis.

    Notes
    -----
    - Only candidates that answered all questions (N_answers == 75) are kept.
    - Reconciled answer columns (columns containing 'REC' and starting with 'answer_') are dropped.
    - Decoded columns for party, education, district, and language are added.
    - Additional statistics columns '_answer_strength' and '_answer_strength_std' are added.
    - The index of the DataFrame is reset.
    """
    # only keep candidates that answered all questions (was required)
    df_candidates = df_candidates[df_candidates["N_answers"] == 75]

    # drop reconciled answer columns
    df_candidates.drop(
        columns=[
            c for c in df_candidates.columns if "REC" in c and c.startswith("answer_")
        ],
        inplace=True,
    )

    # add decoded columns
    decoded_columns = {
        "_party_rec": df_candidates["party_REC6"].map(ID2PARTY_REC),
        "_party": df_candidates["party_short"].map(PARTY_SHORT2PARTY),
        "_education": df_candidates["highest_education"].map(ID2EDUCATION),
        "_district": df_candidates["ID_district"].map(ID2DISTRICT),
        "_language": df_candidates["language"].map(ID2LANGUAGE),
    }
    df_decoded = pd.concat(decoded_columns, axis=1)

    # merge decoded columns with original DataFrame
    df_candidates = pd.concat([df_candidates, df_decoded], axis=1)

    # add stats columns
    df_candidates["_answer_strength"] = (
        (df_candidates[get_cols(df_candidates, col_type="answer")] - 50)
        .abs()
        .mean(axis=1)
    )
    df_candidates["_answer_strength_std"] = (
        (df_candidates[get_cols(df_candidates, col_type="answer")] - 50)
        .abs()
        .std(axis=1)
    )

    # reset index
    df_candidates = df_candidates.reset_index(drop=True)

    return df_candidates


def build_questions() -> pd.DataFrame:
    """
    Build the question DataFrame for the 2023 Smartvote dataset.

    Returns
    -------
    df_questions : pd.DataFrame
        The question DataFrame.
    """
    df_questions = pd.read_excel(
        os.path.join(DATA_FOLDER, SV23_FOLDER, QUESTIONS_FILE), header=1
    )

    # add decoded category column
    df_questions["_category"] = df_questions["category"].map(QUESTION_ID2CATEGORY)
    df_questions["_n_options"] = df_questions["type"].apply(lambda t: int(t[0]))

    return df_questions


def build_voters19(clean: bool = True, verbose: bool = False) -> SVDataFrame:
    """
    Build the voter DataFrame for the 2019 Smartvote dataset.

    Parameters
    ----------
    clean : bool, optional
        Whether to clean the DataFrame. Default is True.
    verbose : bool, optional
        Whether to print additional information. Default is False.

    Returns
    -------
    df_voters19 : SVDataFrame
        The voter DataFrame.
    """
    if clean:
        hash_str = hashlib.sha256(
            inspect.getsource(clean_voters19).encode()
        ).hexdigest()
        cache_path = os.path.join(CACHE_FOLDER, f"df_voters19-{hash_str}.parquet")
        if os.path.exists(cache_path):
            if verbose:
                print("Cleaned voter19 DataFrame found.")
            # load cached dataframe
            df_voters19 = pd.read_parquet(cache_path)
        else:
            print("No cleaned voter19 DataFrame found. Cleaning starts now.")
            # load dataframe
            df_voters19 = pd.read_csv(
                os.path.join(DATA_FOLDER, SV19_FOLDER, VOTERS19_FILE), index_col=0
            )
            # cleaning
            df_voters19 = clean_voters19(df_voters19)
            # cache cleaned dataframe
            df_voters19.to_parquet(ensure_path(cache_path))
            remove_old_cache_files(cache_path)
    else:
        # load preprocessed dataframe
        df_voters19 = pd.read_csv(
            os.path.join(DATA_FOLDER, SV19_FOLDER, VOTERS19_FILE), index_col=0
        )

    return SVDataFrame(df_voters19, term=2019, verbose=verbose)


def clean_voters19(df_voters19: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the voter DataFrame for the 2019 Smartvote dataset.

    This function performs several cleaning steps on the input voter DataFrame to ensure data quality and consistency.
    It handles missing values, removes corrupted or unrealistic data, and adds additional columns for further analysis.

    Parameters
    ----------
    df_voters19 : pd.DataFrame
        The input voter DataFrame containing raw voter data.

    Returns
    -------
    pd.DataFrame
        The cleaned voter DataFrame with additional columns for analysis.

    Notes
    -----
    - Unrealistic 'year_of_birth_REC' values (less than 1900 or greater than 2001) are set to NaN.
    - Irregular 'zip' values (less than 1000 or greater than 9658) are set to NaN.
    - Only recommendations from the deluxe questionnaire with at least 15 answers and at most 75 answers are kept.
    - A '_time' column is added by converting the 'timestamp' column to datetime.
    - Recommendations before the election date (2019-10-21) are removed.
    - Recommendations with more than 14 consecutive equal answers are removed.
    - Duplicate user IDs & voter IDs are removed, keeping the record with the most answers, latest time, and highest recommendation ID.
    - Weights are mapped from categorical to numeric values.
    - Decoded columns for party, education, district, gender, and language are added.
    - Cleavage columns are normalized by dividing by 100.
    - Additional statistics columns '_answer_strength', '_answer_strength_std', '_maxDist_L2_sv', and '_maxDistCorrect_L2_sv' are added.
    - The index of the DataFrame is reset.
    """

    # unrealistic birthYEAR -> nan
    df_voters19.loc[
        (
            (df_voters19["year_of_birth_REC"] < 1900)
            | (df_voters19["year_of_birth_REC"] > 2001)
        ),
        "year_of_birth_REC",
    ] = np.nan

    # irregular zip -> nan
    df_voters19.loc[
        ((df_voters19["zip"] > 9658) | (df_voters19["zip"] < 1000)), "zip"
    ] = np.nan

    # only keep recommendations from deluxe questionnaire with at least 15 answers and at most 75
    df_voters19 = df_voters19[
        (df_voters19["n_answers"] >= 15) & (df_voters19["n_answers"] <= 75)
    ]

    # add timestamp
    df_voters19["_time"] = pd.to_datetime(df_voters19["timestamp"])

    # remove recommendations before publishing and after election date
    df_voters19 = df_voters19[df_voters19["_time"] < "2019-10-21"]

    # remove recommendations with more than 14 consecutive equal answers
    answer_cols = get_cols(df_voters19, col_type="answer")
    df_voters19 = df_voters19[
        count_consecutive_values(df_voters19[answer_cols].values) <= 14
    ]

    # remove duplicate user_ids
    temp = df_voters19[df_voters19["ID_user"].isna()]
    df_voters19 = df_voters19.sort_values(
        by=["n_answers", "_time", "ID_recommendation"], ascending=False
    ).drop_duplicates(subset=["ID_user"], keep="first")
    df_voters19 = df_voters19.merge(temp, how="outer")

    # remove duplicate voter_ids
    df_voters19 = df_voters19.sort_values(
        by=["n_answers", "_time", "ID_recommendation"], ascending=False
    ).drop_duplicates(subset=["ID_voter"], keep="first")

    # map weights from categorical to numeric values
    weight_map = {
        1: 0.5,
        2: 1,
        3: 2,
    }
    weight_cols = get_cols(df_voters19, col_type="weight")
    df_voters19[weight_cols] = df_voters19[weight_cols].map(weight_map.get)

    # add decoded columns
    decoded_columns = {
        "_party": df_voters19["party"].map(ID2PARTY19),
        "_education": df_voters19["education"].map(ID2EDUCATION),
        "_district": df_voters19["ID_district"].map(ID2DISTRICT19),
        "_gender": df_voters19["gender"].map(ID2GENDER19),
        "_language": df_voters19["language"].map(ID2LANGUAGE19),
    }
    df_decoded = pd.concat(decoded_columns, axis=1)

    # merge decoded columns with original DataFrame
    df_voters19 = pd.concat([df_voters19, df_decoded], axis=1)

    # normalize cleavage columns
    for c in [c for c in df_voters19.columns if c.startswith("cleavage_")]:
        df_voters19[c] = df_voters19[c] / 100

    # add stats columns
    df_voters19["_answer_strength"] = (df_voters19[answer_cols] - 50).abs().mean(axis=1)
    df_voters19["_answer_strength_std"] = (
        (df_voters19[answer_cols] - 50).abs().std(axis=1)
    )

    df_voters19["_maxDist_L2_sv"] = np.sqrt(
        ((df_voters19[weight_cols] * 100) ** 2).sum(axis=1)
    )

    max_diff = df_voters19[answer_cols].map(lambda x: max(x, 100 - x)).to_numpy()
    weights = df_voters19[weight_cols].to_numpy()
    df_voters19["_maxDistCorrect_L2_sv"] = np.sqrt(
        np.nansum((max_diff * weights) ** 2, axis=1)
    )

    # reset index
    df_voters19 = df_voters19.reset_index(drop=True)

    return df_voters19


def build_candidates19(clean: bool = True, verbose: bool = False) -> SVDataFrame:
    """
    Build the candidate DataFrame for the 2019 Smartvote dataset.

    Parameters
    ----------
    clean : bool, optional
        Whether to clean the DataFrame. Default is True.
    verbose : bool, optional
        Whether to print additional information. Default is False.

    Returns
    -------
    df_candidates19 : SVDataFrame
        The candidate DataFrame.
    """
    if clean:
        hash_str = hashlib.sha256(
            inspect.getsource(clean_candidates19).encode()
        ).hexdigest()
        cache_path = os.path.join(CACHE_FOLDER, f"df_candidates19-{hash_str}.parquet")
        if os.path.exists(cache_path):
            if verbose:
                print("Cleaned candidate19 DataFrame found.")
            # load cached dataframe
            df_candidates19 = pd.read_parquet(cache_path)
        else:
            print("No cleaned candidate19 DataFrame found. Cleaning starts now.")
            # load original dataframe
            df_candidates19 = pd.read_csv(
                os.path.join(DATA_FOLDER, SV19_FOLDER, CANDIDATES19_FILE), index_col=0
            )
            # cleaning
            df_candidates19 = clean_candidates19(df_candidates19)
            # save cleaned dataframe
            df_candidates19.to_parquet(ensure_path(cache_path))
            remove_old_cache_files(cache_path)
    else:
        # load preprocessed dataframe
        df_candidates19 = pd.read_csv(
            os.path.join(DATA_FOLDER, SV19_FOLDER, CANDIDATES19_FILE), index_col=0
        )

    return SVDataFrame(df_candidates19, term=2019, verbose=verbose)


def clean_candidates19(df_candidates19: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the candidate DataFrame for the 2019 Smartvote dataset.

    This function performs several cleaning steps on the input candidate DataFrame to ensure data quality and consistency.
    It handles missing values, removes unnecessary columns, and adds additional columns for further analysis.

    Parameters
    ----------
    df_candidates19 : pd.DataFrame
        The input candidate DataFrame containing raw candidate data.

    Returns
    -------
    pd.DataFrame
        The cleaned candidate DataFrame with additional columns for analysis.

    Notes
    -----
    - Offset values in 'ID_district', 'ID_party', and 'ID_list' columns are corrected.
    - Only candidates that answered all questions (n_answers == 75) are kept.
    - Decoded columns for district and party are added.
    - Cleavage columns are normalized by dividing by 100.
    - Additional statistics columns '_answer_strength' and '_answer_strength_std' are added.
    - The index of the DataFrame is reset.
    """
    # clean offset ID_district col (ID_party and ID_list seem to have same problem, but there is not reference to
    # verify whether offset is the same as for ID_district)
    offset_cols = ["ID_district", "ID_party", "ID_list"]
    for col in offset_cols:
        df_candidates19[col] = df_candidates19[col] - 4.42e10 + 1

    # only keep candidates that answered all questions (was required)
    df_candidates19 = df_candidates19[df_candidates19["n_answers"] == 75]

    # add decoded columns
    decoded_columns = {
        "_district": df_candidates19["ID_district"].map(ID2DISTRICT19),
        "_party_rec": df_candidates19["party_short"].map(PARTY_SHORT2PARTY_REC19),
        "_party": df_candidates19["party_short"].map(PARTY_SHORT2PARTY19),
    }
    df_decoded = pd.concat(decoded_columns, axis=1)

    # merge decoded columns with original DataFrame
    df_candidates19 = pd.concat([df_candidates19, df_decoded], axis=1)

    # normalize cleavage columns
    for c in get_cols(df_candidates19, col_type="cleavage"):
        df_candidates19[c] = df_candidates19[c] / 100

    # add stats columns
    df_candidates19["_answer_strength"] = (
        (df_candidates19[get_cols(df_candidates19, col_type="answer")] - 50)
        .abs()
        .mean(axis=1)
    )
    df_candidates19["_answer_strength_std"] = (
        (df_candidates19[get_cols(df_candidates19, col_type="answer")] - 50)
        .abs()
        .std(axis=1)
    )

    return df_candidates19


def build_questions19() -> pd.DataFrame:
    """
    Build the question DataFrame for the 2019 Smartvote dataset.

    Returns
    -------
    df_questions19 : pd.DataFrame
        The question DataFrame.
    """
    df_questions = pd.read_csv(
        os.path.join(DATA_FOLDER, SV19_FOLDER, QUESTIONS19_FILE), index_col=0
    )

    df_questions["_n_options"] = df_questions["type"].apply(lambda t: int(t[-1]))

    return df_questions


def build_questions_shared(
    df_questions: pd.DataFrame,
    df_questions19: pd.DataFrame,
    exact_question_ids: list[tuple[int, int]],
    similar_question_ids: list[tuple[int, int]] = None,
    similar_opposite_question_ids: list[tuple[int, int]] = None,
) -> pd.DataFrame:
    """
    Build the shared question DataFrame for the 2019 and 2023 Smartvote datasets.

    Parameters
    ----------
    df_questions : pd.DataFrame
        The question DataFrame for the 2023 Smartvote dataset.
    df_questions19 : pd.DataFrame
        The question DataFrame for the 2019 Smartvote dataset.
    exact_question_ids : list[tuple[int, int]]
        The list of exact question IDs.
    similar_question_ids : list[tuple[int, int]], optional
        The list of similar question IDs. Default is None.
    similar_opposite_question_ids : list[tuple[int, int]], optional
        The list of similar opposite question IDs. Default is None.

    Returns
    -------
    df_questions_shared : pd.DataFrame
        The shared question DataFrame.

    """
    if similar_question_ids is None:
        similar_question_ids = []
    if similar_opposite_question_ids is None:
        similar_opposite_question_ids = []

    # construct df_questions_shared
    SHARED_QUESTION_IDS = [
        (q[0], q[1])
        for q in sorted(
            exact_question_ids + similar_question_ids + similar_opposite_question_ids
        )
    ]
    df_questions_shared = pd.DataFrame(
        {
            "id": [q[0] for q in SHARED_QUESTION_IDS],
            "id19": [q[1] for q in SHARED_QUESTION_IDS],
            "id_shared": list(range(1, len(SHARED_QUESTION_IDS) + 1)),
            "answer_cols": [f"answer_{q[0]}" for q in SHARED_QUESTION_IDS],
            "answer_cols19": [f"answer_{q[1]}" for q in SHARED_QUESTION_IDS],
            "answer_cols_shared": [
                f"answer_{q}" for q in range(1, len(SHARED_QUESTION_IDS) + 1)
            ],
            "weight_cols": [f"weight_{q[0]}" for q in SHARED_QUESTION_IDS],
            "weight_cols19": [f"weight_{q[1]}" for q in SHARED_QUESTION_IDS],
            "weight_cols_shared": [
                f"weight_{q}" for q in range(1, len(SHARED_QUESTION_IDS) + 1)
            ],
        }
    )

    df_questions_shared = df_questions_shared.merge(
        df_questions.rename(
            columns={"ID_question": "id", "question_EN": "question_en"}
        )[["id", "question_en"]],
        on="id",
    ).merge(
        df_questions19.rename(
            columns={"ID_question": "id19", "question_en": "question_en19"}
        )[["id19", "question_en19"]],
        on="id19",
    )

    cols = [
        "id",
        "id19",
        "id_shared",
        "answer_cols",
        "answer_cols19",
        "answer_cols_shared",
        "weight_cols",
        "weight_cols19",
        "weight_cols_shared",
        "exact",
        "question_en",
        "question_en19",
    ]

    if similar_opposite_question_ids:
        cols.insert(-2, "opposite")
        df_questions_shared["opposite"] = False
        df_questions_shared.loc[
            df_questions_shared["id"].isin(
                [q[0] for q in similar_opposite_question_ids]
            ),
            "opposite",
        ] = True

    df_questions_shared["exact"] = False
    df_questions_shared.loc[
        df_questions_shared["id"].isin([q[0] for q in exact_question_ids]), "exact"
    ] = True

    return df_questions_shared[cols]


def build_df_shared(
    df_voters: pd.DataFrame,
    df_candidates: pd.DataFrame,
    df_voters19: pd.DataFrame,
    df_candidates19: pd.DataFrame,
    df_questions_shared: pd.DataFrame,
) -> tuple[SVDataFrame, SVDataFrame, SVDataFrame, SVDataFrame]:
    """
    Build shared DataFrames for the 2019 and 2023 Smartvote datasets.

    Parameters
    ----------
    df_voters : pd.DataFrame
        The voter DataFrame for the 2023 Smartvote dataset.
    df_candidates : pd.DataFrame
        The candidate DataFrame for the 2023 Smartvote dataset.
    df_voters19 : pd.DataFrame
        The voter DataFrame for the 2019 Smartvote dataset.
    df_candidates19 : pd.DataFrame
        The candidate DataFrame for the 2019 Smartvote dataset.
    df_questions_shared : pd.DataFrame
        The shared question DataFrame.

    Returns
    -------
    df_voters_shared : SVDataFrame
        The shared voter DataFrame for the 2023 Smartvote dataset.
    df_candidates_shared : SVDataFrame
        The shared candidate DataFrame for the 2023 Smartvote dataset.
    df_voters19_shared : SVDataFrame
        The shared voter DataFrame for the 2019 Smartvote dataset.
    df_candidates19_shared : SVDataFrame
        The shared candidate DataFrame for the 2019 Smartvote dataset.
    """

    def flip_opposite(df):
        for c in df_questions_shared[df_questions_shared["opposite"]][
            "answer_cols_shared"
        ].values:
            df[c] = 100 - df[c]
        return df

    # voters ------------------------------------------------
    df_voters_shared = df_voters.copy()

    # drop answer and weight columns that couldn't be matched
    cols_to_drop = [
        c
        for c in df_voters_shared.a().columns
        if c not in df_questions_shared["answer_cols"].values
    ] + [
        c
        for c in df_voters_shared.w().columns
        if c not in df_questions_shared["weight_cols"].values
    ]
    df_voters_shared.drop(columns=cols_to_drop, inplace=True)

    answer_renames = df_questions_shared.set_index("answer_cols")[
        "answer_cols_shared"
    ].to_dict()
    weight_renames = df_questions_shared.set_index("weight_cols")[
        "weight_cols_shared"
    ].to_dict()
    renames = {**answer_renames, **weight_renames}
    df_voters_shared = df_voters_shared.rename(columns=renames)

    df_voters_shared["N_answers_shared"] = (
        ~(df_voters_shared[get_cols(df_voters_shared)].isna())
    ).sum(axis=1)
    df_voters_shared["_answer_strength_shared"] = (
        (df_voters_shared[get_cols(df_voters_shared)] - 50).abs().mean(axis=1)
    )
    df_voters_shared["N_weights_shared"] = (
        df_voters_shared[get_cols(df_voters_shared, col_type="weight")]
        .isin([0.5, 2.0])
        .sum(axis=1)
    )

    df_voters_shared = SVDataFrame(df_voters_shared)

    # candidates ------------------------------------------------
    df_candidates_shared = df_candidates.copy()

    # drop answer and weight columns that couldn't be matched
    df_candidates_shared.drop(
        columns=[
            c
            for c in df_candidates_shared.a().columns
            if c not in df_questions_shared["answer_cols"].values
        ],
        inplace=True,
    )

    answer_renames = df_questions_shared.set_index("answer_cols")[
        "answer_cols_shared"
    ].to_dict()
    df_candidates_shared = df_candidates_shared.rename(columns=answer_renames)

    df_candidates_shared["N_answers_shared"] = (
        ~df_candidates_shared[get_cols(df_candidates_shared)].isna()
    ).sum(axis=1)
    df_candidates_shared["_answer_strength_shared"] = (
        (df_candidates_shared[get_cols(df_candidates_shared)] - 50).abs().mean(axis=1)
    )

    df_candidates_shared = SVDataFrame(df_candidates_shared)

    # voters19 ------------------------------------------------
    df_voters19_shared = df_voters19.copy()

    # drop answer and weight columns that couldn't be matched
    cols_to_drop = [
        c
        for c in df_voters19_shared.a().columns
        if c not in df_questions_shared["answer_cols19"].values
    ] + [
        c
        for c in df_voters19_shared.w().columns
        if c not in df_questions_shared["weight_cols19"].values
    ]
    df_voters19_shared.drop(columns=cols_to_drop, inplace=True)
    answer_renames = df_questions_shared.set_index("answer_cols19")[
        "answer_cols_shared"
    ].to_dict()
    weight_renames = df_questions_shared.set_index("weight_cols19")[
        "weight_cols_shared"
    ].to_dict()
    renames = {**answer_renames, **weight_renames}
    df_voters19_shared = df_voters19_shared.rename(columns=renames)

    if "opposite" in df_questions_shared.columns:
        df_voters19_shared = flip_opposite(df_voters19_shared)

    df_voters19_shared["N_answers_shared"] = (
        ~df_voters19_shared[get_cols(df_voters19_shared)].isna()
    ).sum(axis=1)
    df_voters19_shared["_answer_strength_shared"] = (
        (df_voters19_shared[get_cols(df_voters19_shared)] - 50).abs().mean(axis=1)
    )
    df_voters19_shared["N_weights_shared"] = (
        df_voters19_shared[get_cols(df_voters19_shared, col_type="weight")]
        .isin([0.5, 2.0])
        .sum(axis=1)
    )
    df_voters19_shared.drop(columns=["n_answers", "n_weights"], inplace=True)

    df_voters19_shared = SVDataFrame(df_voters19_shared)

    # candidates19 ------------------------------------------------
    df_candidates19_shared = df_candidates19.copy()

    # drop answer and weight columns that couldn't be matched
    df_candidates19_shared.drop(
        columns=[
            c
            for c in df_candidates19_shared.a().columns
            if c not in df_questions_shared["answer_cols19"].values
        ],
        inplace=True,
    )

    answer_renames = df_questions_shared.set_index("answer_cols19")[
        "answer_cols_shared"
    ].to_dict()
    df_candidates19_shared = df_candidates19_shared.rename(columns=answer_renames)

    if "opposite" in df_questions_shared.columns:
        df_candidates19_shared = flip_opposite(df_candidates19_shared)

    df_candidates19_shared["N_answers_shared"] = (
        ~df_candidates19_shared[get_cols(df_candidates19_shared)].isna()
    ).sum(axis=1)
    df_candidates19_shared["_answer_strength_shared"] = (
        (df_candidates19_shared[get_cols(df_candidates19_shared)] - 50)
        .abs()
        .mean(axis=1)
    )

    df_candidates19_shared = SVDataFrame(df_candidates19_shared)

    return (
        df_voters_shared,
        df_candidates_shared,
        df_voters19_shared,
        df_candidates19_shared,
    )
