import pandas as pd
from prince import CA
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from rsfp.data import SVDataFrame


def add_pca_cols(
        df: SVDataFrame,
        n_dims: int = 2,
        standardize: bool = False,
        return_obj: bool = False,
        obj=None,
        feature_cols=None,
        suffix: str = None
):
    """
    Calculate the Principal Component Analysis (PCA) of the answers and add the results in additional columns to the dataframe.

    Parameters
    ----------
    df : SVDataFrame
        The input dataframe containing the answers.
    n_dims : int, optional
        The number of dimensions to keep in the PCA. Default is 2.
    standardize : bool, optional
        Whether to standardize the data before performing PCA. Default is False.
    return_obj : bool, optional
        Whether to return the PCA object and other preprocessing objects. Default is False.
    obj : tuple, optional
        Pretrained preprocessing objects (imputer, scaler, pca) to be used for transforming the data. Default is None.
    feature_cols : list of str, optional
        The columns in the dataframe to be used as features for PCA. Default is None, which uses all answer columns.
    suffix : str, optional
        The suffix to be added to the new PCA columns in the dataframe. Default is None.

    Returns
    -------
    df : SVDataFrame
        The dataframe with the additional PCA columns.
    (imputer, scaler, pca) : tuple, optional
        The preprocessing objects used for transforming the data, if return_obj is True. Otherwise, returns only the dataframe.
    """

    df = df.copy()

    X = df.a() if feature_cols is None else df[feature_cols]
    if obj is None:
        imputer = SimpleImputer()
        X = imputer.fit_transform(X)
        scaler = StandardScaler if standardize else None
        X = scaler.fit_transform(X) if standardize else X
        pca = PCA(n_components=n_dims)
        pca_components = pca.fit_transform(X)
    else:
        imputer, scaler, pca = obj
        X = imputer.transform(X)
        X = scaler.transform(X) if scaler is not None else X
        pca_components = pca.transform(X)

    for i in range(n_dims):
        if suffix is not None:
            df[f'PCA_{i}_{suffix}'] = pca_components[:, i]
            df[f'-PCA_{i}_{suffix}'] = -pca_components[:, i]
        else:
            df[f'PCA_{i}'] = pca_components[:, i]
            df[f'-PCA_{i}'] = -pca_components[:, i]

    return (df, (imputer, scaler, pca)) if return_obj else df


def add_ca_cols(
        df: SVDataFrame,
        n_dims: int = 2,
        return_obj: bool = False,
        obj=None,
        feature_cols=None,
        suffix: str = None
):
    """
    Add Correspondence Analysis (CA) columns to the dataframe.

    Parameters
    ----------
    df : SVDataFrame
        The input dataframe containing the answers.
    n_dims : int, optional
        The number of dimensions to keep in the CA. Default is 2.
    return_obj : bool, optional
        Whether to return the preprocessing objects (imputer, ca). Default is False.
    obj : tuple, optional
        Pretrained preprocessing objects (imputer, ca) to be used for transforming the data. Default is None.
    feature_cols : list of str, optional
        The columns in the dataframe to be used as features for CA. Default is None, which uses all answer columns.
    suffix : str, optional
        The suffix to be added to the new CA columns in the dataframe. Default is None.

    Returns
    -------
    df : SVDataFrame
        The dataframe with the additional CA columns.
    (imputer, ca) : tuple, optional
        The preprocessing objects used for transforming the data, if return_obj is True. Otherwise, returns only the dataframe.
    """

    df = df.copy()

    X = df.a() if feature_cols is None else df[feature_cols]
    if obj is None:
        imputer = SimpleImputer()
        X = imputer.fit_transform(X)
        df_X = pd.DataFrame(X)
        ca = CA(n_components=n_dims)
        ca.fit(df_X)
        row_coordinates = ca.row_coordinates(df_X)
    else:
        imputer, ca = obj
        X = imputer.transform(X)
        df_X = pd.DataFrame(X)
        row_coordinates = ca.row_coordinates(df_X)

    for i in range(n_dims):
        if suffix is not None:
            df[f'CA_{i}_{suffix}'] = row_coordinates[i].values
            df[f'-CA_{i}_{suffix}'] = -row_coordinates[i].values
        else:
            df[f'CA_{i}'] = row_coordinates[i].values
            df[f'-CA_{i}'] = -row_coordinates[i].values

    return (df, (imputer, ca)) if return_obj else df


def add_tsvd_cols(
        df: SVDataFrame,
        n_dims: int = 2,
        standardize: bool = False,
        return_obj: bool = False,
        obj=None,
        feature_cols=None,
        suffix: str = None
):
    """
    Add TruncatedSVD (TSVD) columns to the dataframe.

    Parameters
    ----------
    df : SVDataFrame
        The input dataframe containing the answers.
    n_dims : int, optional
        The number of dimensions to keep in the TSVD. Default is 2.
    standardize : bool, optional
        Whether to standardize the data before performing TSVD. Default is False.
    return_obj : bool, optional
        Whether to return the preprocessing objects (imputer, scaler, tsvd). Default is False.
    obj : tuple, optional
        Pretrained preprocessing objects (imputer, scaler, tsvd) to be used for transforming the data. Default is None.
    feature_cols : list of str, optional
        The columns in the dataframe to be used as features for TSVD. Default is None, which uses all answer columns.
    suffix : str, optional
        The suffix to be added to the new TSVD columns in the dataframe. Default is None.

    Returns
    -------
    df : SVDataFrame
        The dataframe with the additional TSVD columns.
    (imputer, scaler, tsvd) : tuple, optional
        The preprocessing objects used for transforming the data, if return_obj is True. Otherwise, returns only the dataframe.
    """

    df = df.copy()

    X = df.a() if feature_cols is None else df[feature_cols]

    if obj is None:
        imputer = SimpleImputer()
        X = imputer.fit_transform(X)
        scaler = StandardScaler if standardize else None
        X = scaler.fit_transform(X) if standardize else X
        tsvd = TruncatedSVD(n_components=n_dims)
        tsvd_components = tsvd.fit_transform(X)
    else:
        imputer, scaler, tsvd = obj
        X = imputer.transform(X)
        X = scaler.transform(X) if scaler is not None else X
        tsvd_components = tsvd.transform(X)

    for i in range(n_dims):
        if suffix is not None:
            df[f'TSVD_{i}_{suffix}'] = tsvd_components[:, i]
            df[f'-TSVD_{i}_{suffix}'] = -tsvd_components[:, i]
        else:
            df[f'TSVD_{i}'] = tsvd_components[:, i]
            df[f'-TSVD_{i}'] = -tsvd_components[:, i]

    return (df, (imputer, scaler, tsvd)) if return_obj else df
