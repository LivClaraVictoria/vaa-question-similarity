import os
import re
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import statsmodels.api as sm
from plotly import express as px
from plotly import graph_objects as go
from plotly import io as pio
from scipy.stats import gaussian_kde
from sklearn.metrics import mutual_info_score
from statsmodels.stats.diagnostic import het_breuschpagan

# DEFAULT PLOT PARAMETERS FOR JUPYTER NOTEBOOKS

DEFAULT_HEIGHT = 800
DEFAULT_WIDTH = 1000


def calculate_entropy(column) -> float:
    """
    Calculate the entropy of a categorical column.

    Parameters
    ----------
    column : pandas.Series
        The input categorical column.

    Returns
    -------
    float
        The entropy value.
    """
    # Compute probability distribution
    value_counts = column.value_counts(normalize=True)

    # Calculate entropy
    entropy = -np.sum(value_counts * np.log2(value_counts))

    return entropy


def mutual_information_matrix(df, columns) -> pd.DataFrame:
    """
    Calculate the mutual information matrix between specified columns in a pandas DataFrame.

    Parameters:
        df (DataFrame): Input DataFrame.
        columns (list): List of column names for which mutual information is to be calculated.

    Returns:
        DataFrame: Mutual information matrix.
    """
    mutual_info_matrix = pd.DataFrame(index=columns, columns=columns)

    for col1 in columns:
        for col2 in columns:
            mutual_info = mutual_info_score(df[col1], df[col2])
            mutual_info_matrix.loc[col1, col2] = mutual_info

    return mutual_info_matrix


# count number of consecutive equal answers
def count_consecutive_values(arr) -> np.ndarray:
    """
    Count the maximum number of consecutive equal values in each row of a 2D NumPy array.

    Parameters
    ----------
    arr : numpy.ndarray
        The input 2D array.

    Returns
    -------
    numpy.ndarray
        An array containing the maximum number of consecutive equal values in each row of the input array.
    """

    # mask indicating if two consecutive values are the same
    consecutive_mask = np.equal(arr[:, :-1], arr[:, 1:])
    # counting for each row max number of consecutive values that are the same
    res = []
    for row in consecutive_mask:
        max_cnt = 0
        cnt = 0
        for el in row:
            if el:
                cnt += 1
            else:
                max_cnt = max(max_cnt, cnt)
                cnt = 0
        max_cnt = max(max_cnt, cnt)
        res.append(max_cnt + 1)

    return np.array(res)


def ensure_path(path) -> str:
    """
    Ensure that all directories on the specified path exist. If they don't, create them.

    Parameters
    ----------
    path : str
        The path to ensure.

    Returns
    -------
    str
        The path.
    """
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return path


def get_categorical_plot(col: pd.Series, **plot_kwargs) -> go.Figure:
    """
    Display the distribution of a categorical column of a pandas DataFrame using a Plotly figure.

    Parameters
    ----------
    col : pd.Series
        The categorical column of a pandas DataFrame.
    plot_kwargs : dict
        Additional keyword arguments to customize the layout of the figure.

    Returns
    -------
    go.Figure
        A Plotly figure displaying the distribution of the categorical column.
    """

    plot_kwargs = inject_default_kwargs(plot_kwargs, height_width=True, title='Categorical Distribution')

    vc = col.value_counts().sort_index()
    fig = px.bar(x=vc.index, y=vc.values)
    fig.update_layout(xaxis=dict(tickmode='array', tickvals=vc.index, ticktext=vc.index), **plot_kwargs)
    return fig


def get_distplot(df, val_col=None, group_col=None, nbins=50, **plot_kwargs) -> go.Figure:
    """
    Generate a distribution plot based on the DataFrame values.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    val_col : str, optional
        The name of the column containing the values to plot. If None, use all non-NaN values from the DataFrame.
    group_col : str, optional
        The name of the column to group by. If provided, separate distribution plots will be generated for each unique value in this column.
    nbins : int, optional
        The number of bins for the histogram. Default is 50.
    **plot_kwargs : dict
        Additional keyword arguments to be passed to the distribution plot function.

    Returns
    -------
    go.Figure
        The generated distribution plot figure.
    """
    group_labels = ['data'] if val_col is None else [val_col] if group_col is None else df[group_col].dropna().unique()
    hist_data = [df[~np.isnan(df)]] if val_col is None else [df[val_col].dropna().values] if group_col is None else [
        df[df[group_col] == l][val_col].dropna().values for l in group_labels]

    n_unique = len(np.unique(df[~np.isnan(df)])) if val_col is None else len(df[val_col].dropna().unique())
    nbins = nbins if n_unique > nbins else n_unique
    bin_size = (df[val_col].max() - df[val_col].min()) / nbins
    fig = ff.create_distplot(hist_data, group_labels, bin_size=bin_size, show_rug=False,
                             **{key: value for key, value in plot_kwargs.items() if key not in ['height', 'width']}) \
        .update_layout(
        height=DEFAULT_HEIGHT if 'height' not in plot_kwargs else plot_kwargs['height'],
        width=plot_kwargs.get('width', DEFAULT_WIDTH)
    )

    return fig


def lorenz_curve(a, include_diagonal: bool = True, include_gini: bool = True, **plot_kwargs) -> go.Figure:
    """
    Generate a Lorenz curve plot.

    Parameters
    ----------
    a : array-like
        Input array containing the values.
    include_diagonal : bool, optional
        Whether to include the diagonal line of equality. Default is True.
    include_gini : bool, optional
        Whether to include the Gini coefficient in the plot title. Default is True.
    **plot_kwargs : dict
        Additional keyword arguments for customizing the plot.

    Returns
    -------
    go.Figure
        Plotly figure object representing the Lorenz curve.
    """

    if isinstance(a, pd.Series):
        a = a.values
    elif not isinstance(a, np.ndarray):
        a = np.array(a)

    # drop NaN values
    a = a[~np.isnan(a)]

    plot_kwargs = inject_default_kwargs(
        plot_kwargs,
        height_width=True,
        title=f'Lorenz Curve (Gini={gini_coefficient(a):.2f})' if include_gini else 'Lorenz Curve',
        markers=len(a) < 100
    )

    x_range = np.arange(len(a) + 1) / len(a)

    fig = px.area(x=x_range, y=np.cumsum(np.sort(np.concatenate((np.array([0]), a)))) / np.sum(a), **plot_kwargs)
    # fig.update_layout(xaxis_title='Cumulative Share of Elements (from lowest to highest value)', yaxis_title='Cumulative Share of Value')
    fig.update_layout(xaxis_title='Fraction of Least Recommended Candidates',
                      yaxis_title='Cumulative Share of Recommendations')

    if include_diagonal:
        fig.add_traces(
            px.line(x=x_range, y=np.linspace(0, 1, num=len(a) + 1)).update_traces(line=dict(color='black')).data)

    return fig


def gini_coefficient(a: list[float]) -> float:
    """
    Calculate the Gini coefficient for a given array of values.

    Parameters
    ----------
    a : array-like
        Input array containing the values.

    Returns
    -------
    float
        The Gini coefficient.

    Notes
    -----
    This function provides a concise implementation of the Gini coefficient calculation.
    However, it has a time and memory complexity of O(n**2), where n is the length of the input array.
    Avoid passing in large samples to prevent performance issues.
    """

    mean_absolute_difference = np.abs(np.subtract.outer(a, a)).mean()

    relative_mean_absolute_difference = mean_absolute_difference / np.mean(a)

    gini_coef = 0.5 * relative_mean_absolute_difference

    return gini_coef


def get_cols(df_or_list: pd.DataFrame | list[str], col_type: str = 'answer', verbose: bool = False) -> list[str]:
    """
    Get the names of the columns that contain answers.

    Returns
    -------
    df_or_list : pd.DataFrame or list of str
        The input DataFrame or list of column names.
    col_type : str
        The type of columns to filter. Default is 'answer'.
    verbose : bool, optional
        Whether to print a warning if the number of answer columns is not 75. Default is True.
    """

    # check if col_type is one of answer, weight, cleavage
    assert col_type in ['answer', 'weight',
                        'cleavage'], f"col_type has to be one of 'answer', 'weight', 'cleavage'. Provided: {col_type}"

    if isinstance(df_or_list, pd.DataFrame):
        df_or_list = df_or_list.columns

    pattern = r'^answer_\d+$' if col_type == 'answer' else r'^weight_\d+$' if col_type == 'weight' else r'^cleavage_[1-8]$'

    # Filter the list using the regular expression pattern
    filtered_input = [s for s in df_or_list if re.match(pattern, str(s))]

    # warn user if len(filtered_input) is not 75
    if verbose:
        if col_type in ['answer', 'weight'] and len(filtered_input) != 75:
            print(f"Warning: Found {len(filtered_input)} {col_type} columns. Expected 75.")
        elif col_type == 'cleavage' and len(filtered_input) != 8:
            print(f"Warning: Found {len(filtered_input)} {col_type} columns. Expected 8.")

    return sorted(filtered_input, key=lambda x: int(x.split('_')[-1]))


def update_question_type2idx(question_type2idx: dict[int, list[int]], nan_mask: np.ndarray) -> dict[int, list[int]]:
    if question_type2idx is None:
        return None

    sliced_columns = np.where(nan_mask)[0]
    updated_question_type2idx = {}
    for category, indices in question_type2idx.items():
        updated_indices = [idx - sum(idx > col for col in sliced_columns) for idx in indices if
                           idx not in sliced_columns]
        updated_question_type2idx[category] = updated_indices
    return updated_question_type2idx


def remove_old_cache_files(cache_path: str) -> None:
    """
    Remove old cache files in the same directory as the specified cache path.

    Parameters
    ----------
    cache_path : str
        Path to the cache file.

    Returns
    -------
    None
    """

    cache_dir = os.path.dirname(cache_path)
    cache_filename = os.path.basename(cache_path)
    stem = '-'.join(cache_filename.split('-')[:-1])

    # remove old cache files
    for f in os.listdir(cache_dir):
        if f != cache_filename and re.match(fr'^({stem}-[0-9a-f]{{64}})\.parquet$', f):
            os.remove(os.path.join(cache_dir, f))


def get_ols_pvalue(df, x_col, y_col, cov_type: str = 'nonrobust'):
    """
    Calculate the p-value for the OLS regression between two columns in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    x_col : str
        The name of the column containing the independent variable.
    y_col : str
        The name of the column containing the dependent variable.
    cov_type : str, optional
        The covariance type to use for the OLS regression. Default is 'nonrobust'.

    Returns
    -------
    float
        The p-value for the OLS regression.
    """

    return get_ols_model(df, x_col, y_col, cov_type=cov_type).pvalues[x_col]


def get_95_ci(df, x_col, y_col, cov_type: str = 'nonrobust'):
    """
    Calculate the 95% confidence interval for the OLS regression between two columns in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    x_col : str
        The name of the column containing the independent variable.
    y_col : str
        The name of the column containing the dependent variable.
    cov_type : str, optional
        The covariance type to use for the OLS regression. Default is 'nonrobust'.

    Returns
    -------
    tuple
        A tuple containing the lower and upper bounds of the 95% confidence interval.
    """

    return get_ols_model(df, x_col, y_col, cov_type=cov_type).conf_int().loc[x_col].values


def get_ols_model(df, x_col, y_col, cov_type: str = None):
    """
    Fit an OLS regression model to the data in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    x_col : str
        The name of the column containing the independent variable.
    y_col : str
        The name of the column containing the dependent variable.
    cov_type : str, optional
        The covariance type to use for the OLS regression. Default is None.

    Returns
    -------
    statsmodels.regression.linear_model.RegressionResultsWrapper
        The OLS regression model.
    """

    df = df.dropna(subset=[x_col, y_col])

    X = df[x_col]
    X = sm.add_constant(X)
    y = df[y_col]

    model = sm.OLS(y, X).fit(cov_type=cov_type if cov_type is not None else 'nonrobust')

    if cov_type is None:
        # Perform Breusch-Pagan test
        bp_test = het_breuschpagan(model.resid, model.model.exog)

        # If the p-value is less than 0.05, reject the null hypothesis and use 'HC3' covariance type
        if bp_test[1] < 0.05:
            warnings.warn(f"Breusch-Pagan test p-value: {bp_test[1]:.2e}. Using 'HC3' covariance type.")
            model = sm.OLS(y, X).fit(cov_type='HC3')

    return model


def plot_relationship(
        df,
        x,
        y,
        trendline: str = 'ols',
        window_size_frac: int = 20,
        title_prefix: str = None,
        xaxis_title: str = None,
        yaxis_title: str = None,
        precision: int = 2,
        **plot_kwargs
):
    """
    Generate a scatter plot of the relationship between two columns in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    x : str
        The name of the column containing the independent variable.
    y : str
        The name of the column containing the dependent variable.
    trendline : str, optional
        The type of trendline to add to the plot. Default is 'ols'.
    window_size_frac : int, optional
        The fraction of the DataFrame length to use as the window size for the rolling trendline. Default is 20.
    title_prefix : str, optional
        The prefix to add to the plot title. Default is None.
    xaxis_title : str, optional
        The title for the x-axis. Default is None.
    yaxis_title : str, optional
        The title for the y-axis. Default is None.
    precision : int, optional
        The number of decimal places to display in the p-value and confidence interval. Default is 2.
    **plot_kwargs : dict
        Additional keyword arguments to customize the plot.

    Returns
    -------
    go.Figure
        A Plotly figure object representing the scatter plot.
    """

    df = df.dropna(subset=[x, y])

    model = get_ols_model(df, x, y)
    ci_95 = model.conf_int().loc[x].values
    p_value = model.pvalues[x]

    default_title = f"{title_prefix + ' ' if title_prefix is not None else ''}(p value: {p_value:.{precision}e}, 95% CI: [{ci_95[0]:.{precision}f}, {ci_95[1]:.{precision}f}])"

    plot_kwargs = inject_default_kwargs(
        plot_kwargs,
        height_width=True,
        trendline=trendline,
        trendline_options=dict(window=len(df) // window_size_frac, center=True) if trendline == 'rolling' else None,
        trendline_color_override='red',
        title=default_title
    )

    fig = px.scatter(df, x=x, y=y, **plot_kwargs)
    fig.update_layout(
        xaxis_title=xaxis_title if xaxis_title is not None else x,
        yaxis_title=yaxis_title if yaxis_title is not None else y
    )

    return fig


def plot_answer_strength_recommendations(
        df,
        answer_strength_col: str = '_answer_strength',
        spc: bool = True,
        normalized: bool = True,
        method: str = 'L2_sv',
        trendline: str = 'ols',
        window_size_frac: int = 20,
        diff: bool = False,
        **plot_kwargs
) -> go.Figure:
    """
    Generate a scatter plot of candidate recommendations based on candidate answer strength.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing candidate data.
    answer_strength_col : str, optional
        The name of the column containing the answer strength values. Default is '_answer_strength'.
    spc : bool, optional
        If True, use 'spc' method for recommendations, otherwise use 'top'. Default is True.
    normalized : bool, optional
        If True, normalize the recommendation scores. Default is False.
    method : str, optional
        The method to use for recommendations. Default is 'L2'.
    trendline : str, optional
        The type of trendline to add to the plot. Default is 'ols'.
    window_size_frac : int, optional
        The fraction of the DataFrame length to use as the window size for the rolling trendline. Default is 20.
    diff : bool, optional
        If True, plot the difference between the recommendation scores and the answer strength. Default is False.
    **plot_kwargs : dict
        Additional keyword arguments to customize the plot.

    Returns
    -------
    go.Figure
        A Plotly figure object representing the scatter plot.

    """
    df = df.copy()
    y_col = f"_n_recommendations_{'spc' if spc else 'top'}_{method}{'_normalized' if normalized else ''}{'_diff' if diff else ''}"

    if 'log_y_before' in plot_kwargs and plot_kwargs['log_y_before']:
        df[y_col] = np.log(df[y_col])
        plot_kwargs.pop('log_y_before')

    fig = plot_relationship(
        df,
        x=f"{answer_strength_col}{'_diff' if diff else ''}",
        y=f"_n_recommendations_{'spc' if spc else 'top'}_{method}{'_normalized' if normalized else ''}{'_diff' if diff else ''}",
        trendline=trendline,
        window_size_frac=window_size_frac,
        title_prefix="Answer Strengths vs Recommendations",
        xaxis_title=f"Candidate Answer Strength{' Difference' if diff else ''}",
        yaxis_title=f"Candidate Recommendations {' Difference' if diff else ''}{' (Normalized)' if normalized else ''}",
        **plot_kwargs,
    )

    return fig


def inject_default_kwargs(kwargs, height_width: bool = False, **default_kwargs):
    """
    Injects default keyword arguments into a dictionary of keyword arguments.

    Parameters
    ----------
    kwargs : dict
        A dictionary containing keyword arguments.
    height_width : bool, optional
        Whether to inject default height and width values. Default is False.
    **default_kwargs : dict
        Default keyword arguments to be injected into `kwargs` if they are not already present.

    Returns
    -------
    dict
        A dictionary with default keyword arguments injected.
    """
    kwargs = kwargs.copy()
    if height_width:
        # if height and width are not provided, use default values
        default_kwargs['height'] = default_kwargs.get('height', DEFAULT_HEIGHT)
        default_kwargs['width'] = default_kwargs.get('width', DEFAULT_WIDTH)
    # inject default kwargs
    for k, v in default_kwargs.items():
        kwargs[k] = kwargs.get(k, v)
    return kwargs


def plot_histogram_kde(
        df: pd.DataFrame | list,
        value_column: str = None,
        group_column: str = None,
        sort_groups: bool = True,
        kde_padding: float = 0.1,
        n_kde_positions: int = 100,
        **plot_kwargs
) -> go.Figure:
    """
    Generate a histogram and kernel density estimate (KDE) plot for a given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame, list
        The input DataFrame or Iterable.
    value_column : str, optional
        The name of the column in the DataFrame to plot.
    group_column : str, optional
        The name of the column to group by. If provided, separate histograms and KDEs will be generated for each unique value in this column. Default is None.
    sort_groups : bool, optional
        Whether to sort the groups alphabetically. Default is False.
    kde_padding : float, optional
        The padding to add to the KDE x-axis limits. Default is 0.1.
    n_kde_positions : int, optional
        The number of positions at which the KDE is evaluated. Default is 100.
    **plot_kwargs : dict
        Additional keyword arguments to customize the plot.

    Returns
    -------
    go.Figure
        A Plotly figure object representing the histogram and KDE plot.

    """

    if not isinstance(df, pd.DataFrame):
        # convert list to DataFrame
        df = pd.DataFrame({'values': df})
        value_column = 'values'

    if group_column is not None:
        if sort_groups:
            groups = sorted(df[group_column].unique())
            plot_kwargs = inject_default_kwargs(plot_kwargs, category_orders={group_column: groups})
        else:
            groups = df[group_column].unique()

    # Create a Plotly figure
    plot_kwargs = inject_default_kwargs(plot_kwargs, height_width=True, histnorm='probability density',
                                        barmode='overlay')
    fig = px.histogram(df, x=value_column, color=group_column, **plot_kwargs)

    # calculate KDE for each group (if grouping column is provided)
    if group_column:
        df_groups_kde = defaultdict(list)
        for group in groups:
            group = str(group)
            group_data = df[df[group_column] == group][value_column].values
            group_data = group_data[~np.isnan(group_data)]
            # calculate kde
            try:
                kde = gaussian_kde(group_data)
            except Exception as e:
                # warn user about the error using warnings module
                warnings.warn(f"Error calculating KDE for group {group}: {e}")
                continue
            data_spread = group_data.max() - group_data.min()
            x_values = np.linspace(group_data.min() - kde_padding * data_spread,
                                   group_data.max() + kde_padding * data_spread, n_kde_positions)
            y_values = kde(x_values)
            # add values to df
            df_groups_kde['group'].extend([group] * n_kde_positions)
            df_groups_kde['x'].extend(list(x_values))
            df_groups_kde['y'].extend(list(y_values))
        # add kde traces
        df_groups_kde = pd.DataFrame(df_groups_kde)
        fig.add_traces(px.line(df_groups_kde, x='x', y='y', color="group").data)
    else:
        # calculate KDE for the entire data
        data = df[value_column].values
        data = data[~np.isnan(data)]
        kde = gaussian_kde(data)
        data_spread = df[value_column].max() - df[value_column].min()
        x_values = np.linspace(df[value_column].min() - kde_padding * data_spread,
                               df[value_column].max() + kde_padding * data_spread, n_kde_positions)
        y_values = kde(x_values)
        fig.add_traces(px.line(x=x_values, y=y_values).data)

    return fig


def find_first_occurrence(array: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Find the first occurrence of each label in the array.

    This function searches for the first occurrence of each label in the input array and returns an array
    containing the indices of the first occurrence of each label. If a label is not found, the corresponding
    index will be NaN.

    Parameters
    ----------
    array : np.ndarray
        The input array in which to search for labels.
    labels : np.ndarray
        The labels to search for in the array.

    Returns
    -------
    np.ndarray
        An array containing the indices of the first occurrence of each label. If a label is not found, the
        corresponding index will be NaN.

    Notes
    -----
    - If `labels` is a 1D array, it will be reshaped to a 2D array with one column.
    - The function uses `np.argmax` to find the first occurrence of each label.
    """

    first_occurrences = np.full(len(array), np.nan)
    # check if labels are of shape (n, ) and reshape them to (n, 1)
    if len(labels.shape) == 1:
        labels = labels.reshape(-1, 1)
    matches_mask = np.any(array == labels, axis=1)
    first_occurrences[matches_mask] = np.argmax(array[matches_mask] == labels[matches_mask], axis=1)
    return first_occurrences


def get_px_heatmap(data: np.ndarray | pd.DataFrame, row_names=None, column_names=None, colorscale='YlGnBu'):
    if isinstance(data, pd.DataFrame):
        row_names = data.index if row_names is None else row_names
        column_names = data.columns if column_names is None else column_names
        data = data.values
    else:
        row_names = np.arange(data.shape[0]) if row_names is None else row_names
        column_names = np.arange(data.shape[1]) if column_names is None else column_names

    fig = go.Figure(data=go.Heatmap(
        z=data,
        x=column_names,
        y=row_names,
        colorscale=colorscale))

    fig.update_layout(
        xaxis=dict(showticklabels=False),  # Hide x-axis ticks
        yaxis=dict(showticklabels=False),  # Hide y-axis ticks
        height=1000,
        width=1000
    )

    return fig


def save_paper_figure(
        fig,
        filename: str,
        big: bool = False,
        font_family: str = 'Times',
        font_size: int = 18,
        **layout_kwargs
) -> go.Figure:
    """
    Save a Plotly figure with specific layout settings for paper publication.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        The Plotly figure to be saved.
    filename : str
        The name of the file to save the figure as (without extension).
    big : bool, optional
        If True, save the figure with larger dimensions. Default is False.
    font_family : str, optional
        The font family to use for the figure text. Default is 'Times'.
    font_size : int, optional
        The font size to use for the figure text. Default is 18.
    **layout_kwargs : dict, optional
        Additional keyword arguments for customizing the figure layout.

    Returns
    -------
    plotly.graph_objects.Figure
        The updated Plotly figure with the applied layout settings.

    Notes
    -----
    - The figure is saved as a PNG file in the 'images' directory.
    - The default dimensions are 600x360 for small figures and 1200x720 for large figures.
    - The default margins are set to provide adequate spacing for the figure elements.
    - The font color is set to black by default.
    """

    layout_kwargs = inject_default_kwargs(
        layout_kwargs,
        height_width=False,
        title=None,
        template='none',
        width=600 if not big else 1200,
        height=360 if not big else 720,
        margin_l=70, margin_t=0, margin_b=60, margin_r=10,
        font=dict(
            family=font_family,  # Set the font family (Times Roman or Libertinus Sans)
            size=font_size,  # Set the font size for all text elements
            color="black"  # Set the font color (optional)
        ),
    )

    fig.update_layout(
        **layout_kwargs
    )

    pio.write_image(fig, os.path.join('images', f"{filename}.png"), scale=3)

    return fig


def plot_cdf(data, column=None, color=None, dropna=True, **px_kwargs):
    """
    Produces a Plotly CDF plot of a pandas DataFrame column or any iterable.
    If a color column is specified, creates separate CDFs for each unique value in the color column.

    Parameters
    ----------
    data : pandas.DataFrame or iterable
        The input data. If a DataFrame is provided, specify the column to plot.
    column : str, optional
        The name of the column to plot if `data` is a DataFrame. Default is None.
    color : str, optional
        The name of the column to use for color grouping if `data` is a DataFrame. Default is None.
    dropna : bool, optional
        If True, drop NaN values from the data before plotting. Default is True.
    **px_kwargs : keyword arguments
        Additional keyword arguments to pass to `plotly.express.line`.

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        The Plotly Figure object for the CDF plot.

    Raises
    ------
    ValueError
        If `data` is a DataFrame and `column` is not specified.
    """
    if isinstance(data, pd.DataFrame):
        if column is None:
            raise ValueError("Column name must be specified when data is a DataFrame.")
        if color is not None:
            grouped_data = data[[column, color]].copy()
        else:
            grouped_data = data[[column]].copy()
        values = grouped_data[column]
    else:
        grouped_data = pd.DataFrame({'Values': data})
        values = grouped_data['Values']

    # Drop NaN values if dropna is True
    if dropna:
        grouped_data = grouped_data.dropna()

    # Prepare data for plotting
    if color:
        cdf_list = []
        for val in grouped_data[color].unique():
            subset = grouped_data[grouped_data[color] == val][column]
            sorted_values = np.sort(subset)
            cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
            temp_df = pd.DataFrame({column: sorted_values, 'CDF': cdf, color: val})
            cdf_list.append(temp_df)
        cdf_df = pd.concat(cdf_list, ignore_index=True)
    else:
        sorted_values = np.sort(values)
        cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
        cdf_df = pd.DataFrame({column: sorted_values, 'CDF': cdf})

    # Create the Plotly figure
    fig = px.line(cdf_df, x=column, y='CDF', color=color, title='CDF Plot',
                  labels={column: 'Value', 'CDF': 'Cumulative Probability'}, **px_kwargs)

    return fig


def add_matching_percentages(df_voters, correct: bool = False):
    df_voters = df_voters.copy()

    max_dists = df_voters['_maxDist'] if not correct else df_voters['_maxDistCorrect']
    for dist_col in [c for c in df_voters.columns if c.startswith('_matchDist')]:
        df_voters[dist_col.replace('_matchDist', '_matchPerc')] = 1 - (df_voters[dist_col] / max_dists)

    return df_voters
