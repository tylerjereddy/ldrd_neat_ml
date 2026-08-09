from typing import Optional

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from skimage import measure 
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances_argmin_min
from matplotlib.ticker import MaxNLocator
from matplotlib.axes import Axes

# TODO: fix inconsistencies in docstring formatting (issue #12)
class GMMWrapper:
    """
    A wrapper to use a GMM trained on feature data as a 
    classifier in composition space.

    This class bridges the gap between a GMM trained on 
    1D phase data and the 2D composition space of a phase 
    diagram. For any given composition point (x, y), it 
    finds the nearest experimental data point in the 
    composition space and uses that point's phase to predict
    a cluster label with the GMM.

    Attributes:
        gmm (GaussianMixture): The pre-trained Gaussian 
                               Mixture Model.
        x_comp (np.ndarray): The 2D array of composition
                             data (n_samples, 2).
        x_features (np.ndarray): The corresponding 
                                 feature vectors used for 
                                 GMM training.
    """

    def __init__(
        self,
        gmm: GaussianMixture,
        x_comp: np.ndarray,
        x_features: np.ndarray
    ):
        """
        Initializes the GMMWrapper.

        Parameters:
        ----------
            gmm (GaussianMixture): The trained GMM instance.
            x_comp (np.ndarray): The composition data array 
                                 (n_samples, 2).
            x_features (np.ndarray): The feature data array 
                                     (n_samples, 1).
        """
        self.gmm = gmm
        self.x_comp = x_comp
        self.x_features = x_features

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        """
        Predicts cluster labels for new composition points.

        Parameters:
        -----------
            x_test (np.ndarray): An array of new composition
                                 points (n_test_samples, 2).

        Returns:
        -------
            labels (np.ndarray): The predicted GMM cluster labels 
                        for the input points.
        """
        closest_indices, _ = pairwise_distances_argmin_min(
            x_test, 
            self.x_comp
            )
        feature_vectors = self.x_features[closest_indices]
        labels = self.gmm.predict(feature_vectors)
        return labels


def extract_boundary_from_contour(
    z: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    level: float
) -> Optional[np.ndarray]:
    """
    Extracts the longest boundary contour from a grid at 
    a specified level.

    Parameters:
    ----------
        z (np.ndarray): The 2D grid of predicted values.
        xs (np.ndarray): The x-coordinates corresponding
                         to the grid columns.
        ys (np.ndarray): The y-coordinates corresponding 
                         to the grid rows.
        level (float): The contour level to extract.

    Returns:
    --------
        boundary_points (Optional[np.ndarray]): 
            An array of (x, y) coordinates for the boundary,
            or None if no contour is found.
    """
    contours = measure.find_contours(z, level)
    if not contours:
        return None
    longest_contour = max(contours, key=len)
    resolution_x, resolution_y = len(xs), len(ys)
    boundary_points = np.column_stack((
        np.interp(longest_contour[:, 1], 
                  np.arange(resolution_x), xs),
        np.interp(longest_contour[:, 0], 
                  np.arange(resolution_y), ys)
    ))
    return boundary_points

def _standardise_labels(
    cluster_labels: np.ndarray, x_comp: np.ndarray
) -> tuple[np.ndarray, dict[int, int]]:
    """
    Remap raw GMM labels so conventions never flip.

    *Standard convention used downstream*

    ---------  ----------------------------
    label=0    Two Phase   (orange triangle, light-steel-blue background)
    label=1    Single Phase (blue square, turquoise background)
    ---------  ----------------------------

    Parameters
    ----------
    cluster_labels :
        Original labels from :pyclass:`sklearn.mixture.GaussianMixture`.
    x_comp :
        Composition coordinates associated with each label.

    Returns
    -------
    std_labels :
        Relabelled array following the convention above.
    label_map :
        Dict mapping raw → standard labels; apply to any
        future predictions (`z` grids, etc.).
    """
    distances = {
        lbl: np.linalg.norm(x_comp[cluster_labels == lbl].mean(axis=0))
        for lbl in np.unique(cluster_labels)
    }
    
    near_lbl = min(distances, key=distances.get)  # type: ignore[arg-type]
    far_lbl = max(distances, key=distances.get)  # type: ignore[arg-type]

    label_map = {far_lbl: 0, near_lbl: 1}
    std_labels = np.vectorize(label_map.get)(cluster_labels)
    return std_labels, label_map

def _set_axis_style(ax, xrange: list[int] | None, yrange: list[int] | None):
    """
    Apply consistent styling *and* force identical tick-spacing & integer labels.
    """
    ax.set_xlim(xrange)
    ax.set_ylim(yrange)
    ax.set_aspect("auto")

    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.tick_params(
        direction="in",
        length=5,
        width=2,
        which="both",
        top=True,
        right=True,
        labelsize=34,
        pad=20,
    )
    # set consistent use of integer labels
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    # set ``bold`` tick mark font weight
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_weight('bold')

def plot_gmm_decision_regions(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    phase_col: str,
    ax: Axes,
    xrange: list[int],
    yrange: list[int],
    n_components: int,
    random_state: int,
    boundary_color: str,
    resolution: int,
    decision_alpha: float,
    plot_regions: bool,
    region_colors: tuple[str, str] = ("lightsteelblue", "aquamarine"),
) -> tuple[GaussianMixture, np.ndarray, Optional[np.ndarray]]:
    """
    Trains a GMM and creates contour traces for phase 
    regions and boundaries.

    Parameters:
    -----------
        df (pd.DataFrame): DataFrame with composition and
                           phase data.
        x_col (str): Name of the column for the x-axis 
                     component.
        y_col (str): Name of the column for the y-axis 
                     component.
        phase_col (str): Name of the column containing 
                         phase information.
        ax (Axes): matplotlib axis for plotting
        xrange (list[int]): X-axis composition range.
        yrange (list[int]): Y-axis composition range. 
        n_components (int): Number of GMM components.
        random_state (int): Seed for reproducibility.
        region_colors (tuple[str, str]): Colors for the 
                                             phase regions.
        boundary_color (str): Color for the boundary line.
        resolution (int): Grid resolution for the contour plot.
        decision_alpha (float): Opacity of the filled regions.
        plot_regions (bool): Whether to plot the filled regions.

    Returns:
    --------
        gmm, std_labels, boundary_points (tuple): 
               Output tuple containing the GMM model, cluster labels, 
               and boundary points
    """
    df_local, x_col = rename_df_columns(df, x_col)
    x_features = df_local[phase_col].to_numpy().reshape(-1, 1)

    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    raw_labels = gmm.fit_predict(x_features)
    x_comp = df_local[[x_col, y_col]].to_numpy()
    std_labels, label_map = _standardise_labels(raw_labels, x_comp)
    wrapper = GMMWrapper(gmm, x_comp, x_features)
    xs = np.linspace(*xrange, resolution, dtype=float)  # type: ignore[call-overload]
    ys = np.linspace(*yrange, resolution, dtype=float)  # type: ignore[call-overload]
    xx, yy = np.meshgrid(xs, ys)
    z_raw = wrapper.predict(np.c_[xx.ravel(), yy.ravel()])
    z = np.vectorize(label_map.get)(z_raw).reshape(xx.shape)

    if plot_regions:
        ax.contourf(
            xx,
            yy,
            z,
            levels=[-0.5, 0.5, 1.5],
            colors=region_colors,
            alpha=decision_alpha,
        )

    ax.contour(
        xx,
        yy,
        z,
        levels=[0.5],
        colors=boundary_color,
        linewidths=3,
    )
    proxy_artist = Line2D(
        [0], 
        [0], 
        label="Decision Boundary", 
        color=boundary_color, 
        linewidth=3
    )
    ax.legend(handles=[proxy_artist])

    boundary_points = extract_boundary_from_contour(z, xs, ys, level=0.5)
    return gmm, std_labels, boundary_points

def plot_gmm_composition_phase(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    phase_col: str,
    ax: Axes,
    point_cmap: tuple[str, str] = ("#FF8C00", "dodgerblue"),
) -> None:
    """
    Trains a GMM and creates a scatter trace for data points.

    Parameters:
    -----------
        df (pd.DataFrame): DataFrame with composition and 
                           phase data.
        x_col (str): Name of the column for the x-axis 
                     component.
        y_col (str): Name of the column for the y-axis 
                     component.
        phase_col (str): Name of the column containing 
                         phase information.
        ax (Axes): matplotlib axis for plotting
        point_cmap (tuple[str, str]): Colormap for the
                                          scatter points.
    """
    x_features = df[phase_col].to_numpy().reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=42)
    raw_labels = gmm.fit_predict(x_features)
    df, x_col = rename_df_columns(df, x_col)
    
    x_comp = df[[x_col, y_col]].to_numpy()
    std_labels, _ = _standardise_labels(raw_labels, x_comp)
    mask_two = std_labels == 0
    ax.scatter(
        x_comp[mask_two, 0], x_comp[mask_two, 1],
        marker="^", c=point_cmap[0],
        s=250, edgecolors="black", linewidths=1,
        label="Two Phase (Experiment)",                       
    )

    mask_single = std_labels == 1
    ax.scatter(
        x_comp[mask_single, 0], x_comp[mask_single, 1],
        marker="s", c=point_cmap[1],
        s=250, edgecolors="black", linewidths=1,
        label="Single Phase (Experiment)",
    )

def rename_df_columns(
    df: pd.DataFrame,
    in_col: str,
) -> tuple[pd.DataFrame, str]:
    """
    rename dataframe columns from "raw" `.csv` files
    for plotting manuscript figures
    
    Parameters:
    -----------
    df : pd.DataFrame
        dataframe of "raw" phase data
    in_col : str
        column string to replace in dataframe
    
    Returns:
    --------
    df_local : pd.DataFrame
        dataframe of "raw" data with new column names
        for manuscript plots
    out_col : str
        the modified column variable name for plotting
        axis titles
    """
    df_local = pd.DataFrame(df.copy())
    if "PEO" in in_col:
        out_col = in_col.replace("PEO", "PEG")
    elif in_col == "Dextran 9 - 11 kg/mol (wt%)":
        out_col = "Dextran 10 kg/mol (wt%)"
    elif in_col == "Dextran 450 - 650 kg/mol (wt%)":
        out_col = "Dextran 500 kg/mol (wt%)"
    elif in_col == "Sodium citrate (wt%)": 
        out_col = "Sodium Citrate (wt%)"
    else:
        return df_local, in_col
    df_local.rename(columns={in_col: out_col}, inplace=True)
    return df_local, out_col
