# Copyright (c) 2024-2025 Lucas Leão
# tinyshift - A small toolbox for mlops
# Licensed under the MIT License


import numpy as np
import plotly.graph_objects as go
from sklearn.calibration import calibration_curve
from scipy.stats import beta
from sklearn import metrics
import plotly.figure_factory as ff
import plotly.express as px
from sklearn.base import ClassifierMixin
from typing import List


def efficiency_curve(
    clf: ClassifierMixin,
    X: np.ndarray,
    fig_type=None,
    width=800,
    height=400,
):
    """
    Generates an efficiency and validity curve for a conformal prediction classifier.

    This function creates an interactive visualization showing the trade-off between
    efficiency and validity across different error rates (significance levels) in
    conformal prediction. The efficiency measures how often the prediction sets
    contain exactly one class (singleton sets), while validity measures the
    coverage guarantee - the proportion of prediction sets that contain the true label.

    Parameters
    -----------
    clf : ClassifierMixin
        A trained conformal prediction classifier that implements the predict_set method.
        The classifier should be able to generate prediction sets at different
        confidence levels (alpha parameters).
    X : np.ndarray
        Input feature data for evaluation.
    fig_type : str, optional
        Display type for the figure (particularly useful in Jupyter notebooks).
        Common options: None (default), 'png', 'svg', 'browser', or other
        Plotly-supported renderers. (default: None)
    width : int, optional
        Figure width in pixels (default: 800).
    height : int, optional
        Figure height in pixels (default: 400).

    Returns
    --------
    None
        Displays the efficiency and validity curve plot directly.

    Notes
    ------
    - **Efficiency**: Measures the proportion of prediction sets that are singletons
      (contain exactly one class). Higher efficiency means more decisive predictions.
    - **Validity**: Measures the empirical coverage rate - the proportion of
      prediction sets that contain the true label. Should ideally match (1 - α).
    - The curves show how these metrics change across different error rates
      (0.05 to 0.45), helping assess the calibration quality of the conformal predictor.
    - Well-calibrated conformal predictors should show validity close to the
      theoretical guarantee while maximizing efficiency.
    """

    def get_error_metrics(clf: ClassifierMixin, X: np.ndarray) -> List:
        """
        Calculates error metrics for different error rates.

        Args:
            clf (object): The classifier model.
            X (np.ndarray): Input data.

        Returns:
            Tuple: Arrays for error_rate, efficiency_rate, and validity_rate.
        """

        error_rate = np.asarray([0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05])
        efficiency_rate = np.zeros(error_rate.shape)
        validity_rate = np.zeros(error_rate.shape)
        for i, error in enumerate(error_rate):
            predict_set = clf.predict_set(X, alpha=error)
            efficiency_rate[i] = np.sum([np.sum(p) == 1 for p in predict_set]) / len(
                predict_set
            )
            validity_rate[i] = np.sum(predict_set) / len(predict_set)
        return error_rate, efficiency_rate, validity_rate

    if not hasattr(clf, "predict_set"):
        raise ValueError(
            "The classifier must implement the 'predict_set' method for conformal predictions."
        )

    X = np.asarray(X)

    error_rate, efficiency_rate, validity_rate = get_error_metrics(clf, X)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=error_rate,
            y=efficiency_rate,
            mode="lines+markers",
            name="efficicency",
            line=dict(color="darkblue"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=error_rate,
            y=validity_rate,
            mode="lines+markers",
            name="validity",
            line=dict(color="orange"),
        )
    )

    fig.update_layout(
        title="Efficiency & Validity Curve",
        xaxis_title="Error Rate",
        yaxis_title="Score",
        legend=dict(title="Metric"),
        width=width,
        height=height,
    )
    fig.update_layout(hovermode="x")
    fig.update_traces(hovertemplate="%{y}")
    return fig.show(fig_type)


def reliability_curve(
    clf: ClassifierMixin,
    X: np.ndarray,
    y: np.ndarray,
    model_name: str = "Model",
    n_bins=15,
    fig_type=None,
) -> go.Figure:
    """
    Generates a reliability curve (calibration curve) for a binary classifier.

    The reliability curve plots the true probability of the positive class against
    the mean predicted probability in each bin. A perfectly calibrated classifier
    would follow the diagonal line.

    Parameters
    -----------
    clf : ClassifierMixin
        The trained classifier model that implements predict_proba.
    X : np.ndarray
        Input feature data for evaluation.
    y : np.ndarray
        True binary labels (0 or 1).
    model_name : str, optional
        Name to display in the legend (default: "Model").
    n_bins : int, optional
        Number of bins for the reliability curve (default: 15).
    fig_type : str, optional
        Renderer for fig.show() (e.g., 'png', 'svg', 'browser').
        Defaults to None.

    Returns
    --------
    None
        Displays the reliability curve plot directly.

    Notes
    ------
    - Uses quantile strategy for binning to ensure equal-sized bins
    - The diagonal line represents perfect calibration
    - Deviations from the diagonal indicate miscalibration
    """

    if not hasattr(clf, "predict_proba"):
        raise ValueError("The classifier must implement the 'predict_proba' method.")
    if len(np.unique(y)) != 2:
        raise ValueError("This function only supports binary classification.")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")

    y_prob = clf.predict_proba(X)[:, 1]

    v_prob_true, v_prob_pred = calibration_curve(
        y, y_prob, n_bins=n_bins, strategy="quantile"
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=v_prob_pred, y=v_prob_true, mode="lines+markers", name=model_name)
    )

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfectly calibrated",
            line=dict(dash="dash", color="grey"),
        )
    )

    fig.update_layout(
        title="Reliability Curve",
        xaxis_title="Mean predicted probability",
        yaxis_title="Fraction of positives",
        legend_title="Model",
        autosize=False,
    )

    return fig.show(fig_type)


def beta_confidence_analysis(alpha, beta_param, fig_type=None):
    """
    Plot the Beta Probability Density Function (PDF) with filled area under the curve.

    This function creates an interactive visualization of the Beta distribution's PDF to assess
    model reliability. The distribution represents the balance between
    successes (alpha) and failures (beta) to help evaluate deployment confidence.

    Parameters
    -----------
    alpha : int or float
        The alpha (α) parameter representing model successes/correct predictions. Must be positive.
    beta_param : int or float
        The beta (β) parameter representing model failures/incorrect predictions. Must be positive.
    fig_type : str, optional
        Display type for the figure (particularly useful in Jupyter notebooks).
        Common options: None (default), 'notebook', or other Plotly-supported renderers.
        (default: None)

    Returns
    --------
    None
        Displays the Beta PDF plot directly.

    Notes
    ------
    - The Beta distribution is defined on the interval [0, 1]
    - Use the distribution shape and position to inform
      deployment decisions and risk assessment

    Raises
    -------
    ValueError
        If alpha or beta_param are not positive values.
    """

    if alpha <= 0 or beta_param <= 0:
        raise ValueError("Alpha and Beta parameters must be positive.")

    X = np.linspace(0, 1, 1000)
    y_pdf = beta.pdf(X, alpha, beta_param)

    fill_indices = (X >= 0) & (X <= 1)
    x_fill = X[fill_indices]
    y_pdf_fill = y_pdf[fill_indices]

    trace_pdf = go.Scatter(
        x=X, y=y_pdf, mode="lines", name=f"Beta PDF(α={alpha}, β={beta_param})"
    )

    trace_fill = go.Scatter(
        x=x_fill,
        y=y_pdf_fill,
        mode="lines",
        fill="tozeroy",
        fillcolor="rgba(255,0,0,0.2)",
        line=dict(width=0),
        hoverinfo="skip",
        showlegend=False,
    )

    layout = go.Layout(
        title=f"Beta PDF (α={alpha}, β={beta_param})",
        xaxis=dict(
            title="Success Probability (x)",
            range=[
                min(x_fill[np.nonzero(y_pdf_fill)]),
                max(x_fill[np.nonzero(y_pdf_fill)]),
            ],
        ),
        yaxis=dict(title="Probability Density Function (PDF)"),
        width=800,
        height=450,
        hovermode="x unified",
    )

    fig = go.Figure(data=[trace_pdf, trace_fill], layout=layout)

    return fig.show(renderer=fig_type)


def confusion_matrix(
    clf: ClassifierMixin,
    X: np.ndarray,
    y: np.ndarray,
    fig_type=None,
    percentage_by_class=True,
):
    """
    Generate an annotated confusion matrix heatmap for a binary classifier.

    This function creates an interactive visualization of the confusion matrix,
    displaying the counts of true positives, true negatives, false positives,
    and false negatives. The matrix can show percentages either by class or
    as overall percentages of the total predictions.

    Parameters
    -----------
    clf : ClassifierMixin
        The trained classifier model that implements predict method.
    X : np.ndarray
        Input feature data for evaluation.
    y : np.ndarray
        True binary labels (0 or 1).
    fig_type : str, optional
        Display type for the figure (particularly useful in Jupyter notebooks).
        Common options: None (default), 'notebook', or other Plotly-supported renderers.
        (default: None)
    percentage_by_class : bool, optional
        If True, displays percentages by class (each column sums to 100%).
        If False, displays overall percentages (all cells sum to 100%).
        (default: True)

    Returns
    --------
    None
        Displays the confusion matrix heatmap directly.

    Notes
    ------
    - The matrix follows the format: rows represent true labels, columns represent predicted labels
    - TN: True Negative, FP: False Positive, FN: False Negative, TP: True Positive
    - Color intensity represents the magnitude of values in each cell
    """

    y_pred = clf.predict(X)
    tn, fp, fn, tp = metrics.confusion_matrix(y, y_pred).ravel()
    labels = np.array([["FN", "TN"], ["TP", "FP"]])
    cm = np.array([[fn, tn], [tp, fp]])

    if percentage_by_class:
        total = cm.sum(axis=0)
        percentage = cm / total * 100
    else:
        percentage = cm / np.sum(cm) * 100

    annotation_text = np.empty_like(percentage, dtype="U10")

    for i in range(percentage.shape[0]):
        for j in range(percentage.shape[1]):
            annotation_text[i, j] = f"{labels[i, j]} {percentage[i, j]:.2f}"

    fig = ff.create_annotated_heatmap(
        cm,
        x=["Positive", "Negative"],
        y=["Negative", "Positive"],
        colorscale="Blues",
        hoverinfo="z",
        annotation_text=annotation_text,
    )

    fig.update_layout(width=400, height=400, title="Confusion Matrix")
    return fig.show(fig_type)


def score_distribution(
    clf: ClassifierMixin,
    X: np.ndarray,
    nbins=15,
    fig_type=None,
):
    """
    Generate a histogram showing the distribution of predicted scores for a classifier.

    This function creates an interactive visualization displaying the distribution of
    predicted probabilities from a binary classifier. The histogram helps identify
    patterns in model predictions and potential calibration issues.

    Parameters
    ----------
    clf : ClassifierMixin
        The trained classifier model that implements predict_proba method.
    X : np.ndarray
        Input feature data for evaluation.
    nbins : int, optional
        Number of bins for the histogram (default: 15).
    fig_type : str, optional
        Display type for the figure (particularly useful in Jupyter notebooks).
        Common options: None (default), 'notebook', or other Plotly-supported renderers.
        (default: None)

    Returns
    -------
    None
        Displays the histogram plot directly.

    Notes
    -----
    - Shows the distribution of predicted probabilities for the positive class
    - Useful for understanding model confidence patterns
    - Can help identify potential calibration issues
    - Well-calibrated models typically show varied probability distributions
    """
    y_prob = clf.predict_proba(X)[:, 1]
    fig = px.histogram(y_prob, nbins=nbins)
    fig.update_layout(
        title="Histogram of Predicted Scores",
        xaxis_title="Predicted Scores",
        yaxis_title="Count",
        legend_title="Modelos",
        autosize=False,
    )
    fig.update_layout(hovermode="x")
    fig.update_traces(hovertemplate="%{y}")
    fig.update_layout(showlegend=False)
    return fig.show(fig_type)
