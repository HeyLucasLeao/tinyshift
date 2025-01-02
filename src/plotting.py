import plotly.graph_objects as go
import numpy as np
import plotly.express as px
import scipy.stats


def kde(data, column_name, width=600, height=400, fig_type=None):
    """
    This function generates a Kernel Density Estimate (KDE) plot for a given column in the DataFrame.
    """
    # Create x values for the plot based on the range of the column
    x_vals = np.linspace(data[column_name].min(), data[column_name].max(), 1000)

    # Compute the Kernel Density Estimate (KDE) of the selected column
    kde = scipy.stats.gaussian_kde(data[column_name])

    # Create the KDE plot
    fig = px.line(x=x_vals, y=kde(x_vals))

    # Customize the layout of the plot
    fig.update_layout(
        title=f"Distribution of {column_name} with Kernel Density Estimate (KDE)",
        xaxis_title="Metric",
        yaxis_title="Density",
        width=width,
        height=height,
    )

    # Display the plot
    return fig.show(fig_type)


def performance_by_time(reference, analysis, width=800, height=400, fig_type=None):
    """
    This function generates a time-series plot that shows the performance of a metric over time
    with a fixed confidence interval and threshold lines.

    Parameters:
    - reference: A DataFrame containing the reference values such as confidence interval bounds,
                 upper and lower thresholds, and mean.
    - analysis: A DataFrame containing the time-series data with datetime and metric values.

    The plot includes:
    - A line for the metric values over time.
    - A shaded area representing the fixed confidence interval.
    - Horizontal lines indicating the upper and lower thresholds, and the mean value.
    """
    # Creating the plot
    fig = go.Figure()

    # Adding the metric values as a line with markers
    fig.add_trace(
        go.Scatter(
            x=analysis["datetime"],
            y=analysis["metric"],
            mode="lines+markers",
            name="Metric",
        )
    )

    # Adding the fixed confidence interval (shaded area)
    fig.add_trace(
        go.Scatter(
            x=analysis["datetime"],
            y=[reference["ci_lower"], reference["ci_upper"]],
            fill="toself",
            fillcolor="rgba(0, 100, 255, 0.2)",  # Blue with opacity
            line=dict(color="rgba(255,255,255,0)"),  # No border line
            name="Fixed Confidence Interval",
        )
    )

    fig.add_hrect(
        y0=reference["ci_lower"],
        y1=reference["ci_upper"],
        line_width=0,
        fillcolor="lightblue",
        opacity=0.5,
    )

    # Adding horizontal lines for thresholds and mean
    fig.add_hline(
        y=reference["upper_threshold"],
        line_dash="dash",
        line_color="firebrick",
        name="Upper Threshold",
        opacity=0.5,
    )

    fig.add_hline(
        y=reference["lower_threshold"],
        line_dash="dash",
        line_color="firebrick",
        name="Lower Threshold",
        opacity=0.5,
    )

    fig.add_hline(
        y=reference["mean"],
        line_dash="dash",
        line_color="darkslateblue",
        opacity=0.3,
        name="Mean",
    )

    # Adding labels and title
    fig.update_layout(
        title="Metric Over Time with Fixed Confidence Interval",
        xaxis_title="Time",
        yaxis_title="Metric",
        showlegend=True,
        width=width,
        height=height,
    )

    fig.update_layout(hovermode="x")
    fig.update_traces(hovertemplate="%{y}")

    # Display the plot
    return fig.show(fig_type)
