import plotly.graph_objects as go
import numpy as np
import plotly.express as px
import scipy.stats
import pandas as pd
from typing import Union


class Plot:
    def __init__(self, statistics, distribution, confidence_interval):
        self.confidence_interval = confidence_interval
        self.statistics = statistics
        self.distribution = distribution

    def _update_layout(
        self,
        title: str,
        xaxis_title: str,
        yaxis_title: str,
        width: int,
        height: int,
    ):
        """
        Helper function to update layout settings for the plots.
        """
        return dict(
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            width=width,
            height=height,
            showlegend=True,
            bargap=0,
            bargroupgap=0,
        )

    def _add_limits(self, fig):
        """
        Helper function to add the lower and upper limits and the mean line to the plot.
        """
        lower_limit, upper_limit = self.statistics.get(
            "lower_limit"
        ), self.statistics.get("upper_limit")
        if lower_limit:
            fig.add_hline(
                y=lower_limit,
                line_dash="dash",
                line_color="firebrick",
                name="Lower Limit",
                opacity=0.5,
            )
        if upper_limit:
            fig.add_hline(
                y=upper_limit,
                line_dash="dash",
                line_color="firebrick",
                name="Upper Limit",
                opacity=0.5,
            )

        fig.add_hline(
            y=self.statistics["mean"],
            line_dash="dash",
            line_color="darkslateblue",
            opacity=0.3,
            name="Mean",
        )

    def kde(
        self,
        width: int = 600,
        height: int = 400,
        fig_type: str = None,
    ):
        """
        Generate a Kernel Density Estimate (KDE) plot for the distribution's metric.
        """
        x_vals = np.linspace(self.distribution.min(), self.distribution.max(), 1000)
        kde = scipy.stats.gaussian_kde(self.distribution)

        # Create KDE plot using plotly.express
        fig = px.line(x=x_vals, y=kde(x_vals))
        fig.update_layout(
            self._update_layout(
                "Distribution of metric with Kernel Density Estimate (KDE)",
                "Metric",
                "Density",
                width,
                height,
            )
        )

        return fig.show(fig_type)

    def bar(
        self,
        analysis: pd.DataFrame,
        width: int = 800,
        height: int = 400,
        fig_type: str = None,
    ):
        """
        Generate a diverging bar plot showing metric over time relative to a reference line.
        """
        reference_line = self.statistics["mean"]
        positive_bars = np.maximum(analysis - reference_line, 0)
        negative_bars = np.maximum(reference_line - analysis, 0)

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=analysis.get("datetime"),
                y=positive_bars,
                base=[reference_line] * len(positive_bars),
                name="Above Reference",
                marker_color="lightslategrey",
                customdata=analysis.loc[analysis >= reference_line],
                hovertemplate="(%{x},%{y:.3f})",
                opacity=0.7,
            )
        )

        fig.add_trace(
            go.Bar(
                x=analysis.get("datetime"),
                y=negative_bars,
                base=reference_line - negative_bars,
                name="Below Reference",
                marker_color="crimson",
                customdata=analysis.loc[analysis < reference_line],
                hovertemplate="(%{x},%{base:.3f})",
                opacity=0.7,
            )
        )
        if self.confidence_interval:
            fig.add_hrect(
                y0=self.statistics.get("ci_lower"),
                y1=self.statistics.get("ci_upper"),
                line_width=0,
                fillcolor="lightblue",
                opacity=0.5,
                name="Fixed Confidence Interval",
            )

        self._add_limits(fig)

        fig.update_layout(
            self._update_layout(
                "Metric Over Time with Fixed Confidence Interval",
                "Time",
                "Metric",
                width,
                height,
            )
        )

        return fig.show(fig_type)

    def scatter(
        self,
        analysis: Union[pd.Series, np.ndarray, list],
        width: int = 800,
        height: int = 400,
        fig_type: str = None,
    ):
        """
        Generate a time-series plot showing the metric performance with confidence interval and thresholds.
        """

        upper_limit = self.statistics.get("upper_limit")
        lower_limit = self.statistics.get("lower_limit")
        index = (
            analysis.index
            if isinstance(analysis, pd.Series)
            else list(range(len(analysis)))
        )

        def marker_color(y):
            if (upper_limit is not None and y > upper_limit) or (
                lower_limit is not None and y < lower_limit
            ):
                return "firebrick"
            return "#1f77b4"

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=index,
                y=analysis,
                mode="markers",
                name="Metric",
                marker=dict(color=[marker_color(row) for row in analysis]),
            ),
        )

        if self.confidence_interval:
            fig.add_trace(
                go.Scatter(
                    x=index,
                    y=[self.statistics["ci_lower"], self.statistics["ci_upper"]],
                    fill="toself",
                    fillcolor="rgba(0, 100, 255, 0.2)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="Fixed Confidence Interval",
                )
            )

            fig.add_hrect(
                y0=self.statistics["ci_lower"],
                y1=self.statistics["ci_upper"],
                line_width=0,
                fillcolor="lightblue",
                opacity=0.5,
            )

        self._add_limits(fig)

        fig.update_layout(
            self._update_layout(
                "Metric Over Time with Fixed Confidence Interval",
                "Time",
                "Metric",
                width,
                height,
            )
        )

        return fig.show(fig_type)
