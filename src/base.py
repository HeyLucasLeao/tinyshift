from . import plot, threshold, stats


class BaseModel:
    def __init__(
        self,
        reference_distribution,
        confidence_level,
        statistic,
        n_resamples,
        random_state,
        drift_limit,
    ):
        self.statistics = stats.generate(
            reference_distribution,
            confidence_level,
            statistic,
            n_resamples,
            random_state,
        )
        self.plot = plot.Plot(self.statistics, reference_distribution)
        threshold.generate(self.statistics, reference_distribution, drift_limit)
