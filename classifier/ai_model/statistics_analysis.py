import numpy as np
from scipy import stats

class StatisticalAnalysis:

    @staticmethod
    def correlation(x, y):
        r, p = stats.pearsonr(x, y)
        return {"correlation": r, "p_value": p}

    @staticmethod
    def linear_regression(x, y):
        slope, intercept, r, p, std_err = stats.linregress(x, y)
        return {
            "slope": slope,
            "intercept": intercept,
            "r_value": r,
            "p_value": p,
            "std_error": std_err
        }

    @staticmethod
    def anova(*groups):
        """Однофакторна ANOVA"""
        f_stat, p_value = stats.f_oneway(*groups)
        return {"F_statistic": f_stat, "p_value": p_value}

    @staticmethod
    def fisher_test(x, y):
        slope, intercept, r, p, std_err = stats.linregress(x, y)
        n = len(x)
        f_value = (r**2 / (1 - r**2)) * (n - 2)
        return {"F_value": f_value, "p_value": p}

    @staticmethod
    def confidence_interval(x, y, confidence=0.95):
        """95% довірчий інтервал"""
        slope, intercept, r, p, std_err = stats.linregress(x, y)
        n = len(x)
        t_value = stats.t.ppf((1 + confidence) / 2, n - 2)
        ci_slope = (slope - t_value * std_err, slope + t_value * std_err)
        return {"slope_confidence_interval": ci_slope}
