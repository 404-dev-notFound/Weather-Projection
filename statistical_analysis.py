import pymannkendall as mk
import numpy as np
import matplotlib.pyplot as plt

class ClimateTrendAnalyzer:
    def __init__(self):
        """
        Analyzer for validating long-term trends in climate data using non-parametric statistics.
        Designed to support Extreme Precipitation and Urban Heat Island research papers.
        """
        pass

    def run_mk_test(self, time_series_data, alpha=0.05):
        """
        Runs the standard Mann-Kendall test on a 1D time series to detect monotonic trends.
        
        Args:
            time_series_data: list or numpy array of values over time.
            alpha: Significance level (default 0.05 for 95% confidence).
            
        Returns:
            Dictionary containing trend direction, p-value, and the Z-statistic.
            (Positive Z = Increasing trend, Negative Z = Decreasing trend).
        """
        result = mk.original_test(time_series_data, alpha=alpha)
        return {
            'trend': result.trend, 
            'p_value': result.p,
            'z_stat': result.z,
            'slope': result.slope
        }

    def run_sqmk_test(self, time_series_data, years_list, title="Sequential MK Trend Mutation Analysis"):
        """
        Runs the Sequential Mann-Kendall (SQ-MK) test to identify chronological mutation points
        (abrupt structural shifts in the climate data).
        
        Args:
            time_series_data: list or 1D numpy array of values (e.g., annual precipitation peaks).
            years_list: list of years corresponding to the data for plotting.
            title: Title for the generated output plot.
            
        Returns:
            Dictionary containing forward stats, backward stats, identified mutation years, and the plot figure.
        """
        n = len(time_series_data)
        
        def _calc_seq_mk(data):
            """Helper function to calculate the sequential standard normal deviate u(t)."""
            sk = 0
            u = np.zeros(len(data))
            for i in range(1, len(data)):
                for j in range(i):
                    if data[i] > data[j]:
                        sk += 1
                    elif data[i] < data[j]:
                        sk -= 1
                
                # Expected value and variance of sk based on MK theory
                E_sk = i * (i + 1) / 4.0
                Var_sk = i * (i + 1) * (2 * i + 5) / 72.0
                
                # Standard normal deviate
                if Var_sk > 0:
                    u[i] = (sk - E_sk) / np.sqrt(Var_sk)
                else:
                    u[i] = 0
            return u

        # Forward Sequence u(t)
        u_f = _calc_seq_mk(time_series_data)
        
        # Backward Sequence u'(t) (calculated logically from end to beginning)
        u_b_rev = _calc_seq_mk(time_series_data[::-1])
        u_b = -1 * u_b_rev[::-1] 
        
        # --- Plotting the Sequence ---
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(years_list, u_f, label="Forward Sequence $u(t)$", color='blue', linewidth=2)
        ax.plot(years_list, u_b, label="Backward Sequence $u'(t)$", color='red', linestyle='--', linewidth=2)
        
        # 95% Confidence Limits (alpha=0.05 -> z_critical = +/- 1.96)
        ax.axhline(y=1.96, color='black', linestyle=':', label='95% Confidence Bounds')
        ax.axhline(y=-1.96, color='black', linestyle=':')
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Sequential Statistic Value', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.5)
        plt.tight_layout()
        
        # --- Mutation Point Detection ---
        # A significant structural shift occurs where the forward and backward lines cross
        mutation_points = []
        for i in range(1, len(u_f)):
            if (u_f[i-1] - u_b[i-1]) * (u_f[i] - u_b[i]) < 0: # Intersection detected
                # Check if intersection happens outside the confidence bounds (significance)
                if abs(u_f[i]) > 1.96 or abs(u_b[i]) > 1.96: 
                    mutation_points.append(years_list[i])
                    
        return {
            'forward_stats': u_f,
            'backward_stats': u_b,
            'significant_mutation_years': mutation_points,
            'plot_fig': fig
        }

    def analyze_spatial_grid(self, spatiotemporal_cube):
        """
        Runs the MK trend test for every individual pixel in a 17x17 grid over time,
        producing a complete map of statistical warming/drying gradients.
        
        Args:
            spatiotemporal_cube: numpy array of shape (Time_Years, Height, Width) 
                                 e.g., (75, 17, 17) containing annual averages/sums.
        Returns:
            z_grid: Map of Z-statistics.
            p_grid: Map of p-values (significance).
        """
        T, H, W = spatiotemporal_cube.shape
        z_grid = np.zeros((H, W))
        p_grid = np.zeros((H, W))
        
        for i in range(H):
            for j in range(W):
                ts = spatiotemporal_cube[:, i, j]
                # Small safety check for NaN clusters
                if np.isnan(ts).all():
                    z_grid[i, j], p_grid[i, j] = np.nan, np.nan
                    continue
                
                result = mk.original_test(ts)
                z_grid[i, j] = result.z
                p_grid[i, j] = result.p
                
        return z_grid, p_grid

if __name__ == "__main__":
    analyzer = ClimateTrendAnalyzer()
    print("Statistical Trend Analyzer initialized.")
    print("Ready to process AI-downscaled temporal arrays for MK and SQ-MK validation.")
