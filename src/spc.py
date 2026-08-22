import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import math
import matplotlib.pyplot as plt


class SPCEngine:
    """Statistical Process Control engine with capability indices, control charts, and WECO rules.

    Usage:
        spc = SPCEngine(list_of_dicts)
        cp, cpk, pp, ppk = spc.calculate_capability('outer_area', target, tol)
        summary = spc.summary('outer_area', target, tol)
    """

    def __init__(self, samples: List[Dict[str, Any]] = None):
        self.samples = samples or []
        self.df = pd.DataFrame(self.samples)

    def add_samples(self, samples: List[Dict[str, Any]]):
        self.samples.extend(samples)
        self.df = pd.DataFrame(self.samples)

    def _get_series(self, key: str) -> List[float]:
        if self.df.empty or key not in self.df.columns:
            return []
        vals = self.df[key].dropna().astype(float).tolist()
        return vals

    def mean_std(self, key: str) -> Tuple[float, float]:
        vals = self._get_series(key)
        if not vals:
            return 0.0, 0.0
        arr = np.array(vals)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1) if len(arr) > 1 else 0.0)
        return mean, std

    def calculate_capability(self, key: str, target: float, tol: float) -> Tuple[float, float, float, float]:
        vals = self._get_series(key)
        if not vals:
            return 0.0, 0.0, 0.0, 0.0
        arr = np.array(vals)
        mean = float(arr.mean())
        sigma_sample = float(arr.std(ddof=1) if len(arr) > 1 else 0.0)
        sigma_pop = float(arr.std(ddof=0))

        USL = target + tol
        LSL = target - tol

        cp = ((USL - LSL) / (6.0 * sigma_sample)) if sigma_sample > 0 else 0.0
        cpk = min(((USL - mean) / (3.0 * sigma_sample) if sigma_sample > 0 else 0.0),
                  ((mean - LSL) / (3.0 * sigma_sample) if sigma_sample > 0 else 0.0))

        pp = ((USL - LSL) / (6.0 * sigma_pop)) if sigma_pop > 0 else 0.0
        ppk = min(((USL - mean) / (3.0 * sigma_pop) if sigma_pop > 0 else 0.0),
                   ((mean - LSL) / (3.0 * sigma_pop) if sigma_pop > 0 else 0.0))

        return cp, cpk, pp, ppk

    def control_limits(self, key: str, k: float = 3.0) -> Dict[str, float]:
        mean, sigma = self.mean_std(key)
        return {"mean": mean, "ucl": mean + k * sigma, "lcl": mean - k * sigma}

    def apply_weco_rules(self, key: str) -> List[str]:
        vals = self._get_series(key)
        signals: List[str] = []
        if not vals:
            return signals
        mean, sigma = self.mean_std(key)
        if sigma == 0:
            return signals

        n = len(vals)

        # Rule 1: One point beyond 3σ
        for i, v in enumerate(vals):
            if abs(v - mean) > 3 * sigma:
                signals.append(f"Rule1: point {i} beyond 3σ (value={v:.3f})")

        # Rule 2: Two of three consecutive points beyond 2σ on same side
        for i in range(n - 2):
            window = vals[i:i + 3]
            pos = [w - mean for w in window]
            cnt_pos = sum(1 for p in pos if p > 2 * sigma)
            cnt_neg = sum(1 for p in pos if p < -2 * sigma)
            if cnt_pos >= 2:
                signals.append(f"Rule2: window {i}-{i+2} two of three >+2σ")
            if cnt_neg >= 2:
                signals.append(f"Rule2: window {i}-{i+2} two of three <-2σ")

        # Rule 3: Four of five beyond 1σ on same side
        for i in range(n - 4):
            window = vals[i:i + 5]
            pos = [w - mean for w in window]
            cnt_pos = sum(1 for p in pos if p > 1 * sigma)
            cnt_neg = sum(1 for p in pos if p < -1 * sigma)
            if cnt_pos >= 4:
                signals.append(f"Rule3: window {i}-{i+4} four of five >+1σ")
            if cnt_neg >= 4:
                signals.append(f"Rule3: window {i}-{i+4} four of five <-1σ")

        # Rule 4: Nine consecutive on same side of mean
        for i in range(n - 8):
            window = vals[i:i + 9]
            if all(w > mean for w in window):
                signals.append(f"Rule4: 9 points {i}-{i+8} above mean")
            if all(w < mean for w in window):
                signals.append(f"Rule4: 9 points {i}-{i+8} below mean")

        return signals

    def summary(self, key: str, target: float, tol: float) -> Dict[str, Any]:
        mean, sigma = self.mean_std(key)
        cp, cpk, pp, ppk = self.calculate_capability(key, target, tol)
        limits = self.control_limits(key)
        weco = self.apply_weco_rules(key)
        return {
            "mean": mean,
            "std": sigma,
            "cp": cp,
            "cpk": cpk,
            "pp": pp,
            "ppk": ppk,
            "ucl": limits["ucl"],
            "lcl": limits["lcl"],
            "weco_signals": weco,
        }

    def plot_xbar_r_chart(self, metric: str, target: float, tolerance: float):
        if self.df.empty or metric not in self.df.columns:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No Data", ha='center')
            return fig

        data = self.df[metric].dropna().values
        subgroup_size = 2
        n_groups = len(data) // subgroup_size
        if n_groups == 0:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Not enough data for subgroups (min 2)", ha='center')
            return fig

        data = data[:n_groups * subgroup_size].reshape((n_groups, subgroup_size))
        x_bars = np.mean(data, axis=1)
        r_vals = np.ptp(data, axis=1)

        x_grand_mean = np.mean(x_bars)
        r_bar = np.mean(r_vals)

        # A2, D3, D4 constants for n=2
        A2 = 1.880
        D3 = 0
        D4 = 3.267

        ucl_x = x_grand_mean + A2 * r_bar
        lcl_x = x_grand_mean - A2 * r_bar
        ucl_r = D4 * r_bar
        lcl_r = D3 * r_bar

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        ax1.plot(x_bars, marker='o', linestyle='-', color='blue')
        ax1.axhline(x_grand_mean, color='green', linestyle='--', label='Mean')
        ax1.axhline(ucl_x, color='red', linestyle='-', label='UCL')
        ax1.axhline(lcl_x, color='red', linestyle='-', label='LCL')
        ax1.axhline(target + tolerance, color='orange', linestyle=':', label='USL')
        ax1.axhline(target - tolerance, color='orange', linestyle=':', label='LSL')
        ax1.set_title(f'X-Bar Chart ({metric})')
        ax1.legend()

        ax2.plot(r_vals, marker='o', linestyle='-', color='purple')
        ax2.axhline(r_bar, color='green', linestyle='--', label='R-Bar')
        ax2.axhline(ucl_r, color='red', linestyle='-', label='UCL')
        ax2.axhline(lcl_r, color='red', linestyle='-', label='LCL')
        ax2.set_title(f'R Chart ({metric})')
        ax2.legend()

        plt.tight_layout()
        return fig

    def plot_pareto(self):
        if self.df.empty or 'classification' not in self.df.columns:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No Data", ha='center')
            return fig

        counts = self.df['classification'].value_counts()
        if 'Good Part' in counts.index:
            counts = counts.drop('Good Part')

        if counts.empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No Defects Recorded", ha='center')
            return fig

        cum_percent = counts.cumsum() / counts.sum() * 100

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.bar(counts.index, counts.values, color='C0')
        ax1.set_ylabel('Frequency', color='C0')
        ax1.tick_params(axis='y', labelcolor='C0')
        ax1.tick_params(axis='x', rotation=45)

        ax2 = ax1.twinx()
        ax2.plot(counts.index, cum_percent.values, color='C1', marker='D', ms=7)
        ax2.set_ylabel('Cumulative %', color='C1')
        ax2.tick_params(axis='y', labelcolor='C1')
        ax2.set_ylim([0, 105])

        plt.title('Defect Pareto Chart')
        plt.tight_layout()
        return fig
